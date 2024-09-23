#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   trl_stackllama.py
@Time   :   2024/09/11 15:08:29
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用stackoverflow数据、RLHF微调llama3.1
            流程：
                https://huggingface.co/blog/stackllama
            SFT QLora:
                https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/instruction-tune-llama-2-int4.ipynb
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import argparse
import os
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset.
    """
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token
    )
    return train_dataset, valid_dataset


def run_training(
    args,
    train_data: ConstantLengthDataset,
    val_data:ConstantLengthDataset
):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0
    # training_args = TrainingArguments(
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        packing=True,
        eval_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=0.3,
        weight_decay=args.weight_decay,
        run_name="llama3.1-8b-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=not args.gradient_checkpointing,
        disable_tqdm=False, # disable tqdm since with packing values are in correct
        # 和training_args.distributed_state.distributed_type同步设置，否则Accelerator()报错
        # deepspeed="/home/yhao/code/learning/AIGC/Lllama3/trainer_deepspeed.json"
    )
    # training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index},
        use_cache=False,
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model) # transformers...LlamaForCausalLM dtype=float32
    model = get_peft_model(model, lora_config)     # peft...PeftModelForCausalLM     dtype=float32

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()
    print("Training...")
    trainer.train(resume_from_checkpoint=args.output_dir + "/checkpoint-1708/")
    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def handle_sft(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


def handle_rm(args):
    pass


def handle_rlhf(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="trl with rlhf pipeline")

    sft_parser = subparsers.add_parser(name="sft", help="supervised finetune")
    sft_parser.add_argument("--model_path", type=str, default="/media/data/LLMS/Llama3-hf")
    sft_parser.add_argument("--dataset_name", type=str, default=datafolder)
    sft_parser.add_argument("--subset", type=str, default="data/finetune")
    sft_parser.add_argument("--split", type=str, default="train")
    sft_parser.add_argument("--size_valid_set", type=int, default=4000)
    sft_parser.add_argument("--streaming", action="store_true")
    sft_parser.add_argument("--shuffle_buffer", type=int, default=5000)
    sft_parser.add_argument("--seed", type=int, default=0)
    sft_parser.add_argument("--num_workers", type=int, default=None)
    sft_parser.add_argument("--seq_length", type=int, default=1024)

    sft_parser.add_argument("--output_dir", type=str, default=f"{projfolder}/checkpoints")
    sft_parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    sft_parser.add_argument("--batch_size", type=int, default=4)
    sft_parser.add_argument("--max_steps", type=int, default=10000)
    sft_parser.add_argument("--eval_freq", default=1000, type=int)
    sft_parser.add_argument("--save_freq", default=1000, type=int)
    sft_parser.add_argument("--log_freq", default=1, type=int)
    sft_parser.add_argument("--learning_rate", type=float, default=1e-4)
    sft_parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    sft_parser.add_argument("--num_warmup_steps", type=int, default=100)
    sft_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    sft_parser.add_argument("--weight_decay", type=float, default=0.05)

    sft_parser.add_argument("--fp16", action="store_true", default=False)
    sft_parser.add_argument("--bf16", action="store_true", default=False)
    

    sft_parser.set_defaults(func=handle_sft)


    rm_parser = subparsers.add_parser(name="rm", help="reward model")


    rlhf_parser = subparsers.add_parser(name="rlhf", help="reinforcement learning")


    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    projfolder = "/media/data/LLMS/StackLlama"
    datafolder = f"{projfolder}/stack-exchange-paired"
    main()