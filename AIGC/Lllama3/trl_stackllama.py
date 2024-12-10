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
from functools import partial
from dataclasses import dataclass, field  # 早期版本的Pydantic
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers import AutoModelForSequenceClassification, Trainer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import BitsAndBytesConfig
from transformers import Adafactor, pipeline
import evaluate
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from peft import PeftConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


def handle_merge(args):
    """Merge peft adapter to model.
    """
    peft_config = PeftConfig.from_pretrained(args.adapter_model_name)
    if peft_config.task_type == "SEQ_CLS":
        # The sequence classification task is used for the reward model in PPO
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    # Load the PEFT model
    model = PeftModel.from_pretrained(model, args.adapter_model_name)
    model.eval()
    model = model.merge_and_unload()

    model.save_pretrained(f"{args.output_name}")
    tokenizer.save_pretrained(f"{args.output_name}")


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


def create_sft_datasets(tokenizer, args):
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


def run_sft_training(
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
    train_dataset, eval_dataset = create_sft_datasets(tokenizer, args)
    run_sft_training(args, train_dataset, eval_dataset)


def preprocess_rm_function(examples, tokenizer):
    """Turn the dataset into pairs of post + summaries
    where text_j is the preferred question + answer and text_k is the other.
    Then tokenize the dataset

    Args:
        examples: datasets.formatting.formatting.LazyBatch
                  examples.data: {'qid': [id1, ..., id6], 'question': [q1, ..., q6]}
    """
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)
        
        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


def create_rm_datasets(tokenizer, args):
    train_dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        verification_mode="no_checks"
    )
    columns_to_remove = train_dataset.column_names
    if args.train_subset > 0:
        train_dataset = train_dataset.select(range(args.train_subset))
    
    train_dataset = train_dataset.map(
        partial(preprocess_rm_function, tokenizer=tokenizer),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= args.max_length and len(x["input_ids_k"]) <= args.max_length,
        num_proc=args.num_workers
    )

    eval_dataset = load_dataset(
        args.dataset_name,
        data_dir="data/evaluation",
        split=args.split,
        verification_mode="no_checks"
    )
    if args.eval_subset:
        eval_dataset = eval_dataset.select(range(args.eval_subset))
    eval_dataset = eval_dataset.map(
        partial(preprocess_rm_function, tokenizer=tokenizer),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=columns_to_remove
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= args.max_length and len(x["input_ids_k"]) <= args.max_length,
        num_proc=args.num_workers
    )

    return train_dataset, eval_dataset


def compute_rm_metrics(eval_pred):
    """
    """
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    # 这里对应了模型的num_labels=1
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)


"""
1、装饰器装饰class
2、class形式的data_collator
是不是因为dataclass装饰器默认添加的类方法
"""
@dataclass
class RewardDataCollatorWithPadding:
    """与Dataset.collate_fn差不多，主要就是pad和batch
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """伪函数，在Trainer中传入，实际上能()就行
        """
        features_j = []
        features_k = []
        # batch中的每个元素
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Define how to compute the reward loss.
        We use the InstructGPT pairwise logloss: https://huggingface.co/papers/2203.02155
        
        A > B > C:
            loss = -(r(A) - r(B) + r(A) - r(C) + r(B) - r(C))
        """
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -torch.nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


def run_rm_training(args, train_dataset, eval_dataset, tokenizer):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_CLS,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        # IMPORTANT sigmoid输出的是二分类概率了
        args.model_path, num_labels=1, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not args.gradient_checkpointing

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    model_name_split = args.model_path.split("/")[-1]
    output_name = (
        f"{model_name_split}_peft_stack-exchange-paired_rmts__{args.train_subset}_{args.learning_rate}"
    )
    output_dir = projfolder + "/" + output_name
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=args.bf16,
        logging_strategy="steps",
        logging_steps=args.log_freq,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        # run_name="llama3.1-8b-reward",
        # report_to="wandb",
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_rm_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train(args.resume_from_checkpoint)
    print("Saving last checkpoint of the model")
    model.save_pretrained(output_dir + "/last_checkpoint")


def handle_rm(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = create_rm_datasets(tokenizer, args)
    run_rm_training(args, train_dataset, eval_dataset, tokenizer)


def create_rl_datasets(tokenizer, args):
    def preprocess_function(examples, tokenizer):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    train_dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        verification_mode="no_checks"
    )
    train_dataset = train_dataset.select(range(40000))
    original_columns = train_dataset.column_names
    ds = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=original_columns
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=args.num_workers)
    ds.set_format(type="torch")

    return ds


def rl_data_collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def run_rlhf_training(args, train_dataset, tokenizer):
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        # load_in_8bit=args.load_in_8bit,  # 13188 MiB
        # load_in_4bit=args.load_in_4bit,  # 10354 MiB
        quantization_config=bnb_config,    # 6000  MiB
        device_map={"": current_device},
        peft_config=lora_config,
    )
    optimizer = None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=args.learning_rate,
        )

    config = PPOConfig(
        tracker_project_name=os.environ["WANDB_PROJECT"],
        steps=args.steps,
        model_name=args.model_path,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
    )
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=rl_data_collator,
        optimizer=optimizer
    )

    # We then build the sentiment analysis pipeline using our reward model, passing the
    # model name and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_name,
        device_map={"": current_device},
        model_kwargs={"quantization_config": bnb_config},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )
    if sentiment_pipe.model.config.pad_token_id is None:
        if not isinstance(sentiment_pipe.model.config.eos_token_id, list):
            sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id
        else:
            sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.total_ppo_epochs:
            break
        
        question_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - args.reward_baseline) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch}")


def handle_rlhf(args):
    set_seed(args.seed)
    # tqdm.pandas()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_rl_datasets(tokenizer, args)
    run_rlhf_training(args, train_dataset, tokenizer)


def main():
    # 共享参数
    parser = argparse.ArgumentParser(description="Shared parser", add_help=False)
    parser.add_argument("--model_path", type=str, default="/media/data/LLMS/Llama3-hf")
    parser.add_argument("--dataset_name", type=str, default=datafolder)
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="If you want to resume training where it left off.")
    subparsers = parser.add_subparsers(help="trl with rlhf pipeline")

    # ========================= SFT ==================================
    sft_parser = subparsers.add_parser(name="sft", help="supervised finetune", parents=[parser])
    sft_parser.add_argument("--size_valid_set", type=int, default=4000)
    sft_parser.add_argument("--streaming", action="store_true")
    sft_parser.add_argument("--shuffle_buffer", type=int, default=5000)
    sft_parser.add_argument("--seq_length", type=int, default=1024)
    sft_parser.add_argument("--output_dir", type=str, default=f"{projfolder}/checkpoints")
    sft_parser.add_argument("--max_steps", type=int, default=10000)
    sft_parser.add_argument("--num_warmup_steps", type=int, default=100)

    sft_parser.set_defaults(func=handle_sft)

    # ========================= RM ==================================
    rm_parser = subparsers.add_parser(name="rm", help="reward model", parents=[parser])
    rm_parser.add_argument("--local_rank", type=int, default=-1, help="Used for multi-gpu")
    rm_parser.add_argument("--train_subset", type=int, default=100000, help="The size of the subset of the training data to use")
    rm_parser.add_argument("--eval_subset", type=int, default=50000, help="The size of the subset of the eval data to use")
    rm_parser.add_argument("--max_length", type=int, default=512)
    rm_parser.add_argument("--num_train_epochs", type=int, default=1, help="The number of training epochs for the reward model.")
    rm_parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU.")
    rm_parser.add_argument("--optim", type=str, default="adamw_hf", help="The optimizer to use.")
    
    rm_parser.set_defaults(func=handle_rm)

    # ========================= RL ==================================
    rlhf_parser = subparsers.add_parser(name="rlhf", help="reinforcement learning", parents=[parser])
    rlhf_parser.add_argument("--steps", type=int, default=20000, help="number of epochs")
    rlhf_parser.add_argument("--load_in_8bit", type=bool, default=False, help="whether to load the model in 8bit")
    rlhf_parser.add_argument("--load_in_4bit", type=bool, default=True, help="whether to load the model in 4bit")
    rlhf_parser.add_argument("--reward_model_name", type=str, help="the reward model name")
    rlhf_parser.add_argument("--reward_baseline", type=float, default=0.0, help="a baseline value that is subtracted from the reward")
    rlhf_parser.add_argument("--adafactor", action="store_true")
    rlhf_parser.add_argument("--log_with", type=str, default="", help="use 'wandb' to log with wandb")
    rlhf_parser.add_argument("--mini_batch_size", type=int, default=1, help="the PPO minibatch size")
    rlhf_parser.add_argument("--early_stopping", action="store_true")
    rlhf_parser.add_argument("--target_kl", type=float, default=0.1, help="kl target for early stopping")
    rlhf_parser.add_argument("--ppo_epochs", default=4, help="the number of ppo epochs")
    rlhf_parser.add_argument("--init_kl_coef", type=float, default=0.2, help="Initial KL penalty coefficient (used for adaptive and linear control)")
    rlhf_parser.add_argument("--adap_kl_ctrl", action="store_false")
    rlhf_parser.add_argument("--output_max_length", type=int, default=128, help="maximum length for generation")
    rlhf_parser.add_argument("--output_dir", type=str, default="", help="n steps to save the model")
    rlhf_parser.set_defaults(func=handle_rlhf)

    # ========================= Merge ==================================
    merge_parser = subparsers.add_parser(name="merge", help="merge adapter to model", parents=[parser])
    merge_parser.add_argument("--adapter_model_name", type=str, default=None, help="the adapter name")
    merge_parser.add_argument("--base_model_name", type=str, default=None, help="the base model name")
    merge_parser.add_argument("--output_name", type=str, default=None, help="the merged model name")
    merge_parser.set_defaults(func=handle_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    projfolder = "/media/data/LLMS/StackLlama"
    datafolder = f"{projfolder}/stack-exchange-paired"
    main()