#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   lora_huanhuan.py
@Time   :   2024/09/05 15:15:36
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel


def process_func(example):
    """对json格式的文本dataset进行处理
    """    
    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
            f"现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        add_special_tokens=False # 不在开头加 special_tokens
    )
    response = tokenizer(
        f"{example['output']}<|eot_id|>",
        add_special_tokens=False
    )
    # FIXME 为什么要加，又输入给谁？？？
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 因为eos token咱们也是要关注的所以 补充为1
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # FIXME 这是个啥啊
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def lora_finetune():
    df = pd.read_json(f"{projdir}/huanhuan.json")
    ds = Dataset.from_pandas(df)

    tokenizer.pad_token = tokenizer.eos_token
    messages = [
        {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
        {"role": "user", "content": '你好呀'},
        {"role": "assistant", "content": "你好，我是甄嬛，你有什么事情要问我吗？"},    
    ]
    print(tokenizer.apply_chat_template(messages, tokenize=False))
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenized_id)
    print(tokenizer.decode(tokenized_id[0]['input_ids']))
    print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))

    model = AutoModelForCausalLM.from_pretrained(modelcache, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    print(model.dtype)

    # 配置lora
    config = LoraConfig(
        peft_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8,                  # Lora 秩
        lora_alpha=32,        # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1      # Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 配置训练参数
    args = TrainingArguments(
        output_dir=outputdir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()


def lora_load_chat():
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        modelcache,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(
        model,
        model_id=f"{outputdir}/checkpoint-699"
    )

    prompt = "梅姐姐的孩子到底是不是朕的！"
    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id    
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    projdir = "/media/data/LLMS/LlamaHuanhuan"
    modelcache = "/media/data/LLMS/Llama3-hf"
    outputdir = "/media/data/LLMS/LlamaHuanhuan"
    
    tokenizer = AutoTokenizer.from_pretrained(modelcache, use_fast=False, trust_remote_code=True)

    # lora_finetune()

    lora_load_chat()