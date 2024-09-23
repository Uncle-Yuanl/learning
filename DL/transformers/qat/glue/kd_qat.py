#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   kd_qat.py
@Time   :   2024/09/23 10:35:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   尝试在知识蒸馏中加上Quantization-Aware Training
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import numpy as np
import json
from pathlib import Path
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from optimum.intel import (
    OVConfig,
    OVTrainer,
    OVTrainingArguments,
    OVModelForSequenceClassification
)
from nncf.common.utils.os import safe_open


model_id = "/media/data/pretrained_models/Distilbert"
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
save_dir = "/media/data/qat_models/disbert_learing"
dataset = load_dataset("/media/data/datasets/glue/sst2")
dataset = dataset.map(
    lambda examples: tokenizer(examples["sentence"], padding=True), batched=True
)
metric = evaluate.load("/media/data/metrics/glue", "sst2")
def compute_metrics(eval_preds):
    preds = np.argmax(eval_preds.predictions, axis=1)
    return metric.compute(predictions=preds, references=eval_preds.label_ids)

# Load the default quantization configuration detailing the quantization we wish to apply
with safe_open(Path("DL/transformers/qat/glue/bert-base-jpqd.json")) as f:
    compression = json.load(f)
ov_config = OVConfig(compression=compression)
trainer = OVTrainer(
    model=model,
    args=OVTrainingArguments(
        save_dir,
        num_train_epochs=1.0,
        do_train=True,
        do_eval=True,
        distillation_temperature=3,
        distillation_weight=0.9
    ),
    train_dataset=dataset["train"].select(range(300)),
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    ov_config=ov_config,
    task="text-classification",
)
# Train the model while applying quantization
train_result = trainer.train()
metrics = trainer.evaluate()
# Export the quantized model to OpenVINO IR format and save it
trainer.save_model()
model = OVModelForSequenceClassification.from_pretrained(save_dir)