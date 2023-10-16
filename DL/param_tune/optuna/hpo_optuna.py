#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   hpo_optuna.py
@Time   :   2023/10/10 15:23:01
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用optuna超参寻优
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import os
import shutil
import re
import json
import srsly
from typing import Union, Optional
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import random
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.integrations import MLflowCallback
from transformers import TrainerState, TrainerControl
from transformers import DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

import mlflow

import sys
sys.path.append("/code/ConceptAINutrition/scripts")
# from dataprocess.data_process import TrainProcesser
from modelm.nutritiondatasets import NutritionConceptDataset
from modelm.modules import NutritionConceptModel
from modelm.nutritionmetrics import dominant_matrix


LABEL2ID = {
    " Greate": 0,
    " Good": 1,
    " Bad": 2,
    " Worse": 3,
}
pretrained = "/media/data/pretrained_models/Distilbert"


def make_data(datapath, kpi, tokenizer, labelmap):
    trainset = NutritionConceptDataset(
        data_path=datapath,
        kpi=kpi,
        mission='train',
        tokenizer=tokenizer,
        labelmap=labelmap
    )
    validset = NutritionConceptDataset(
        data_path=datapath,
        kpi=kpi,
        mission='valid',
        tokenizer=tokenizer,
        labelmap=labelmap
    )
    trainloader = DataLoader(
        trainset, batch_size=8,
        shuffle=True, collate_fn=NutritionConceptDataset.collate_fn
    )
    validloader = DataLoader(
        validset, batch_size=8,
        shuffle=False, collate_fn=NutritionConceptDataset.collate_fn
    )

    return trainloader, validloader


def trainer_metrics(pred: EvalPrediction):
    gtls = pred.label_ids
    prds = pred.predictions.argmax(-1)
    maf1 = f1_score(gtls, prds, average='macro')
    kcomt = confusion_matrix(gtls, prds, labels=list(LABEL2ID.values()))
    pas = dominant_matrix(kcomt, tolerance=True)
    return {
        'pid': os.getpid(),
        'maf1': maf1 if pas else maf1 * 0.1,
        'confusion': int(pas),
        'origin f1': maf1
    }


class HpoCallback(MLflowCallback):
    """Make sure every trial has logged parameters.
    不行诶，trainer会创建一个“self_defined”的run，然而每个trial又会创建，无法改名，且log内容很奇怪
    """
    def __init__(self, model_flag, trainer_best):
        super().__init__()
        # 记录当前Trainer最好的metric trial指标超过就保存
        self.model_flag = model_flag
        self.trainer_best = trainer_best

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        cp = {k: v for k, v in args.__dict__.items() if isinstance(v, (int, float, bool))}
        mlflow.log_params(cp)

    def on_train_end(self, args, state, control, **kwargs):
        # mlflow.log_metric(key="best_maf1", value=max(m.get("eval_maf1", 0) for m in state.log_history))
        mlflow.log_metric(key="best_maf1", value=state.best_metric)
        # save the best model
        if state.best_metric > self.trainer_best:
            self.trainer_best = state.best_metric
            sp = args.output_dir + f"/{self.model_flag}_best"
            shutil.move(state.best_model_checkpoint, sp)
            logger.critical(f"New best {self.model_flag} model saved to {sp}")

        return super().on_train_end(args, state, control, **kwargs)


def my_objective(metrics):
    return metrics["eval_maf1"]


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
    }


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    return model


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    trainloader, validloader = make_data(datapath, kpi, tokenizer, {0:0, 1:0, 2:1, 3:1})

    train_args = ORTTrainingArguments(
        output_dir="/media/data/pretrained_models/temp",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        deepspeed="/code/ConceptAINutrition/scripts/usage/trainer_deepspeed.json",
        save_strategy="epoch",
        evaluation_strategy='epoch',
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="maf1",
        greater_is_better=True,
        run_name="hyperparameter_search"
    )

    trainer = ORTTrainer(
        # model=model,
        args=train_args,
        train_dataset=trainloader.dataset,
        eval_dataset=validloader.dataset,
        tokenizer=tokenizer,
        data_collator=NutritionConceptDataset.collate_fn,
        feature='sequence-classification',
        compute_metrics=trainer_metrics,
        model_init=model_init,
        # callbacks=[HpoCallback("flag", 0)]
    )

    hc = HpoCallback("flag", 0)
    trainer.add_callback(hc)

    best_trial = trainer.hyperparameter_search(
        backend="optuna",
        direction="maximize",
        compute_objective=my_objective,
        n_trials=4,
        hp_space=optuna_hp_space
    )

    print(type(best_trial))  # transformers.trainer_utils.BestRun
    # need to reload
    print(hc.trainer_best)
    print()


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "optuna_learning"
    os.environ["MLFLOW_NESTED_RUN"] = "True"
    # os.environ["MLFLOW_RUN_ID"] = "236761747750136088"  # not work

    datapath = Path("/home/yhao/temp/nutrition_search_total/SEED-1229_SCALE-1")
    kpi = "CVM"

    main()