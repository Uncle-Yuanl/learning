#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   mlflow_log.py
@Time   :   2023/10/11 10:43:32
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
from transformers import TrainerState, TrainerControl
from transformers import DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

import optuna
from optuna.integration.mlflow import MLflowCallback

import sys
sys.path.append("/code/ConceptAINutrition/scripts")
# from dataprocess.data_process import TrainProcesser
from modelm.nutritiondatasets import NutritionConceptDataset
from modelm.modules import NutritionConceptModel

from hpo_optuna import model_init, make_data
from hpo_optuna import optuna_hp_space, my_objective, trainer_metrics


LABEL2ID = {
    " Greate": 0,
    " Good": 1,
    " Bad": 2,
    " Worse": 3,
}


def objective(trial: optuna.trial.FrozenTrial):
    """Trainer.hyperparameter_search好像没办法直接集成mlflow
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    trainloader, validloader = make_data(datapath, kpi, tokenizer, {0:0, 1:0, 2:1, 3:1})

    train_args = TrainingArguments(
        output_dir="/media/data/pretrained_models/temp",
        # learning_rate=2e-5,
        # per_device_train_batch_size=2,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [2, 4, 6, 8]),
        weight_decay=trial.suggest_float("weight_decay", 0.1, 0.9),
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        save_strategy="epoch",
        evaluation_strategy='epoch',
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="maf1",
        greater_is_better=True,
        run_name=os.environ["MLFLOW_RUN_NAME"]
    )

    trainer = Trainer(
        # model=model,
        args=train_args,
        train_dataset=trainloader.dataset,
        eval_dataset=validloader.dataset,
        tokenizer=tokenizer,
        data_collator=NutritionConceptDataset.collate_fn,
        # feature='sequence-classification',
        compute_metrics=trainer_metrics,
        model_init=model_init
    )

    trainer.train()

    metrics = trainer.evaluate(validloader.dataset)

    return metrics["eval_confusion"], metrics["eval_maf1"]


def main():
    mlflc = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        create_experiment=False,
        metric_name=["eval_confusion", "eval_maf1"],
        mlflow_kwargs={
            "experiment_id": os.environ["MLFLOW_EXPERIMENT_ID"],
            # "run_name": "trial",
            "nested": True
        }
    )
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=2, gc_after_trial=True, callbacks=[mlflc])

    best_trials = study.best_trials
    reproduce_res = objective(best_trials[0])
    print(reproduce_res)


if __name__ == "__main__":
    trackuri = "http://localhost:5000"
    experiment_id = 236761747750136088
    os.environ["MLFLOW_TRACKING_URI"] = trackuri
    os.environ["MLFLOW_EXPERIMENT_ID"] = str(experiment_id)
    os.environ["MLFLOW_RUN_NAME"] = "optuna_mlflow_learning"
    # os.environ["MLFLOW_EXPERIMENT_NAME"] = "optuna_learning"

    pretrained = "/media/data/pretrained_models/Distilbert"
    datapath = Path("/home/yhao/temp/nutrition_search_total/SEED-1229_SCALE-1")
    kpi = "CVM"

    main()