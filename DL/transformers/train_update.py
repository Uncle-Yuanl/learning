#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   train_update.py
@Time   :   2023/08/24 11:32:15
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习如何优化训练逻辑
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
# import multiprocessing as mp
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

import sys
sys.path.append("/code/ConceptAINutrition/scripts")
# from dataprocess.data_process import TrainProcesser
from modelm.modules import NutririonConceptDataset, NutririonConceptModel


class TrainProcesser:
    pass


def _wrapper_train_process(args):
    return _train_process(*args)


def _train_process(submodel, datapath, kpi, tokenizer, labelmap):
    trainset = NutririonConceptDataset(
        data_path=datapath,
        kpi=kpi,
        mission='train',
        tokenizer=tokenizer,
        labelmap=labelmap
    )
    validset = NutririonConceptDataset(
        data_path=datapath,
        kpi="",
        mission='valid',
        tokenizer=tokenizer,
        labelmap=labelmap
    )
    trainloader = DataLoader(
        trainset, batch_size=submodel.train_args.get('per_device_train_batch_size', 8),
        shuffle=True, collate_fn=NutririonConceptDataset.collate_fn
    )
    validloader = DataLoader(
        validset, batch_size=submodel.train_args.get('per_device_train_batch_size', 8),
        shuffle=False, collate_fn=NutririonConceptDataset.collate_fn
    )

    ds_config = "/code/ConceptAINutrition/scripts/usage/trainer_deepspeed.json"
    submodel.ort_train(trainloader, validloader=validloader, ds_config=ds_config)

    return submodel.model

def main(kpi, tp, **kwargs):
    baseleaner_pretrained = tp.baseleaner
    baseleaner = AutoModelForSequenceClassification.from_pretrained(
        baseleaner_pretrained,
        num_labels=tp.numlabels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(baseleaner_pretrained)

    submodels = [
        NutririonConceptModel(
            model=baseleaner,
            kpi=kpi,
            cfgpath='/code/ConceptAINutrition/scripts/usage/trainer_config.yaml'
        ) for _ in range(5)
    ]
    for i, sm in enumerate(submodels):
        od = f"/home/yhao/temp/fake_pretrain/s_{i}"
        if not os.path.exists(od):
            os.makedirs(od)
        sm.train_args['output_dir'] = od

    # # each submodel per process
    # poolsize = 1
    # pool = mp.Pool(poolsize)
    # print(f'启动【{poolsize}】个子进程训练submodels......')
    # submodelspool = pool.imap(
    #     func=_wrapper_train_process,
    #     iterable=((submodels[subid], f"{datapath}/{kpi}", f'Ensemble_{subid}', tokenizer, None) for subid in range(5))
    # )
    # submodels_trained = list(submodelspool)
    # pool.close()
    # pool.join()

    # 主进程直接训练
    submodels_trained = [_train_process(submodels[subid], f"{datapath}/{kpi}", f'Ensemble_{subid}', tokenizer, None) for subid in range(2)]

    print()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    country = "GER"
    datapath = "/home/yhao/temp/nutrition_search_total/SEED-1229_SCALE-1"

    kpi = "Need_Desire"

    tp = TrainProcesser()
    tp.baseleaner = "/media/data/pretrained_models/distilbert-base-german-cased"
    tp.numlabels = 4
    main(kpi, tp)
