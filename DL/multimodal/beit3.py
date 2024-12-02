#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   beit3.py
@Time   :   2024/11/05 10:23:19
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习beit特征处理过程
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
import re
import json
import srsly
from io import StringIO
from typing import Dict
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torchvision.datasets.folder import default_loader
from timm.models import create_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision import transforms
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
)
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
from timm.data.mixup import Mixup
from randaug import RandomAugment
from transformers import XLMRobertaTokenizer

from modeling_utils import _get_base_config
from modeling_finetune import BEiT3ForVisualQuestionAnswering
from utils import merge_batch_tensors_by_dict_key, get_world_size, get_rank


LABEL2ID = {
    " Greate": 0,
    " Good": 1,
    " Bad": 2,
    " Worse": 3,
}


API2ID = {
    "Value Segment": 0,
    "Lower Mainstream": 1,
    "Upper Mainstream": 2,
    "Premium": 3,
    "Super Premium": 4,
    "Masstige": 5
}


class NutritionConceptDataset(Dataset):
    """Dataset for concept ai

    Placehold for image
    """
    COMPATTERN = '(?<=Company: ).*(?=\nProduct:)'
    PROPATTERN = '(?<=\nProduct: ).*(?=\nAd:)'
    CONPATTERN = '(?<=\nAd:).*(?=\nPrice:)'
    APIPATTERN = '(?<=\nApi:).*(?=\nAspect:)'

    def __init__(
        self,
        image_data_path,
        text_data_path,
        kpi,
        mission,
        transform,
        tokenizer,
        max_len=512,
        labelmap: Dict[int, int] = None,
        **kwargs
    ):
        self.image_data_path = image_data_path
        self.text_data_path = text_data_path
        self.kpi = kpi
        self.mission = mission
        self.loader = default_loader
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labelmap = labelmap
        self.country = kwargs.get("country")

        self.data = self._train_dev_read()

    def image_map(self):
        imglist = os.listdir(self.image_data_path)
        idfunc = lambda x: re.sub(".jpeg|.jpg|.png|.PNG", "", x)
        return {idfunc(x): x  for x in imglist}

    def _train_dev_read(self):
        """load kpi train/valid data, same path with openai fine-tune data
        """
        if isinstance(self.text_data_path, dict):
            datafile = self.text_data_path[self.kpi].get(self.mission)
            if isinstance(datafile, StringIO):
                datafile.seek(0)
                datalines = [json.loads(l) for l in datafile.readlines()]
            else:
                datalines = []

        elif os.path.isdir(self.text_data_path):
            datafile = self.text_data_path / f'{self.kpi}' / f'{self.mission}.jsonl'
            datalines = srsly.read_jsonl(datafile)
            
        else:
            raise AttributeError('You should use openai data path in order to comparation')

        # re to extract raw content
        compattern = self.COMPATTERN
        propattern = self.PROPATTERN
        conpattern = self.CONPATTERN
        apipattern = self.APIPATTERN

        imgmapping = self.image_map()

        data = [
            {
                'img_path': self.image_data_path / imgmapping.get(line["uuid"], "none.jpg"),
                "index": line["uuid"],
                'content': re.search(conpattern, line['prompt'], re.S).group(),
                'api': re.search(apipattern, line['prompt'], re.S).group() if re.search(apipattern, line['prompt'], re.S) else 'Lower Mainstream',
                'label': line['completion']
            } for line in datalines
        ]
        data = [x for x in data if os.path.exists(x["img_path"])]
        logger.info(f"Num of {self.mission}: {len(data)}")
        self.weights = self._reweight(data)

        return data
        
    def _reweight(self, data):
        """calculate the data weight
        """
        weights_dict = {LABEL2ID[l]: len(data) / w for l, w in Counter([x['label'] for x in data]).items()}
        # make sure the correct num
        weights_dict.update({LABEL2ID[k]: 1 for k in LABEL2ID.keys() if LABEL2ID[k] not in weights_dict})

        # make sure the correct corresponse
        # IMPORTANT 标签合并，可能数量上会出错
        weights = [x[1] for x in sorted(weights_dict.items(), key=lambda x: x[0])]

        # dont diff too much
        weights = np.log(weights)

        return weights
    
    def _get_text_segment(self, text_segment):
        if isinstance(text_segment, str):
            tokens = self.tokenizer(text_segment, max_length=self.max_len, truncation=True)['input_ids']
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")

        return tokens

    def _get_image(self, img_path):
        """load image and transform
        """
        image = self.loader(img_path)
        return self.transform(image)

    def _get_label(self, label):
        """Unilever -> Openai mapping
        
        If we combine labels, self.labelmap is nessary.
        """
        # IMPORTANT label id starts from 0
        label_id = LABEL2ID[label]

        if self.labelmap is not None:
            label_id = self.labelmap[label_id]
            
        return label_id
    
    def _get_image_text_api_example(self, index: int, data: dict):
        item = self.data[index]
        img_path = item['img_path']
        img = self._get_image(img_path)
        data["image"] = img

        content = item['content']
        if self.country:
            assert isinstance(self.country, str)
            content = self.country + ". " + content
        input_ids = self._get_text_segment(content)
        data['input_ids'] = input_ids

        api = item['api'].strip()
        api_id = API2ID.get(api, len(API2ID))
        data['condition_ids'] = api_id

        label = item['label']
        label_id = self._get_label(label)
        data['label'] = label_id
    
    def __getitem__(self, index):
        data = {}
        self._get_image_text_api_example(index, data)
        return data

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return ConcatDataset([self, other])

    @staticmethod
    def collate_fn(batch):
        """batch process in dataloader
        1、padding
        2、convert to tensor

        Before collate_fn, transformers removed unexpected keys
        So judge keys here

        Args:
            data: list(batch) of dict(__getitem__)
        """
        images = torch.stack([d["image"] for d in batch])
        tokens = [torch.tensor(d['input_ids']) for d in batch]
        labels = [d['label'] for d in batch]

        inputs = pad_sequence(tokens, batch_first=True)
        mask = (inputs != 0)
        labels = torch.tensor(labels)

        # special for Acceptable Costs
        apis_dict = {}
        if 'condition_ids' in batch[0]:
            apis = torch.tensor([d['condition_ids'] for d in batch])
            apis_dict['condition_ids'] =  apis

        fnoutput = {
            "images": images,
            'input_ids': inputs,
            'attention_mask': mask,
            'labels': labels
        }
        fnoutput.update(apis_dict)

        return fnoutput


def build_transform(is_train, input_size=224, train_interpolation="bicubic", randaug=True):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(input_size, scale=(0.5, 1.0), interpolation=train_interpolation), 
            transforms.RandomHorizontalFlip(),
        ]
        if randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def make_dataloader():
    trainset = NutritionConceptDataset(
        image_data_path=imagedir,
        text_data_path=imagedir,
        kpi=kpi,
        mission="train",
        transform=build_transform(True),
        tokenizer=tokenizer,
        max_len=1024
    )

    validset = NutritionConceptDataset(
        image_data_path=imagedir,
        text_data_path=imagedir,
        kpi=kpi,
        mission="valid",
        transform=build_transform(False),
        tokenizer=tokenizer,
        max_len=1024
    )

    num_tasks = get_world_size()
    global_rank = get_rank()
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=8,
        sampler=torch.utils.data.DistributedSampler(
            trainset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True
        ),
        collate_fn=trainset.collate_fn
    )

    validloader = DataLoader(
        dataset=validset,
        batch_size=8,
        sampler=torch.utils.data.SequentialSampler(
            validset
        ),
        collate_fn=trainset.collate_fn
    )

    return trainloader, validloader


def make_model():
    drop_path = 0.1
    vocab_size = 64010
    # 直接用timm.create_model用不了
    # model = create_model(
    #     model_config,
    #     pretrained=False,
    #     drop_path_rate=drop_path,
    #     vocab_size=vocab_size,
    #     checkpoint_activations=None,
    # )
    args = _get_base_config(
        drop_path_rate=drop_path,
        checkpoint_activations=None
    )
    model = BEiT3ForVisualQuestionAnswering(
        args=args,
        num_classes=3
    ).to("cuda")
    ckpt_state_dict = torch.load(f"{modeldir}/beit3_base_indomain_patch16_224.pth")["model"]
    model.load_state_dict(ckpt_state_dict, strict=False)
    model = model.to(device)

    return model


def evaluate(model, validloader, **kwargs):
    """report evaluation metric

    Args:
        validloader: 验证数据
        writeres: boolean 将metric写入文件
        tta: boolean 是否使用TTA
        stricter: boolean 是否使用更严格的混淆矩阵判别
    """
    model.eval()    
    gtls = []
    prds = []
    with torch.no_grad():
        for _, data in enumerate(validloader):
            valid_x = {
                "image": data['images'].to(device),
                "question": data['input_ids'].to(device),
                "padding_mask": data['attention_mask'].to(device)
            }
            valid_y = data['labels']
            gtls.extend(valid_y)
            pred = model(**valid_x).argmax(1).to('cpu')
            prds.extend(pred)
    
    kacc = accuracy_score(gtls, prds)
    kmaf1 = f1_score(gtls, prds, average='macro')
    kmif1 = f1_score(gtls, prds, average='micro')
    kcomt = confusion_matrix(gtls, prds, labels=list(LABEL2ID.values()))
    print(f"acc score: {kacc}")
    print(f"maf1 score: {kmaf1}")
    print(f"mif1 score: {kmif1}")
    print(f"confusion matrix:\n{kcomt}")


def train(
    model,
    trainloader,
    validloader=None,
    epochs=20,
    learning_rate=1e-5,
    mixup_fn=None
):
    size = len(trainloader.dataset)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}\n----------------------------------")
        for data_iter_step, data in enumerate(trainloader):
            if mixup_fn is not None:
                data["images"], data["labels"] = mixup_fn(data["images"], data["labels"])
            train_x = {
                "image": data['images'].to(device),
                "question": data['input_ids'].to(device),
                "padding_mask": data['attention_mask'].to(device)
            }
            train_y = data['labels'].to(device)
            # Compute prediction error
            pred = model(**train_x)
            loss = loss_fn(pred, train_y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if data_iter_step % 10 == 0:
                loss, current = loss.item(), (data_iter_step + 1) * len(train_x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if validloader:
            evaluate(model, validloader)


def main():
    trainloader, validloader = make_dataloader()
    model = make_model()
    # TODO 如果图片上有文字 真的需要嘛
    mixup = Mixup(
        mixup_alpha=0, cutmix_alpha=0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode="batch",
        label_smoothing=0.1, num_classes=3
    )
    train(model, trainloader, validloader)


if __name__ == "__main__":
    modeldir = "/media/data/MultiModal/beit3"
    imagedir = Path("/media/data/datasets/ConceptUkImage")
    device = "cuda"
    kpi="CVM"
    tokenizer = XLMRobertaTokenizer(f"{modeldir}/beit3.spm")
    main()
    print()