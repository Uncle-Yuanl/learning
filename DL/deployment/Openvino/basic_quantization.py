#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   basic_quantization.py
@Time   :   2023/12/05 10:53:02
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   basic quantization of openvino with nccf
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
import re
import srsly
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import timeit

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import nncf
from nncf import NNCFConfig
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import QuantizationParameters, AdvancedQuantizationParameters
import onnx
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.intel_cpu as intel_cpu
from openvino.tools.mo import convert_model

import sys
sys.path.append("/home/yhao/code/learning/DL/deployment")
from Onnx.cln import DistilBertForSequenceClassification


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


LABELIDMAP = {
    "Credibility": {0: 0, 1: 0, 2: 1, 3: 1},
    "Acceptable_Costs": {0: 0, 1: 1, 2: 2, 3: 2},
    "Advantage": {0: 0, 1: 1, 2: 2, 3: 2},
    "Attention_Catching": {0: 0, 1: 1, 2: 2, 3: 2},
    "Clear,_Concise_Message": {0: 0, 1: 1, 2: 2, 3: 2},
    "CVM": {0: 0, 1: 1, 2: 2, 3: 2},
    "Distinct_Proposition": {0: 0, 1: 1, 2: 2, 3: 2},
    "Message_Connection": {0: 0, 1: 1, 2: 2, 3: 2},
    "Need_Desire": {0: 0, 1: 1, 2: 2, 3: 2},
}


class NutritionConceptDataset(Dataset):
    """Dataset for concept ai

    Placehold for image
    """
    def __init__(
        self,
        data_path,
        kpi,
        mission,
        tokenizer,
        max_len=512,
        labelmap: Dict[int, int] = None,
        **kwargs
    ):
        self.data_path = data_path
        self.kpi = kpi
        self.mission = mission
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labelmap = labelmap

        self.data = self._train_dev_read()

    def _train_dev_read(self):
        """load kpi train/valid data, same path with openai fine-tune data
        """
        if os.path.isdir(self.data_path):
            datafile = self.data_path / f'{self.kpi}' / f'{self.mission}.jsonl'
            datalines = srsly.read_jsonl(datafile)
            # re to extract raw content
            compattern = '(?<=Company: ).*(?=\nProduct:)'
            propattern = '(?<=\nProduct: ).*(?=\nAd:)'
            conpattern = '(?<=\nAd:).*(?=\nPrice:)'
            apipattern = '(?<=\nApi:).*(?=\nAspect:)'

            data = [
                {
                    'img_path': '',
                    'index': re.search(compattern, line['prompt'], re.S).group() + '_' + re.search(propattern, line['prompt'], re.S).group(),
                    'content': re.search(conpattern, line['prompt'], re.S).group(),
                    'api': re.search(apipattern, line['prompt'], re.S).group() if re.search(apipattern, line['prompt'], re.S) else 'Lower Mainstream',
                    'label': line['completion']
                } for line in datalines
            ]
            self.weights = self._reweight(data)
            return data
        else:
            raise AttributeError('You should use openai data path in order to comparation')

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
        """placehold for multi modal
        """
        return 'NotImplementedError'

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

    @staticmethod
    def collate_fn(data):
        """batch process in dataloader
        1、padding
        2、convert to tensor

        Before collate_fn, transformers removed unexpected keys
        So judge keys here

        Args:
            data: list(batch) of dict(__getitem__)
        """
        tokens = [torch.tensor(d['input_ids']) for d in data]
        labels = [d['label'] for d in data]

        inputs = pad_sequence([torch.hstack([t, torch.zeros(512 - len(t), dtype=torch.int64)]) for t in tokens], batch_first=True)
        mask = (inputs != 0).type(torch.LongTensor)
        labels = torch.tensor(labels)

        # special for Acceptable Costs
        apis_dict = {}
        if 'condition_ids' in data[0]:
            # IMPORTANT dim(0) = 1
            apis = torch.tensor([d['condition_ids'] for d in data])
            apis_dict['condition_ids'] =  apis

        fnoutput = {
            'input_ids': inputs,
            'attention_mask': mask,
            'labels': labels
        }
        fnoutput.update(apis_dict)

        fnoutput = {k: np.array(v) for k, v in fnoutput.items()}
        return fnoutput


def quantize_model(modeloutput):
    """
    wo advanced_parameters:
        MEM: 1049
    wi advanced_parameters:
        MEM: 800
    """
    model = onnx.load(datafolder / model_name)
    quantized_model = nncf.quantize(
        model, calibration_dataset, 
        target_device=nncf.TargetDevice.CPU,
        model_type=ModelType.TRANSFORMER,
        advanced_parameters=AdvancedQuantizationParameters(
            weights_quantization_params=QuantizationParameters(num_bits=8),
            activations_quantization_params=QuantizationParameters(num_bits=8)
        )
    )

    with open(datafolder / f"{modeloutput}.onnx", "wb") as f:
        f.write(quantized_model.SerializeToString())

    """
    All failed
        # # convert ONNX model to OpenVINO model
        # ov_quantized_model = convert_model(quantized_model)

        # compile the model to transform quantized operations to int8
        # model_int8 = ov.compile_model(quantized_model)
    """    
    # convert to openvino IR model
    ov_model = convert_model(datafolder / f"{modeloutput}.onnx")

    # save the model
    ov.save_model(ov_model, datafolder / f"{modeloutput}.xml")


def use_raw_torch():
    models = [
        DistilBertForSequenceClassification.from_pretrained(datafolder) for _ in range(10)
    ]
    global input_fp32
    input_fp32 = {k: torch.tensor(v) for k, v in input_fp32.items()}
    with torch.no_grad():
        res = [model(**input_fp32) for model in models]

    return res


def use_raw_onnx():
    """
    Metric:
        0.5, 0.43823529411764706

    MEM:
        load:  850 MiB
        infer: 998 MiB
    """
    core = ov.runtime.Core()
    ov_model = core.read_model(model=str(datafolder / model_name))
    ov_compiled_quantized_model = core.compile_model(model=ov_model, device_name="CPU")
    output_layer_1 = ov_compiled_quantized_model.outputs[0]
    res = ov_compiled_quantized_model(input_fp32)[output_layer_1]

    print(f"time with torch_mode.forward: {timeit.timeit(lambda: ov_compiled_quantized_model(input_fp32), number=100)}")

    check_acc(ov_compiled_quantized_model, output_layer_1)

    return res


def use_quantized_onnx():
    core = ov.runtime.Core()
    ov_model = core.read_model(model=str(datafolder / "nncf_quantized_model.onnx"))
    ov_compiled_quantized_model = core.compile_model(model=ov_model, device_name="CPU")
    output_layer_1 = ov_compiled_quantized_model.outputs[0]
    res = ov_compiled_quantized_model(input_fp32)[output_layer_1]

    print(f"time with torch_mode.forward: {timeit.timeit(lambda: ov_compiled_quantized_model(input_fp32), number=100)}")

    return res


def use_quantized_openvino(modeloutput):
    """
    quantized_model:
        Metric:
            0.5333, 0.44845034788108795

        MEM:
            load:  1198 MiB
            infer: 1049 MiB
    """
    # load openvino IR model
    core = ov.runtime.Core()
    quantized_xml_model = core.read_model(model=datafolder / f"{modeloutput}.xml")
    quantized_xml_model = core.compile_model(model=quantized_xml_model, device_name="CPU")
    output_layer_2 = quantized_xml_model.outputs[0]
    res = quantized_xml_model(input_fp32)[output_layer_2]

    print(f"time with torch_mode.forward: {timeit.timeit(lambda: quantized_xml_model(input_fp32), number=100)}")

    check_acc(quantized_xml_model, output_layer_2)

    return res


def use_quantized_openvino_optimized(modeloutput):
    core = ov.Core()
    # in case of Performance
    # core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
    core.set_property("CPU", {hints.inference_precision: ov.Type.bf16})
    core.set_property("CPU", intel_cpu.sparse_weights_decompression_rate(0.8))
    quantized_xml_model = core.read_model(model=datafolder / f"{modeloutput}.xml")

    config = {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        hints.num_requests: "1",
        hints.inference_precision: ov.Type.bf16 
    }
    quantized_xml_model = core.compile_model(
        model=quantized_xml_model,
        device_name="CPU",
        config=config
    )

    output_layer_2 = quantized_xml_model.outputs[0]
    res = quantized_xml_model(input_fp32)[output_layer_2]

    # print(f"time with torch_mode.forward: {timeit.timeit(lambda: quantized_xml_model(input_fp32), number=100)}")

    check_acc(quantized_xml_model, output_layer_2)

    return res


def use_quantized_openvino_optimized_10_models(modeloutput):
    core = ov.Core()
    # in case of Performance
    # core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
    core.set_property("CPU", {hints.inference_precision: ov.Type.bf16})
    core.set_property("CPU", intel_cpu.sparse_weights_decompression_rate(0.8))
    quantized_xml_model = core.read_model(model=datafolder / f"{modeloutput}.xml")

    config = {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        hints.num_requests: "1",
        hints.inference_precision: ov.Type.bf16 
    }
    quantized_xml_models = [
        core.compile_model(
            model=quantized_xml_model,
            device_name="CPU",
            config=config
        ) for _ in range(10)
    ]

    output_layer_2 = quantized_xml_model.outputs[0]
    res = [quantized_xml_model(input_fp32)[output_layer_2] for quantized_xml_model in quantized_xml_models]

    # print(f"time with torch_mode.forward: {timeit.timeit(lambda: quantized_xml_model(input_fp32), number=100)}")

    # check_acc(quantized_xml_model, output_layer_2)

    return res


def use_quantized_openvino_optimized_gpu(modeloutput):
    core = ov.Core()
    # in case of Performance
    core.set_property("GPU", {hints.execution_mode: "PERFORMANCE"})
    core.set_property("GPU", {hints.inference_precision: ov.Type.bf16})
    core.set_property("GPU", intel_cpu.sparse_weights_decompression_rate(0.8))
    quantized_xml_model = core.read_model(model=datafolder / f"{modeloutput}.xml")

    config = {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        hints.num_requests: "1",
        hints.inference_precision: ov.Type.bf16 
    }
    quantized_xml_model = core.compile_model(
        model=quantized_xml_model,
        device_name="GPU",
        config=config
    )

    output_layer_2 = quantized_xml_model.outputs[0]
    res = quantized_xml_model(input_fp32)[output_layer_2]

    # print(f"time with torch_mode.forward: {timeit.timeit(lambda: quantized_xml_model(input_fp32), number=100)}")

    check_acc(quantized_xml_model, output_layer_2)

    return res


def check_acc(model, outkey):
    gtls = []
    prds = []

    for datadict in validloader:
        gtls.append(datadict.pop("labels"))
        prds.append(model(list(datadict.values()))[outkey].argmax())

    print(f"acc: {accuracy_score(gtls, prds)}")
    print(f"macro f1: {f1_score(gtls, prds, average='macro')}")


def main():
    pass


if __name__ == '__main__':
    datafolder = Path("/home/yhao/data/onnx/disbert_cln")
    tokenpath = "/media/data/pretrained_models/RoBERTa"
    model_name = "model_wisignature.onnx"

    tokenizer = AutoTokenizer.from_pretrained(tokenpath)
    trainset = NutritionConceptDataset(
        data_path=datafolder,
        kpi="",
        mission="train",
        tokenizer=tokenizer,
        labelmap=LABELIDMAP["Acceptable_Costs"]
    )
    validset = NutritionConceptDataset(
        data_path=datafolder,
        kpi="",
        mission="valid",
        tokenizer=tokenizer,
        labelmap=LABELIDMAP["Acceptable_Costs"]
    )
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=trainset.collate_fn)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, collate_fn=validset.collate_fn)

    calibration_dataset = nncf.Dataset(trainloader)
    input_fp32 = next(iter(validloader)) # FP32 model input
    input_fp32.pop("labels", None)

    model_name = "nncf_quantized_model_forceint8"
    # quantize_model(model_name)
    # use_raw_torch()

    input_fp32 = list(input_fp32.values())
    # use_raw_onnx()
    # use_quantized_openvino(model_name)
    # use_quantized_openvino_optimized(model_name)
    # use_quantized_openvino_optimized_10_models(model_name)

    use_quantized_openvino_optimized_gpu(model_name)