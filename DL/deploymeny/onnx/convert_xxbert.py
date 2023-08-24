#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   convert_xxbert.py
@Time   :   2023/08/10 14:34:52
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   convert torch xxbert model to onnx model
            including self defined subclass
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from pathlib import Path
import numpy as np
import timeit
from inspect import signature
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession

from cln import DistilBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("/media/data/pretrained_models/Distilbert")


def load_torch_model(modelpath):
    model = DistilBertForSequenceClassification.from_pretrained(modelpath).eval().to('cpu')

    return model


def load_and_convert(modelpath, modelname):
    model = load_torch_model(modelpath)

    # IMPORTANT 这个地方的args维度还是很重要的，后期在使用onnx推理的时候，会做broadcast操作，如Add算子
    # dummy_input = torch.randint(0, 21128, (10, 20, 768), device="cuda")
    # dummy_input = model.dummy_inputs
    batch_size = 1
    max_len = 512
    dummy_input = {
        "input_ids": torch.randint(0, 21128, (batch_size, max_len)),
        "attention_mask": torch.randint(0, 2, (batch_size, max_len)),
        "condition_ids": torch.randint(0, 6, (batch_size, ))
    }
    torch.onnx.export(
        model=model,
        args=dummy_input,                   # passed
        # args=tuple(dummy_input.values()), # passed
        # args=list(dummy_input.values()),  # failed
        f=str(modelpath / f'{modelname}.onnx'),
        verbose=False,
        input_names=list(signature(model.forward).parameters.keys()),
        output_names=['logits']
    )


def onnx_vs_torch(sentence, output_path, model_name):
    """Check precision
    """
    input_dict = tokenizer(sentence)
    input_dict = {k: torch.LongTensor(v + [0] * (512 - len(v))) for k, v in input_dict.items()}
    input_dict.update({"condition_ids": torch.LongTensor([1])})

    torch_model = load_torch_model(output_path)
    def _infer(model, inputs):
        with torch.no_grad():
            return model(**inputs).logits[0]
        
    y_torch = _infer(torch_model, input_dict)

    input_dict_array = {
        k: np.expand_dims(v, 0) for k, v in input_dict.items()
    }
    input_dict_array.update({"condition_ids": np.array([1])})
    # sess_onnx = ReferenceEvaluator(str(output_path / f"{model_name}.onnx"))
    # InferenceSession more faster
    sess_onnx = InferenceSession(
        path_or_bytes=str(output_path / f"{model_name}.onnx"),
        providers=['CPUExecutionProvider']
    )
    y_onnx = sess_onnx.run(None, input_dict_array)[0][0]

    print(f"result of torch_model:\t {y_torch.tolist()}")
    print(f"result of onnx model:\t {y_onnx}")
    print(f"difference: {np.abs(y_torch - y_onnx).max()}")  # 4.76837158203125e-07
    # 11.286639904021285
    print(f"time with torch_mode.forward: {timeit.timeit(lambda: torch_model(**input_dict), number=100)}")
    # 12.064857240009587
    print(f"time with onnx inference: {timeit.timeit(lambda: sess_onnx.run(None, input_dict_array), number=100)}")


if __name__ == "__main__":
    output_path = Path("/home/yhao/data/onnx/disbert_cln")
    modelname = 'model_wisignature'
    
    # load_and_convert(output_path, modelname)

    example = (
        """New Knorr Gourmet Sides offer an exciting range of restaurant-style rice and risotto side dishes. """
        """These dishes are not only easy to prepare but also start with the best quality real ingredients. """
        """They bring the excitement that your weekday meals deserve! The Gourmet Sides are available in several varieties, """
        """including Thai Curry Coconut Rice with Lemongrass, Savory and Sweet Teriyaki Rice with Ginger and Soy Sauce, """
        """Creamy Mushroom & Parmesan Risotto with Chives, and Creamy Roasted Chicken & Leek Risotto with Aged Parmesan. """
        """It's important to note that all other Knorr Sides are still available as well."""
    )
    onnx_vs_torch(example, output_path, modelname)