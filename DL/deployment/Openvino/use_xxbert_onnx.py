#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   use_xxbert_onnx.py
@Time   :   2023/08/24 16:20:53
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   xxbert转成onnx后再转为openvino，做Intel cpu推理
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from pathlib import Path
from pprint import pprint
import numpy as np
import timeit
import torch
from transformers import AutoTokenizer
from openvino.runtime import Core
from openvino.runtime import serialize


def check_device(core: Core):
    devices = core.available_devices

    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


def load_onnx_model(core, onnx_model_path, export=False):
    onnx_model = core.read_model(model=onnx_model_path)
    compiled_onnx_model = core.compile_model(model=onnx_model, device_name="CPU")

    if export:
        pl = onnx_model_path.split('/')
        dir, modelname = '/'.join(pl[:-1]), pl[-1].split('.')[0]
        xml_path = f"{dir}/{modelname}.xml"
        serialize(onnx_model, xml_path=xml_path)

    pprint(f"type of onnx model: {type(onnx_model)}")
    pprint(f"type of compiled model onnx: {type(compiled_onnx_model)}")
    pprint(f"check model inputs: \n{onnx_model.inputs}")
    pprint(f"certain input layer: {onnx_model.input(0)}")
    pprint(f"name of the first input layer: {onnx_model.input(0).any_name}")

    pprint(f"check model outputs: \n{onnx_model.outputs}")
    pprint(f"certain output layer: {onnx_model.output(0)}")
    pprint(f"name of the first output layer: {onnx_model.output(0).any_name}")

    # inputs and outpus of model and compiled_model are different
    pprint(f"check compiled model outputs: \n{compiled_onnx_model.outputs}")

    return compiled_onnx_model


def make_infer_data():
    sentence = (
        """New Knorr Gourmet Sides offer an exciting range of restaurant-style rice and risotto side dishes. """
        """These dishes are not only easy to prepare but also start with the best quality real ingredients. """
        """They bring the excitement that your weekday meals deserve! The Gourmet Sides are available in several varieties, """
        """including Thai Curry Coconut Rice with Lemongrass, Savory and Sweet Teriyaki Rice with Ginger and Soy Sauce, """
        """Creamy Mushroom & Parmesan Risotto with Chives, and Creamy Roasted Chicken & Leek Risotto with Aged Parmesan. """
        """It's important to note that all other Knorr Sides are still available as well."""
    )
    tokenizer = AutoTokenizer.from_pretrained("/media/data/pretrained_models/Distilbert")
    input_dict = tokenizer(sentence)
    input_dict = {k: torch.LongTensor(v + [0] * (512 - len(v))) for k, v in input_dict.items()}
    input_dict.update({"condition_ids": torch.LongTensor([1])})
    input_dict_array = {
        k: np.expand_dims(v, 0) for k, v in input_dict.items()
    }
    input_dict_array.update({"condition_ids": np.array([1])})

    return input_dict_array


def infer_model(compiled_model, input_data):
    """The CompiledModel inference result is a dictionary where keys are the Output class instances 
    (the same keys in compiled_model.outputs that can also be obtained with compiled_model.output(index)) 
    and values - predicted result in np.array format.

    Important:
        - input_layer and output_layer derived from compiled model

    """
    input_layers = compiled_model.inputs
    output_layer = compiled_model.output(0)

    # # for single input models only
    # result = compiled_model(input_data)[output_layer]

    # # for multiple inputs in a list
    # result = compiled_model([input_data])[output_layer]

    # # or using a dictionary, where the key is input tensor name or index
    # result = compiled_model({input_layer.any_name: input_data})[output_layer]

    input_dict = {il.any_name: v for il, v in zip(input_layers, input_data.values())}
    result = compiled_model(input_dict)[output_layer]

    return result


def time_record():
    return infer_model(compiled_onnx_model, make_infer_data())


if __name__ == "__main__":
    modeldir = Path("/home/yhao/data/onnx/disbert_cln")
    modelname = "model_wisignature.onnx"
    core = Core()
    check_device(core)
    compiled_onnx_model = load_onnx_model(core, modeldir / modelname, False)
    result = infer_model(compiled_onnx_model, make_infer_data())
    print(result)
    # 2.9 s
    print(f"time with openvino inference: {timeit.timeit(lambda: time_record, number=100)}")
