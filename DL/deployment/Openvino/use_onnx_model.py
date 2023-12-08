#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   load_onnx_and_convert.py
@Time   :   2023/08/10 11:43:04
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   load a onnx model and convert it to openvino format
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import timeit
from pathlib import Path
import numpy as np
from openvino.runtime import Core
from onnx.reference import ReferenceEvaluator

from openvino.runtime import serialize


x = np.random.randn(4, 1000).astype(np.float32)
a = np.random.randn(1000, 1000).astype(np.float32) / 10
b = np.random.randn(1, 1000).astype(np.float32)
feeds = {'X': x, 'A': a, 'B': b}


def onnx_inference(onnx_model_path, feeds):
    sess_onnx = ReferenceEvaluator(str(onnx_model_path))
    y_onnx = sess_onnx.run(None, feeds)[0]

    return sess_onnx, y_onnx


def load_onnx_and_inference(onnx_model_path):
    core = Core()
    model_onnx = core.read_model(model=str(onnx_model_path))
    logger.info(f"type of model_onnx loaded by openvino: {type(model_onnx)}")

    compiled_model = core.compile_model(model=model_onnx, device_name="CPU")
    logger.info(f"type of compiled_model by openvino: {type(compiled_model)}")

    print(compiled_model.inputs)
    print(compiled_model.outputs)    
    # OpenVINO
    openvino_output = compiled_model([x, a, b])
    logger.info(f"type of openvino_output: {type(openvino_output)}")
    result = openvino_output["Y"]
    # ONNX
    sess_onnx, y_onnx = onnx_inference(onnx_model_path, feeds)

    # diff: 4.76837158203125e-06
    print(f"difference: {np.abs(y_onnx - result).max()}")
    # ONNX有波动，0.21 ~ 0.48  OpenVINO稳定0.16
    print(f"time with ONNX: {timeit.timeit(lambda: sess_onnx.run(None, feeds), number=1000)}")
    print(f"time with OpenVINO: {timeit.timeit(lambda: compiled_model([x, a, b]), number=1000)}")


def load_and_convert_xml(onnx_model_path):
    core = Core()
    model_onnx = core.read_model(model=str(onnx_model_path))

    serialize(model_onnx, xml_path=onnx_model_path.parent / f"{onnx_model_path.stem}.xml")


def load_xml_and_inference(xml_model_path):
    core = Core()
    model_xml = core.read_model(model=str(xml_model_path))
    compiled_model = core.compile_model(model=model_xml, device_name="CPU")

    y_xml = compiled_model([x, a, b])["Y"]
    onnx_model_path = str(xml_model_path.parent / f"{xml_model_path.stem}.onnx")
    _, y_onnx = onnx_inference(onnx_model_path, feeds)

    print(f"difference: {np.abs(y_onnx - y_xml).max()}")


if __name__ == "__main__":
    output_path = Path("/home/yhao/data/onnx")

    load_onnx_and_inference(output_path / "linear_regression.onnx")
    load_and_convert_xml(output_path / "linear_regression.onnx")
    load_xml_and_inference(output_path / "linear_regression.xml")