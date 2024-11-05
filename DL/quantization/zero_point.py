#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   zero_point.py
@Time   :   2024/10/16 11:20:27
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实现零点量化
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import torch


def zeropoint_quantize(rawtensor, signed=True):
    # 计算量化常数
    x_range = torch.max(rawtensor) - torch.min(rawtensor)
    x_range = 1 if x_range == 0 else x_range
    qc = (2 ** 8 - 1) / x_range
    # 计算零点
    if signed:
        # use signed integer if necessary (perhaps due to HW considerations).
        zeropoint = (-qc * torch.min(rawtensor) - 128).round()
        quantensor = torch.clip((qc * rawtensor + zeropoint).round(), -128, 127)
        quantensor = quantensor.to(torch.int8)
    else:
        # unsigned integer to represent the quantized range
        zeropoint = (-qc * torch.min(rawtensor)).round()
        quantensor = torch.clip((qc * rawtensor + zeropoint).round(), 0, 255)
        quantensor = quantensor.to(torch.uint8)

    dquantensor = (quantensor - zeropoint) / qc
    return quantensor, dquantensor


if __name__ == "__main__":
    tensor1 = torch.tensor([-0.3, 0.1, 0, 0.7])

    qt1, dqt1 = zeropoint_quantize(tensor1)
    qt2, dqt2 = zeropoint_quantize(tensor1, False)

    print(qt1, dqt1)
    print(qt2, dqt2)