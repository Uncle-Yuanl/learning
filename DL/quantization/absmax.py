#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   absmax.py
@Time   :   2024/10/16 10:57:41
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   最大值量化
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import torch


def absmax_quantize(rawtensor):
    """实现的是转为int8
    """
    # 计算量化常数 quantization constant
    qc = 127 / torch.max(torch.abs(rawtensor))
    quantensor = (rawtensor * qc).round()
    dequantensor = quantensor / qc  # 此时还是float32
    return quantensor.to(torch.int8), dequantensor


if __name__ == "__main__":
    tensor1 = torch.tensor([-0.3, 0.1, 0, 0.7])
    tensor2 = torch.tensor([-0.3, 0.1, 0, 0.3])
    tensor3 = torch.tensor([-0.3, 0.1, 0])

    qt1, dqt1 = absmax_quantize(tensor1)
    qt2, dqt2 = absmax_quantize(tensor2)
    qt3, dqt3 = absmax_quantize(tensor3)

    print(qt1, dqt1)
    print(qt2, dqt2)
    print(qt3, dqt3)