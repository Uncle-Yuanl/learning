#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   base_api.py
@Time   :   2024/11/05 10:24:00
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习torch.quantization中基础的api实现
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import torch
from torch.ao.quantization import QuantStub, DeQuantStub


def use_quantstub():
    x = torch.Tensor([[[[-1,-2,-3],[1,2,3]]]])
    xq = torch.quantize_per_tensor(x, scale=0.0472, zero_point=64, dtype=torch.quint8)
    print(xq)  # tensor显示的数值是【反量化后】的值
    print(xq.int_repr())  # 显示量化后的int值

    # DL.quantization.zero_point.py
    tensor1 = torch.tensor([-0.3, 0.1, 0, 0.7])
    # 注意这里scale = 1 / qc
    tensorq = torch.quantize_per_tensor(tensor1, scale=0.00392156862745098, zero_point=76, dtype=torch.quint8)
    print(tensorq.int_repr())


def learn_fuse_module():
    pass


if __name__ == "__main__":
    use_quantstub()
    print()