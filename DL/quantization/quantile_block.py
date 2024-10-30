#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   quantile.py
@Time   :   2024/10/22 15:10:27
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   QLora NF4 分位数 + 分块K位量化
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import math
from scipy.stats import norm
import torch


def calculate_16qi(offset=0.99, num_bins=16):
    """野生办法计算向量的16个分位数
    问题：
        0没有映射到0
        没有完全使用4位数据类型的全部16比特位
    """
    # CDF反函数均分
    yrange = torch.linspace(1 - offset, offset, num_bins + 1)
    quantile = norm.ppf(yrange).tolist()
    # 计算分位数(等分点中点)
    tmp = [(quantile[1:][idx] + val) / 2 for idx, val in enumerate(quantile[:-1])]
    # t_max, t_min = 1, -1     # 为什么范围是[-1, 1]
    t_max, t_min = 128, -127   # -127 128也行
    r_max, r_min = tmp[-1], tmp[0]
    S = (r_max - r_min) / (t_max - t_min)
    # 缩放
    Z = t_max - r_max / S
    # 分位数量化到[-1,1]
    Q = [x/S + Z for x in tmp]
    return Q


def create_normal_map(offset=0.9677083, use_extra_value=True, bits=8):
    """这里是用NF4表示8bit量化，因此v2是用256减，如果是4bit就用16减，非对称中就剩一位了
    Args:
        use_extra_value: True -> 正数部分8bit位，负数7bit位，然后0占一位
    """
    bits = 2 ** bits
    if use_extra_value:
        # 笔记中估算的方法，非对称
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        # we have 15 non-zero values in this data type
        v2 = [0] * (bits - 15)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        # 对称量化
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        # we have 14 non-zero values in this data type
        v2 = [0] * (bits - 14)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    
    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == bits

    return values


def quantize(input_blocked_tensor):
    """分块 -> 保存每个块的量化常数 -> 使用量化常数归一当前值 -> 找到在map中对应的下标
    
    Args:
        input_blocked_tensor: 分块后的多维向量
    """
    quantize_result = []
    quantize_constant = []
    for block in input_blocked_tensor:
        c = max([abs(val) for val in block])
        quantize_constant.append(c)
        norm_block = [val / c for val in block]
        block_result = []
        for norm_val in norm_block:
            # 查找归一后的值在Q中的位置
            # 位置就是量化后的值，因为Q就是根据bit位计算出的map
            min_sim = math.inf
            idx = -1
            for j, q in enumerate(Q): # 寻找Q中最近值的索引
                sim = abs(norm_val - q)
                if sim < min_sim:
                    min_sim = sim
                    idx = j
            block_result.append(idx)
        quantize_result.append(block_result)
    return quantize_constant, quantize_result


def dequantize(quantize_constant, quantize_result):
    dequantize_result = []
    for c, block in zip(quantize_constant, quantize_result):
        dequantize_block = []
        for val in block:
            dequantize_block.append(Q[val].item() * c)
        dequantize_result.append(dequantize_block)

    return dequantize_result


if __name__ == "__main__":
    # Q = calculate_16qi()
    Q = create_normal_map(bits=4)
    print(Q)

    input_blocked_tensor = [
        [-1.28645003578589, -1.817660483275528, 9.889441349505042, 0.010208034676132627],
        [ -15.009014631551885, 1.4136255086268115, -7.815595761491153, 10.766760590950263], 
        [-0.731406153917959, 3.468224595908726, 2.445252541840315, -8.970824523299282], 
        [-9.641638854625175, 7.696158363188889, -5.323939281255154, 5.97160401402024]
    ]
    quantize_constant, quantize_result = quantize(input_blocked_tensor)
    dequantize_result = dequantize(quantize_constant, quantize_result)
    print()