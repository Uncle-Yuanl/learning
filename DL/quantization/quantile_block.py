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


def create_normal_map(offset=0.9677083, use_extra_value=True):
    """这里是用NF4表示8bit量化，因此v2是用256减，如果是4bit就用16减，非对称中就剩一位了
    Args:
        use_extra_value: True -> 正数部分8bit位，负数7bit位，然后0占一位
    """
    if use_extra_value:
        # 笔记中估算的方法，非对称
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        # we have 15 non-zero values in this data type
        v2 = [0] * (256 - 15)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        # 对称量化
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        # we have 14 non-zero values in this data type
        v2 = [0] * (256 - 14)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    
    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 256

    return values


if __name__ == "__main__":
    # Q = calculate_16qi()
    Q = create_normal_map()
    print(Q)