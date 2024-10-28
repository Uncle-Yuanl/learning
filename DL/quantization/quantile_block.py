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


if __name__ == "__main__":
    Q = calculate_16qi()
    print(Q)