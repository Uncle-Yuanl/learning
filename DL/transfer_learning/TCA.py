#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   TCA.py
@Time   :   2023/06/07 22:41:59
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   迁移成分分析--基于最大均值差异法的迁移学习
            两个概率分布p和q之间的最大均值差异的平方 = RKHS(可再生核希尔伯特空间)上平均嵌入的距离
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)

    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T)

    return K


class TCA:
    def __init__(self, kernel_type='primal'):
        self.kernel_type = kernel_type

    def fit(self, Xs, Xt):
        """
        """
        pass

