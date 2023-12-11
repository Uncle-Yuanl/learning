#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   trace_and_script.py
@Time   :   2023/12/11 15:43:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   learning the difference between trace and script
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import torch


def func_raw(x):
    return x * 2


def func_if(x):
    if x.mean() > 1.0:
        # return 1.0  Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions
        return torch.tensor(1.0)
    else:
        return torch.tensor(0.0)
    

def func_for(x):
    for _ in range(2):
        x *= x
    
    return x


def use_trace(func, x):
    ftrace = torch.jit.trace(func, (torch.ones(2,2)))

    print(type(ftrace))  # <class 'torch.jit.ScriptFunction'>
    print(ftrace.graph)
    print(ftrace(x))


"""================= def use_script(func, x): ====================
"""
@torch.jit.script
def func_if_script_float(x):
    if x.mean() > 1.0:
        return 1.0
        # return torch.tensor(1.0)  # Previous return statement returned a value of type Tensor but this return statement returns a value of type float
    else:
        return 0.0


@torch.jit.script
def func_if_script_for(x):
    for _ in range(2):
        x *= x
    
    return x


if __name__ == '__main__':
    x = torch.ones(2,2).add_(1.0)

    # use_trace(func_raw, x)
    # use_trace(func_if, x)
    # use_trace(func_for, x)

    # print(func_if_script_float.graph)
    # print(func_if_script_float(x))

    print(func_if_script_for(x))