#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   global_eval.py
@Time   :   2023/07/11 11:12:43
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   global() locals() and eval()
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


GLOBAL_VAR = "global_var"

print('before function define')
print(globals().keys())


def func(param1, param2=2):
    local_var = "local_var"
    print('in one function')
    print(locals())

func(1)

print('after function define')
print(globals().keys())

eval_value = eval('GLOBAL_VAR')
print(type(eval_value), eval_value)

eval_value = eval('1.1')
print(type(eval_value), eval_value)

eval_value = eval('abc')
print(type(eval_value), eval_value)