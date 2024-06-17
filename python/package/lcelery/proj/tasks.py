#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   tasks.py
@Time   :   2024/06/17 17:16:34
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实际放任务的脚本
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


from .celery import app


@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def xsum(numbers):
    return sum(numbers)