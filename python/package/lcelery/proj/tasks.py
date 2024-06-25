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

import time
from celery import group
from celery.result import allow_join_result
from .celery import app


@app.task
def add(x, y):
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def task_use_task(x, y):
    x1 = add(x, y)
    y1 = add(x, y)
    return mul(x1, y1)


@app.task
def xsum(numbers):
    return sum(numbers)


@app.task
# 新添task之后，需要重新启动worker
def xconcat(s1, s2):
    if isinstance(s1, int):
        s1 = str(s1)
    if isinstance(s2, int):
        s2 = str(s2)
        
    return s1 + "-" + s2


@app.task
def cpu_cost_task():
    time.sleep(5)
    return "wake up"


@app.task
def cpu_error_task(num):
    time.sleep(5)
    if num != 0:
        raise ValueError
    
    return num
    

@app.task
def task_with_group(x):
    nestedg = group(
        add(i, i) for i in range(x)
    )
    # results = nestedg.apply_async()

    return nestedg


@app.task
def task_with_groupapply(x):
    nestedg = group(
        add(i, i) for i in range(x)
    )
    results = nestedg.apply_async()

    return results


@app.task
def task_with_groupcall(x):
    nestedg = group(
        add(i, i) for i in range(x)
    )
    results = nestedg()

    return results


@app.task
def task_with_groupdelaycall(x):
    nestedg = group(
        add.s(i, i) for i in range(x)
    )
    # results = nestedg()
    results = nestedg.apply_async()

    return results


@app.task(bind=True)
def task_with_groupapplybind(self, x):
    nestedg = group(
        add(i, i) for i in range(x)
    )
    results = nestedg.apply_async()

    return results


@app.task
def task_with_groupapplyget(x):
    """启动worker直接找不到这个函数了，神奇
    """
    nestedg = group(
        add(i, i) for i in range(x)
    )
    results = nestedg.apply_async().get()

    return results


@app.task()
def task_with_groupallowjoin(x):
    """
    """
    nestedg = group(
        add(i, i) for i in range(x)
    )
    results = nestedg()
    
    with allow_join_result():
        res = results.get()

    # return res