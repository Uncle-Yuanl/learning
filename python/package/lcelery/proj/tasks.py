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
from celery.result import GroupResult
from .celery import app


@app.task
def add(x, y):
    logger.info(f"x: {x}, y: {y}")
    if isinstance(x, GroupResult):
        xs = x.get()
        return [x + y for x in xs]
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
    """不能用get
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
        add.s(i, i) for i in range(x, x + x)
    )
    results = nestedg.apply_async()
    
    with allow_join_result():
        res = results.get()

    return res
        

@app.task
def task_with_nested_group(x):
    """这个参数生成有问题，add.s(i, j)只有第一个j是3，其他都是5
    """
    inner_nested_group = group(
        group(
            add.s(i, j) for i in range(x)
        ) for j in range(3, 6)
    )

    results = inner_nested_group.apply_async()

    return results


@app.task
def task_with_nested_groupimp(x):
    """这个是没问题
    """
    inner_nested_group = group(
        group(
            add.s(i, i) for i in range(j)
        ) for j in range(x, 6)
    )

    results = inner_nested_group.apply_async()

    # Error: 目前不知道怎么在task中获取结果
    # do something with results
    # res = results.get()
    # res = results.join()

    return results


@app.task
def task_with_nested_groupapplyimp(x):
    """
    """
    inner_nested_group = group(
        group(
            add.s(i, i) for i in range(j)
        ) for j in range(x, 6)
    )

    results = inner_nested_group.apply()

    return results