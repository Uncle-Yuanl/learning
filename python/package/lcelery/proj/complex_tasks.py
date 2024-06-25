#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   complex_tasks.py
@Time   :   2024/06/25 13:55:48
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   一些相对复杂些的任务
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import time
from celery import group
from celery import subtask, shared_task
from celery.result import allow_join_result
from .celery import app


@app.task
def make_params(start, num):
    results = []
    for i in range(start, start + num):
        results.append((i, i + 1))

    return results


@app.task
def simle_task(s1, s2):
# def simle_task(args):
    # logger.info(f"args: {args}")
    # s1, s2 = args
    logger.info(f"args: s1: {s1}, s2: {s2}")
    if isinstance(s1, int):
        s1 = str(s1)
    if isinstance(s2, int):
        s2 = str(s2)
    
    time.sleep(5)

    return s1 + "-" + s2


@shared_task
def dismap(result, some_task):
    logger.info(f"type of some task: {type(some_task)}")
    logger.info(f"content of some task: {some_task}")
    some_task = subtask(some_task)
    logger.info(f"type of some task: {type(some_task)}")

    group_task = group(
        some_task.clone([*arg]) for arg in result
    )

    results = group_task()
    logger.info(f"type of results: {type(results)}")
    # results = results.wait()
    # logger.info(f"type of results: {type(results)}")

    # 这里的results是前置函数的返回值
    # 如果chain前置simle_task(3, 5)，则如[(3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    return results
