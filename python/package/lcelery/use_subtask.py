#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   use_subtask.py
@Time   :   2024/06/25 13:47:30
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用subtask和map来控制任务流
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

from celery import chain, group
from proj.complex_tasks import make_params, simle_task, dismap


def simple_workflow():
    cwf = chain(make_params.s(3, 5), dismap.s(simle_task.s()))

    results = cwf.apply() # debug能跳过去  实际在这里就发送simle_task任务了
    print(type(results))  # EagerResult

    res = results.get()
    print(type(res))      # GroupResult
    print(res)            # single id

    res = res.get()
    print(type(res))
    print(res)            # final result


def simple_workflow_async():
    """实际上任务也是能调用的，只是因为异步无法直接获取结果，只能返回id
    """
    cwf = chain(make_params.s(3, 5), dismap.s(simle_task.s()))

    results = cwf.apply_async()  # 发送任务
    print(type(results))         # AsyncResult

    res = results.get()
    print(type(res))             # list, len = 2. res[0]：group组任务的id
    print(res)                   # res[1]：组里面每个subtask的id


def simple_workflow_asynccollect():
    """
    """
    cwf = chain(make_params.s(3, 5), dismap.s(simle_task.s()))

    results = cwf.apply_async()  # 发送任务
    print(type(results))         # AsyncResult

    print(results.ready())
    print(results.successful())
    res = results.collect()
    print(type(res))

    res = [v for v in res if isinstance(v[1], str)]
    # res[0]: (<AsyncResult: d6e6f674-797c-48ac-87d5-52b14ad1649b>, '3-4')
    print(res)


if __name__ == "__main__":
    # simple_workflow()
    # simple_workflow_async()
    simple_workflow_asynccollect()