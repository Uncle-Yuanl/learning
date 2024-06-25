#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   group_chain_deepdive.py
@Time   :   2024/06/19 12:15:10
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   深度学习一下group, chain的知识点，配合cpu耗时任务
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import time
from celery import group, chain
from proj.tasks import cpu_cost_task, cpu_error_task
from proj.tasks import task_with_group


def single_processor():
    """貌似celery -A proj worker -l INFO已经多进程了  很奇怪
    """
    s = time.time()
    gs = group(cpu_cost_task.s() for _ in range(2))
    res = gs().get()
    print(time.time() - s)  # 5s
    print(res)


def multi_processor_state_check():
    gs = group(cpu_error_task.s(_ % 2) for _ in range(10))
    try:
        res = gs().get()
        print(res)
    except Exception as e:
        print("error")

    async_res = gs.apply_async()
    print(type(async_res))

    res = async_res.get()  # 直接报错
    for r in res:
        print(type(r))


def group_retry():
    gs = group(cpu_cost_task.s() for _ in range(2))
    async_res = gs.apply_async(retry=True, retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    })
    res = async_res.get()
    print()


def group_with_group():
    sg1 = group(cpu_cost_task.s() for _ in range(2))
    sg2 = group(cpu_cost_task.s() for _ in range(2))

    g = group(sg1, sg2)

    s = time.time()
    results = g.apply_async()
    print(results.get())
    print(time.time() - s)

    print()


def use_grouptask():
    """
    Returns:
        {
            'task': 'celery.group',
            'args': [],
            'kwargs': {
                'tasks': [0, 2]
                },
            'options': {},
            'subtask_type': 'group',
            'immutable': False,
            'chord_size': None
        }
    可以理解为将task本身返回了，tasks中的0和2是子任务id
    类似bind=True
    """
    from proj.tasks import task_with_group
    gt = task_with_group.s(2)
    # res = gt.delay().get()
    res = gt.apply_async().get()  # same res = dict
    print(res)


def use_grouptaskapply():
    """直接报错：'int' object has no attribute 'app'
    """
    from proj.tasks import task_with_groupapply
    gt = task_with_groupapply.s(2)
    res = gt().get()
    print(res)


def use_grouptaskapplybind():
    """直接报错：'int' object has no attribute 'app'
    """
    from proj.tasks import task_with_groupapplybind
    gt = task_with_groupapplybind.s(2)
    res = gt.apply_async().get()
    print(res)    


def use_grouptaskget():
    """此时这个task_with_groupapplyget函数，直接变成了class function
    并不是celery的封装对象了
    """
    from proj.tasks import task_with_groupapplyget
    gt = task_with_groupapplyget.s(2)
    res = gt.apply_async().get()
    print(res)   


def group_with_grouptask():
    from proj.tasks import task_with_group       # .get()返回的结果是dict，里面是group子任务的信息
    from proj.tasks import task_with_groupapply  # 直接报错，int has not attribute app，直接返回的结果，而不是celery task
    
    gts = group(
        task_with_group.s(i) for i in range(2)
    )

    res = gts.apply_async()
    res = res.get()
    print(res)
    

def loop_use_grouptask():
    grouplist = []
    for i in range(3):
        grouplist.append(
            task_with_group.s(2).apply_async().get()
        )
    
    res = [g.apply_async().get() for g in grouplist]

    print(res)


if __name__ == "__main__":
    # single_processor()
    # multi_processor_state_check()
    # group_retry()
    # group_with_group()
    # use_grouptask()
    # use_grouptaskapply()
    # use_grouptaskapplybind()
    # use_grouptaskget()
    # group_with_grouptask()
    loop_use_grouptask()