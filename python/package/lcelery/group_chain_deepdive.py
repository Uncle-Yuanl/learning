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
from proj.tasks import add, mul, xconcat
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
    # res = gt().get()  # 直接报错：'int' object has no attribute 'app'
    gt = task_with_groupapply.delay(2)
    res = gt.collect()  # 'int' object has no attribute 'app'
    print(res)



def use_grouptaskcall():
    """直接报错：'int' object has no attribute 'app'
    """
    from proj.tasks import task_with_groupcall
    gt = task_with_groupcall.delay(2)
    res = list(gt.collect())
    print(res)


def use_grouptaskdelaycall():
    """res1和2都成功
    注意点：
        grouptask中，group中的subtask需要.s但是不能get也不能delay
    
    总结：
        1、@app.task不能使用get, delay
        2、group中subtask需要.s
        3、group直接call还是apply_async都可，就是别再get了
        4、主函数中用的时候，需要delay或者get + apply_async + get

    """
    from proj.tasks import task_with_groupdelaycall
    gt = task_with_groupdelaycall.delay(2)
    res1 = list(gt.collect())
    print(res1)
    gt = task_with_groupdelaycall.s(2)
    res2 = list(gt.apply_async().collect())
    print(res2)


def use_grouptaskapplybind():
    """直接报错：'int' object has no attribute 'app'
    """
    from proj.tasks import task_with_groupapplybind
    gt = task_with_groupapplybind.s(2)
    res = gt.apply_async().get()
    print(res)    


def use_grouptaskget():
    """一样的：'int' object has no attribute 'app'
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



def group_with_grouptaskdelaycall():
    """最后一个成功
    """
    from proj.tasks import task_with_groupdelaycall
    gt = group(
        task_with_groupdelaycall.s(i) for i in range(5, 8)
    )

    results = gt.apply_async()  # GroupResult
    res = results.get()
    print(res)        # 是可以直接get，不报错但是返回的是task id
    
    # gt = group(
    #     task_with_groupdelaycall.delay(i) for i in range(5, 8)
    # )
    # results = gt.apply_async()  # 报错
    # res = results.get()
    # print(res)

    gt = group(
        task_with_groupdelaycall.s(i) for i in range(5, 8)
    )
    results = gt()       # GroupResult
    res = results.get()  # 是可以直接get，不报错但是返回的是task id

    gt = group(
        task_with_groupdelaycall.s(i) for i in range(5, 8)
    )
    results = gt.apply()           # GroupResult
    res = results.get()            # res每个元素都是GroupResult
    res = [x.get() for x in res]   # Final result
    print(res)


def loop_use_grouptask():
    grouplist = []
    for i in range(3):
        grouplist.append(
            task_with_group.s(2).apply_async().get()
        )
    
    res = [g.apply_async().get() for g in grouplist]

    print(res)


def group_with_grouptaskallow():
    """不管task有没有return
    一样的：'int' object has no attribute 'app'
    """
    from proj.tasks import task_with_groupallowjoin
    
    gts = group(
        task_with_groupallowjoin.s(i) for i in range(2)
    )

    res = gts.apply_async()
    res = res.get()
    print(res)


def first_chain():
    """前面的结果作为第一个参数
    """
    c = chain(
        add.s(1, 2),
        # mul.s(3)
        xconcat.s(5)  # 3-5
    )

    results = c.apply_async()
    res = results.get()
    print(res)


def first_chain_back():
    """前面的结果作为第一个参数
    """
    downstream = xconcat.s(5)
    c = chain(
        add.s(1, 2),
        downstream  # 还是3-5
    )

    results = c.apply_async()
    res = results.get()
    print(res)


if __name__ == "__main__":
    # single_processor()
    # multi_processor_state_check()
    # group_retry()
    # group_with_group()
    # use_grouptask()
    # use_grouptaskapply()
    # use_grouptaskcall()
    # use_grouptaskdelaycall()
    # use_grouptaskapplybind()
    # use_grouptaskget()
    # group_with_grouptask()
    group_with_grouptaskdelaycall()
    # loop_use_grouptask()
    # group_with_grouptaskallow()
    # first_chain()
    # first_chain_back()