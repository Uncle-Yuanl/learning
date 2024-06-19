#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   use_celery.py
@Time   :   2024/06/18 11:19:10
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实际调用celery实现多任务并发的脚本
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from celery import group, chain
from proj.tasks import add, mul, xconcat

def get_result_and_status():
    # Need result backend
    res = add.delay(2, 2)
    print(res.get(timeout=1))
    print(res.id)
    print(res.state)
    print(res.successful(), res.failed())


def get_with_error():
    res = add.delay(2, '2')
    res.get(timeout=1)
    # 没有任何信息，而且程序是执行完成的状态
    # res.get(propagate=False)


def learning_signature():
    """What is a signature:
    Celery wraps the arguments and execution options of a single task invocation.
    So it's called signature and can be passed to functions or even serialized and sent across the wire
    """
    # make a signature
    sig = add.signature((2, 2), countdown=10)
    print(type(sig)) # celery.canvas.Signature
    # shortcut
    sig = add.s(2, 2)

    # do the task with signature
    res = sig.delay()
    print(res.get(timeout=1))

    # incomplete partial, functiontool partial
    s2 = add.s(2)
    print(type(s2))
    res = s2.delay(8)
    print(res.get())

    # partial in what seqence
    s3 = xconcat.s('hello')
    res = s3.delay('world')
    print(res.get())  # world-hello, partial逆序


def learning_group_and_chain():
    """Concept
    A group calls a list of tasks in parallel with can retrive result in order.
    Chain: Tasks can be linked together so that after one task returns the other is called.
    """
    g1 = group(add.s(i, i) for i in range(10))
    print(type(g1))
    # Notice: 这边可以不同.delay() 而直接使用__call__()
    # 时间上没啥区别，毕竟有GIL
    print(g1().get())

    # chain 前一个任务的结果会传给后一个signature，类似shell中的xargs
    res = chain(add.s(4, 4) | mul.s(8))().get()
    print(res)
    
    # chain初始任务的签名
    c1 = chain(add.s(2) | mul.s(8))
    res = c1(2).get()
    print(res)

    # shortcut of chain
    res = (add.s(4, 2) | mul.s(8))().get()
    print(res)


if __name__ == '__main__':
    # get_result_and_status()
    # get_with_error()
    # learning_signature()
    learning_group_and_chain()