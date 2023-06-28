#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   distribution.py
@Time   :   2023/06/24 14:56:50
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习网络的分布式
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import os
import time, random, queue
from multiprocessing.managers import BaseManager


# 发送任务的队列
task_queue = queue.Queue()
print(f"进程： {os.getpid()}\t发送任务的队列的类型是： {type(task_queue)}\t地址为： {id(task_queue)}")

# 结果队列
result_queue = queue.Queue()


class QueueManager(BaseManager):
    pass


# 将两个队列注册到网络上
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口，设置验证码
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动queue
manager.start()

# 获取通过网络访问的queue对象
task = manager.get_task_queue()
print(f"进程： {os.getpid()}\t获取通过网络访问的task对象的类型是： {type(task)}\t地址为： {id(task)}")
result = manager.get_result_queue()
print(f"进程： {os.getpid()}\t获取通过网络访问的result对象的类型是： {type(result)}\t地址为： {id(result)}")

# 放几个任务
numdata = 60
for i in range(numdata):
    # n = random.randint(0, 10000)
    n = i
    print('Put task %d...' % n)
    task.put(n)

# 从result队列获取结果
print('Try get results...')
for i in range(numdata):
    r = result.get(timeout=60)
    print('Result: %s' % r)

# 关闭master
manager.shutdown()
print('master exit.')