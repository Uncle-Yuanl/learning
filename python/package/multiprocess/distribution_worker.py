#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   distribution_worker.py
@Time   :   2023/06/25 00:01:36
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   分布式worker,会在自己的windows上也有一份
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import time, sys, queue
from multiprocessing.managers import BaseManager

# 创建类似的QueueManager
class QueueManager(BaseManager):
    pass


# 只从网络上获取Queue对象，因此注册时只提供名称
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

# 连接到服务器，看各自ip
server_addr = '127.0.0.1'
machine = 'localhost'
print('Connect to server %s...' % server_addr)

# 端口、验证码需要与master一致
m = QueueManager(address=(server_addr, 5000), authkey=b'abc')

# 从网络连接
m.connect()

# 获取Queue对象
task = m.get_task_queue()
result = m.get_result_queue()

# 干活
# for i in range(60):
while True:
    try:
        n = task.get(timeout=1)
        print(f"worker: {machine} run task %d * %d..." % (n, n))
        r = '%d * %d = %d' % (n, n, n * n)
        time.sleep(1)
        result.put(r)
        time.sleep(1)  # 给windows机器上场的机会
    except EOFError:
        print('task queue is empty.')
        break

# 处理结束
print(f"worker: {machine} exit.")
