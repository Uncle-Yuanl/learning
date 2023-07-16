#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   coroutine.py
@Time   :   2023/06/25 16:53:50
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   协程、生成器概念
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


def consumer():
    r = 'init'
    while True:
        print('2')
        # receive the param from c.send(n)，and assign the value to n
        # Caution：yield = baton relay
        n = yield r
        print(f"result yielded： {n}")
        if not n:
            return
        print('[CONSUMER] Consuming %s..., and doing something' % n)
        r = n * n

    
def producer(c):
    """
    Args:
        c: the generator object
    """
    r = c.send(None)  # start the generator, and consumer [go into] while loop
    print('[PRODUCER] Consumer return: %s' % r)
    n = 0
    while n < 2:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        # image that send a baton relay
        # cosumer yield r and give this here
        # and consumer [already in] while loop
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()


c = consumer()
print(type(c))
producer(c)


# ========================= 原理 =====================================
# 生成器之间的切换 除了此案例还有OneNote上的send方法
def generator():
    n = 0
    while n < 10:
        print(f'生产者中, 生成第{n + 1} 个结果...')
        yield n
        time.sleep(2)

    yield None


def consumer(gen):
    while True:
        n = gen.__next__()
        print(f'消费者中, 消费第{n + 1} 个结果...')


# 交替【切换】进行
gen = generator()
consumer(gen)
