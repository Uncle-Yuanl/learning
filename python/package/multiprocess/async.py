#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   async.py
@Time   :   2023/06/26 10:10:52
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用async和await完成异步IO
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import time
import asyncio
import threading


async def my_sleep(s):
    time.sleep(s)
    return 'my sleep'


async def hello():
    print('start hello: %s' % threading.currentThread())
    # r = await asyncio.sleep(2)
    # r = my_sleep(2)
    r = await my_sleep(2)  # await获取函数的返回值
    print(f"await的结果类型为： {type(r)}, 值为： {r if r is not None else 'None'}")
    print('finish hello: %s' % threading.currentThread())


# # 获取EventLoop:
# loop = asyncio.get_event_loop()
# # 多个任务，多个生成器
# tasks = [hello(), hello()]
# print(f"加上async后，生成器变为了： {type(tasks[0])}")
# # 执行coroutine，注意传入的是生成器对象
# loop.run_until_complete(asyncio.wait(tasks))
# loop.close()


# HTTP请求实践
async def wget(host):  # generator可以传参
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)  # <class 'coroutine'>，执行函数，发出IO请求，没有await
    print('type connect', type(connect))

    reader, writer = await connect  # 发起请求，等待结果 【耗时】
    print('type reader, writer', type(reader), type(writer))
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    await writer.drain()
    while True:
        line = await reader.readline()
        if line == b'\r\n':
            break
        # 【问题】这个地方为什么没有因为异步并发而乱序
        # 【理解】可能还在循环中，且没有await到其他协程中
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # ignore the body, close the socket
    writer.close()

loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()