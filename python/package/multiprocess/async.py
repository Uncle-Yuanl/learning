#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   async.py
@Time   :   2023/06/26 10:10:52
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用async和await完成异步IO
            特别需要注意的是：在异步函数体中，尽量不要使用普通函数
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


# ========================= 简单案例：定义、执行 =======================
async def my_sleep(s):
    print('in coroutine')
    time.sleep(s)
    return 'my sleep'

# loop = asyncio.get_event_loop()
# print('in main')
# result = loop.run_until_complete(my_sleep(2))
# print(result)
# loop.close()


# ========================= 协程之间的调用 ===================
async def hello_async():
    print('start hello: %s' % threading.currentThread())
    r = await asyncio.sleep(2)  # await获取asyncio异步函数的返回值
    print(f"await的结果类型为： {type(r)}, 值为： {r if r is not None else 'None'}")
    print('finish hello: %s' % threading.currentThread())

async def hello_self():
    print('start hello: %s' % threading.currentThread())
    r = await my_sleep(2)  # await获取自定义异步函数的返回值
    print(f"await的结果类型为： {type(r)}, 值为： {r if r is not None else 'None'}")
    print('finish hello: %s' % threading.currentThread())


# # 获取EventLoop:
# loop = asyncio.get_event_loop()
# # 多个任务，多个生成器
# tasks = [hello_async(), hello_async()]
# print(f"加上async后，生成器变为了： {type(tasks[0])}")
# # 执行coroutine，注意传入的是生成器对象
# loop.run_until_complete(asyncio.wait(tasks))    # 异步执行，多个start hello
# print("*" * 10)
# tasks = [hello_self(), hello_self()]
# loop.run_until_complete(asyncio.wait(tasks))    # 顺序执行，第一个hello整体结束，再调第二个
# loop.close()


# ======================== 尝试gather方式，还是不行 ====================
async def gather_tasks():
    tasks = [hello_self(), hello_self()]
    await asyncio.gather(*tasks)

# # 顺序执行，第一个hello整体结束，再调第二个
# asyncio.run(gather_tasks())


# ======================= 为什么自定义的不行呢 ==========================
# 在协程中调用普通函数，会破坏协程的非阻塞
# 要么都是用协程异步函数，要么使用loop.call_soon call_later等函数

async def my_sleep_async(s):
    print('in coroutine')
    # time.sleep(s)
    await asyncio.sleep(s)
    return 'my sleep'

async def hello_async():
    print('start hello: %s' % threading.currentThread())
    r = await my_sleep_async(3)
    print("finish hello")


async def hello_call_soon(loop):
    print('start hello: %s' % threading.currentThread())
    loop.call_soon(time.sleep(2))
    print("finish hello")


# loop = asyncio.get_event_loop()
# # 实现异步
# tasks = [hello_async(), hello_async()]
# loop.run_until_complete(asyncio.wait(tasks))

# # 直接报错
# tasks = [hello_call_soon(loop), hello_call_soon(loop)]
# loop.run_until_complete(asyncio.wait(tasks))
# loop.close()


# ========================= 失败案使用async和await =======================
async def hello_failed():
    print('start hello: %s' % threading.currentThread())
    r = await my_sleep(3)  # await获取函数的返回值
    print(f"await的结果类型为： {type(r)}, 值为： {r if r is not None else 'None'}")
    print('finish hello: %s' % threading.currentThread())


async def hello_failed_too():
    print('start hello: %s' % threading.currentThread())
    await my_sleep(3)  # await获取函数的返回值
    print(f"not assgin")
    print('finish hello: %s' % threading.currentThread())


async def my_print(s, name):
    time.sleep(s)
    return f'Hello {name}'


async def hello_without_print_in_main():
    r1 = await my_print(3, 'P1')  # await获取函数的返回倷
    print(f"r1: {r1}")
    
    r2 = await my_print(1, 'P2')
    print(f"r2: {r2}")


# # 获取EventLoop:
# loop = asyncio.get_event_loop()
# # print("*" * 10)
# loop.run_until_complete(hello_without_print_in_main())
# loop.close()


# ========================= HTTP请求实践 ==========================
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

# loop = asyncio.get_event_loop()
# tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com']]
# loop.run_until_complete(asyncio.wait(tasks))
# loop.close()


# ========================== aiohttp ==========================
import aiohttp
import asyncio

from time import time

async def base_main():

    async with aiohttp.ClientSession() as session:

        demo_url = 'http://192.168.8.117:7071/api/ConceptAIPredictionService'
        async with session.post(demo_url, data={"country": "UK","content": "Knorr chicken stock","api": "Lower Mainstream"}) as resp:
            pokemon = await resp.text()
            # print(len(pokemon))


async def multi_main():

    async with aiohttp.ClientSession() as session:

        demo_url = 'http://192.168.8.117:7071/api/ConceptAIPredictionService'

        for i in range(100):
            async with session.post(demo_url, data={"country": "UK","content": "Knorr chicken stock","api": "Lower Mainstream"}) as resp:
                pokemon = await resp.text()
                # print(len(pokemon))


async def get_api_result(session, demo_url):
    async with session.post(demo_url, data={"country": "UK","content": "Knorr chicken stock","api": "Lower Mainstream"}) as resp:
        pokemon = await resp.text()
        return pokemon


async def multi_stack_main():
    """multi request and get the response
    """
    async with aiohttp.ClientSession() as session:
        demo_url = 'http://192.168.8.117:7071/api/ConceptAIPredictionService'

        tasks = []
        for i in range(100):
            tasks.append(asyncio.ensure_future(get_api_result(session, demo_url)))

        results = await asyncio.gather(*tasks)
        # for pokemon in results:
        #     print(pokemon)


# start = time()
# asyncio.run(multi_main())
# print(f"time cose: {time() - start}")

# start = time()
# # asyncio.run(base_main())
# # asyncio.run(multi_main())
# asyncio.run(multi_stack_main())  # 没有快特别多，可能因为api响应太快
# print(f"time cose: {time() - start}")


