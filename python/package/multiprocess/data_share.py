"""
主要研究多进程如何方便地聚合各自的数据
"""

import multiprocessing
import time
from contextlib import contextmanager
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@contextmanager
def timer(name):
    start = time.time()
    yield
    cost = round(time.time() - start, 2)
    logger.info(f"[{name}] done in {cost} s")


def func(data):
    """比较耗时的任务
    """
    for d in data:
        time.sleep(0.01)

    return data


def func_iterable(d):
    """imap的处理对象是iterable的每个生成对象
    """
    time.sleep(0.01)

    return d


def func_iterable_ma(d, arg):
    """imap的处理对象是iterable的每个生成对象
    """
    time.sleep(0.01)

    return d, arg


def queue_callback(data):
    """回调函数，将结果放入队列
    """
    queue.put(data)
    logger.info(f'队列中添加数据量： {len(data)}')


def manager_callback(data):
    """回调函数，将结果放入服务器进程
    """
    manage_list.extend(data)
    logger.info(f'服务器进程list中添加数据量： {len(data)}')


def apply_method(data, callback, numprocess=4):
    """通过传入不同callback来测试不同容器的通信特点
    结论： 
        queue: 起进程会慢一些，然后非常快。但是从queue中取数据很烦
        manager: 
    """
    pool = multiprocessing.Pool(numprocess)
    with timer('apply_async + Queue'):
        for i in range(numprocess):
            numperprocess = len(data) // numprocess + 1
            pool.apply_async(func, args=(data[i * numperprocess : (i + 1) * numperprocess], ), callback=callback)
        pool.close()
        pool.join()


def func_iterable_ma(d, arg):
    """imap的处理对象是iterable的每个生成对象
    """
    time.sleep(0.01)

    return d, arg


def func_iterable_mal(data, arg):
    """imap的处理对象是iterable的每个生成对象
    """
    for d in data:
        time.sleep(0.01)

    return data, arg


def outer(td):
    return func_iterable_ma(*td)


def list_outer(tld):
    return func_iterable_mal(*tld)

    
def get_data_directly(data, func, numprocess=4):
    """学习UCPhrase的写法，直接获取结果
    """
    pool = multiprocessing.Pool(numprocess)
    pool_func = pool.imap(func, data)
    with timer('imap直接接受结果'):
        # results = list(pool_func)
        results = [r for r in pool_func]  # 两个结果一样
        pool.close()
        pool.join()

    return results


if __name__ == "__main__":
    logger = logging.getLogger('【多进程学习】')

    num_cores = multiprocessing.cpu_count()
    logger.info(f'cpu核心数为： {num_cores}')

    queue = multiprocessing.Queue()
    manage_list = multiprocessing.Manager().list()  # 代理对象
    # print(type(queue), type(mqueue))

    data = list(range(2000))

    # # 20s
    # with timer('单进程遍历'):
    #     res = func(data)

    # # 7s
    # apply_method(data, queue_callback)
    # print(queue.qsize())

    # # 7s
    # apply_method(data, manager_callback)
    # print(len(manage_list))

    results = get_data_directly(data, func_iterable)
    print(len(results), type(results[0]))

    # 测试多参数的imap
    # data = ((i, 'a') for i in data)
    # results = get_data_directly(data, outer)

    # # 测试多参数的，且参数为list的imap
    # data = [([1,2,3], 'a'), ([4,5,6], 'b')]
    # data = (x for x in data)
    # results = get_data_directly(data, list_outer)

    print('end')