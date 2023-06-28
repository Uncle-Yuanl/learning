#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   overload.py
@Time   :   2023/02/07 15:48:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   通过实现python函数重载，学习命名空间、装饰器
'''

import logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# # ======================  查看命名空间 ===========================
# print('before func global: \n', globals())

def outer(a, b=10, *args, **kwargs):
    def inner(b):
        print(a + b)
    print('local: \n', locals())
    return inner


# print('after func global: \n', globals())


# # ======================= 封装函数类 ==============================
from inspect import getfullargspec


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        """检索function_map"""
        func = Namespace.get_instance().get(self.fn, *args)
        if not func:
            raise Exception('no matching function found！！！')
        return func(*args, **kwargs)            

    def key(self, args=None):
        if not args:
            # 传入函数名，以字符串列表的形式返回函数参数
            args = getfullargspec(self.fn).args

        return tuple([
            self.fn.__module__,
            self.fn.__class__,
            self.fn.__name__,
            len(args or [])
        ])


class Namespace:
    __instance = None

    def __init__(self):
        if self.__instance is None:
            self.function_map = {}
            Namespace.__instance = self
        else:
            raise Exception("cannot instantiate a virtual Namespace again！！！")

    @staticmethod
    def get_instance():
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn):
        """在虚拟命名空间中注册函数，返回Function类的可调用实例"""
        func = Function(fn)
        self.function_map[func.key()] = fn
        return func

    def get(self, fn, *args):
        """函数调用__call__时，检索map"""
        func = Function(fn)
        return self.function_map.get(func.key(args=args))


def overload(fn):
    # 相当于register就是wrapper函数了
    return Namespace.get_instance().register(fn)


@overload
def area(radius):
    return 3.14 * radius ** 2

@overload
def area(l, w):
    return l * w


# =========== 使用dispatch =================
from multipledispatch import dispatch

@dispatch(int)
def dispatch_area(radius):
    return 3 * radius ** 2


@dispatch(float)
def dispatch_area(radius):
    return 3.14 * radius ** 2


@dispatch(int, int)
def dispatch_area(l, w):
    return l * w


if __name__ == "__main__":
    # print(area(3))
    # print(area(3, 4))

    print(dispatch_area(3))
    print(dispatch_area(3.14))
    print(dispatch_area(3, 4))
