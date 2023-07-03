#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   overload.py
@Time   :   2023/07/03 22:38:46
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   深入理解装饰器，多装饰器
'''

import logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def decorator_a(func):
    def wrapper_a(*args, **kwargs):
        print('docator_a wrapper_a func')
        return func(*args, **kwargs) + 3
    return wrapper_a


def decorator_b(func):
    def wrapper_b(*args, **kwargs):
        print('docator_b wrapper_b func')
        return func(*args, **kwargs) * 3
    return wrapper_b


@decorator_b
@decorator_a
def raw_func(x):
    print('raw func')
    return x * 2


print(raw_func(2))


def new_raw_func(x):
    print('raw func')
    return x * 2


"""
由内向外装饰
由外向内执行（函数调用栈）
"""
deco_func = decorator_a(new_raw_func)
print(type(deco_func))
deco_func = decorator_b(deco_func)
print(type(deco_func))
print(deco_func(2))



# ======== @wrapper的作用 ========
def decorator_new_name(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@decorator_new_name
def raw_name(x):
    print(raw_name.__name__)  # wrapper
    print(raw_name.__code__.co_varnames)
    return x * 2

f = raw_name
try:
    print(f.__wrapped__)  # error 
except Exception as e:
    print(e)
print(f(2))


from functools import wraps
def decorator_keep_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator_keep_name
def raw_name(x):
    print(raw_name.__name__)  # raw_name
    print(raw_name.__code__.co_varnames)
    return x * 2  

f = raw_name
print(type(f), f.__wrapped__, id(f.__wrapped__))
print(id(f))
print(f(2))