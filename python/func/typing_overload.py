#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   typing_overload.py
@Time   :   2023/06/28 13:54:29
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习typing中的overload对参数类型检测的功能
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from typing import overload
from typing import Union, Optional


def double(input_: Union[int, list[int]]) -> Union[int, list[int]]:
    """输出的类型和输入类型是强相关的，不会说输入int输出list[int]
    当时该函数的类型检查无法做到
    任何试图使用int- only operations withx ，比如除法，都会导致类型检查失败。为了修复这种错误，我们将被迫使用类型缩小
    """
    if isinstance(input_, int):
        return input_ * 2
    elif isinstance(input_, list):
        return [i * 2 for i in input_]
    else:
        raise TypeError
    

# mypy结果：note: Revealed type is "Union[builtins.int, builtins.list[builtins.int]]"
x = double(12)
reveal_type(x)


# ================ 使用typing overload ====================
@overload
def overload_double(input_: int) -> int:
    """overload装饰器只做函数声明，不做定义
    """
    ...


@overload
def overload_double(input_: list[int]) -> list[int]:
    ...


def overload_double(input_: Union[int, list[int]]) -> Union[int, list[int]]:
    """函数定义
    """
    if isinstance(input_, int):
        return input_ * 2
    elif isinstance(input_, list):
        return [i * 2 for i in input_]
    else:
        raise TypeError

x = overload_double(12)
reveal_type(x)

x = overload_double([11, 12])
reveal_type(x)


def issue_double(input_: Union[int, list[int]]) -> Union[int, list[int]]:
    if isinstance(input_, int):
        return ('1', '2')
    elif isinstance(input_, list):
        return ('1', '2')
    else:
        return ('1', '2')

x = issue_double(12)
reveal_type(x)  # python运行会报错


if __name__ == '__main__':
    x = issue_double(12)