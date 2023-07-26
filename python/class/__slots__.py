#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   __slots__.py
@Time   :   2023/07/18 15:39:42
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习__slots__的继承特性
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

class Parent:
    __slots__ = [
        "x",
        "y"
    ]


class Child(Parent):
    __slots__ = [
        "z"
    ] + Parent.__slots__


c = Child()
c.x = 1
print(c.x)
print(c.__slots__)