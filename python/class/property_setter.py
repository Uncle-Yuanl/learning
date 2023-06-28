#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   property_settr.py
@Time   :   2023/02/13 10:02:18
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习下在类中使用装饰器property和对应的setter,delete
'''

import logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Student:
    def __init__(self):
        self._score = -1

    @property
    def score(self):
        if not 0 <= self._score <= 100:
            print('warning: no score')
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError("score must be a integer")
        if not 0 <= value <= 100:
            raise ValueError("score nust between 0 ~ 100")
        self._score = value


if __name__ == "__main__":
    g = Student()
    g.score = 99
    print(g.score)