#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   lsingleton.py
@Time   :   2024/09/02 11:01:24
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   研究下单例在多次实例化时，实例对象的变化
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


class MySingleton:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, feature) -> None:
        self.feature = feature


if __name__ == "__main__":
    ms1 = MySingleton("a")
    ms2 = MySingleton(2)
    print(f"id ms1: {id(ms1)}，feature {ms1.feature}")
    print(f"id ms2: {id(ms2)}，feature {ms2.feature}")