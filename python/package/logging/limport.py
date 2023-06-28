#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   limport.py
@Time   :   2023/05/12 14:30:38
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import sys
sys.path.append("/home/yhao/learning/python/func")
from lmodule import fun_with_logger


if __name__ == "__main__":
    fun_with_logger()