#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   lmodule.py
@Time   :   2023/05/12 14:29:20
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# 有这个就不会错了
logger = logging.getLogger(f'【{__file__}】')

def fun_with_logger():
    logger.info(f"{__file__}")


if __name__ == "__main__":
    fun_with_logger()