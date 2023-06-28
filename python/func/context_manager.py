#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   context_manager.py
@Time   :   2023/05/11 16:27:24
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
from time import time
from contextlib import contextmanager


@contextmanager 
def timer(name): 
    start_time = time() 
    yield 
    cost = round(time() - start_time, 2) 
    logger.info('[{}] done in {} s'.format(name, cost))

    return locals()


with timer('how to catch name'):
    print('a')