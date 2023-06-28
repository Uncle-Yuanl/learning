#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   reconnect.py
@Time   :   2023/02/13 10:15:14
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习数据库自动重连，防止网络波动导致程序中断
'''

import logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

