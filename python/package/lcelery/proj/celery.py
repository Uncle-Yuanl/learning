#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   celery.py
@Time   :   2024/06/17 17:15:51
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   celery启动app配置脚本
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

from celery import Celery

app = Celery(
    'proj',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2',
    include=[
        'proj.tasks',
        "proj.complex_tasks"
    ]
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()