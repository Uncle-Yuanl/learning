#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   app.py
@Time   :   2025/04/17 16:51:33
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   flask main
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import time
import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host="redis", port=6379)


def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)


@app.route("/")
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count)