#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   image_tasks.py
@Time   :   2024/07/01 14:18:10
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   实际处理图片的任务
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import numpy as np
import cv2
import time
from celery import subtask
from celery import group
from celery.result import allow_join_result
from .celery import app


@app.task
def rescale_func(image, scale, method):
    nw = int(image.shape[1] / scale)
    nh = int(image.shape[0] / scale)
    dsize = (nw, nh)

    newimg = cv2.resize(src=image, dsize=dsize, interpolation=method)

    return newimg


@app.task
def image_tobytes(image):
    return image.tobytes()


@app.task
def rescale_image(image, scales):
    if isinstance(image, bytes):
        image = cv2.imdecode(
            np.frombuffer(image, np.uint8),
            cv2.IMREAD_COLOR
        )

    # # 还是会遇到scales = [2,3]，实际函数2,3,3,3
    # gts = group(
    #     group(
    #         rescale_func.s(image, scale, method) \
    #             for method in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
    #     ) for scale in scales
    # )
    
    # 这个顺序才是对的
    group_list = []
    for scale in scales:
        for method in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
            group_list.append(subtask(rescale_func.s(image, scale, method)))

    gts = group(group_list)

    with allow_join_result():
        results = gts.apply_async().get()

    return results


@app.task
def different_imgtypes(image, scales):
    if isinstance(image, bytes):
        image = cv2.imdecode(
            np.frombuffer(image, np.uint8),
            cv2.IMREAD_COLOR
        )

    if image.shape[0] == 1440:
        results = group(
            image_tobytes.s(image)
        )
        return results.apply_async()

    group_list = []
    for scale in scales:
        for method in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
            group_list.append(subtask(rescale_func.s(image, scale, method)))

    results = group(group_list)

    return results.apply_async()


@app.task
def different_imgtypes_wait(image, scales):
    time.sleep(20)
    if isinstance(image, bytes):
        image = cv2.imdecode(
            np.frombuffer(image, np.uint8),
            cv2.IMREAD_COLOR
        )

    if image.shape[0] == 1440:
        results = group(
            image_tobytes.s(image)
        )
        return results.apply_async()

    group_list = []
    for scale in scales:
        for method in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
            group_list.append(subtask(rescale_func.s(image, scale, method)))

    results = group(group_list)

    return results.apply_async()