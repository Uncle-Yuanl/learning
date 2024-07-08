#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   image_tasks.py
@Time   :   2024/07/01 14:15:44
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   处理图片，报错bytes和base64编码
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import time
from celery import group


def check_stuck_bug():
    """在线上dev时，celery接收到任务并处理，但是主进程接受结果时卡住

    现象：
        结果是能拿得到的，返回的是ndarray
    """
    imgpaths = [
        "/home/yhao/Pictures/uitest/4lin2/HM Vegan Mayo.png",
        "/home/yhao/Pictures/uitest/4lin2/Praise_1.png"
    ]
    
    images = []
    for imgpath in imgpaths:
        with open(imgpath, "rb") as f:
            images.append(f.read())

    print(type(images[0]))

    from proj.image_tasks import rescale_image
    newimgs = group(
        rescale_image.s(img, [2, 3]) for img in images
    )
    
    results = newimgs.apply_async().get()

    print()


def get_difftype_images():
    imgpaths = [
        "/home/yhao/Pictures/uitest/4lin2/HM Vegan Mayo.png",
        "/home/yhao/Pictures/uitest/4lin2/Praise_1.png"
    ]
    images = []
    for imgpath in imgpaths:
        with open(imgpath, "rb") as f:
            images.append(f.read())

    print(type(images[0]))

    from proj.image_tasks import different_imgtypes
    newimgs = group(
        different_imgtypes.s(img, [2, 3]) for img in images
    )
    
    results = newimgs.apply().get()

    print()


def get_difftype_images_taskid():
    imgpaths = [
        "/home/yhao/Pictures/uitest/4lin2/HM Vegan Mayo.png",
        "/home/yhao/Pictures/uitest/4lin2/Praise_1.png"
    ]
    images = []
    for imgpath in imgpaths:
        with open(imgpath, "rb") as f:
            images.append(f.read())

    print(type(images[0]))

    from proj.image_tasks import different_imgtypes_wait
    newimgs = group(
        different_imgtypes_wait.s(img, [2, 3]) for img in images
    )
    
    asyncresults = newimgs.apply_async()

    while not all(r.status == "SUCCESS" for r in asyncresults):
        time.sleep(10)
        print("waiting")

    groupresults = [ag.get() for ag in asyncresults]
    # 必须得先写个ready()或者successful()
    res = [x.get() for x in groupresults if x.ready()]
    print(len(res))
    print(type(res[0]), type(res[0][0]))
    print(type(res[1]), type(res[1][0]))


if __name__ == "__main__":
    # check_stuck_bug()
    # get_difftype_images()
    get_difftype_images_taskid()