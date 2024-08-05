#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   text.py
@Time   :   2024/08/01 11:28:07
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   给图片加title
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')


import cv2
import numpy as np


def add_title(
    image, title, thickness=1,
    bgcolor=255,
    fontbgr=(255, 144, 30)
    ):
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 选择字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 获取文本大小
    font_scale = height / 500
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    
    # 计算文本位置（居中）
    margin = int(font_scale * 10)
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + margin  # 10是顶部边距
    
    # 在图像顶部创建一个白色背景条
    white_space = np.ones((text_y + margin, width, 3), dtype=np.uint8) * bgcolor
    image = cv2.vconcat([white_space, image])
    
    # 添加文本
    cv2.putText(image, title, (text_x, text_y), font, font_scale, fontbgr, thickness, cv2.LINE_AA)
    
    # 画线
    height, width = image.shape[:2]
    lineloc = width // 3
    cv2.line(image, (lineloc, 0), (lineloc, height), (0, 205, 102), thickness=width//40)
    cv2.line(image, (lineloc * 2, 0), (lineloc * 2, height), (0, 205, 102), thickness=width//40)
    return image


def draw_line(image):
    height, width = image.shape[:2]
    lineloc = width // 3
    # cv2.line(image, (lineloc, 0), (lineloc, height), (0, 205, 102), thickness=width//400)
    # cv2.line(image, (lineloc * 2, 0), (lineloc * 2, height), (0, 205, 102), thickness=width//400)
    
    image = cv2.resize(
        image,
        dsize=(width // 3, height // 3)
    )

    return image


if __name__ == "__main__":
    # img = cv2.imread('/home/yhao/Pictures/C1V2.png')
    # print(img.shape)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image = add_title(img, "TEST")
    # print(image.shape)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    img = cv2.imread("/home/yhao/Pictures/Hellman VAS Test/shelf1.png")
    image = draw_line(img)
    
    print()