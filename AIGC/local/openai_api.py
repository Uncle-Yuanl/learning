#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   openai_api.py
@Time   :   2025/04/08 14:56:20
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   使用openai包调用自己封装的接口
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, DefaultHttpxClient


load_dotenv(find_dotenv(".env"), override=True)
client = OpenAI(
    api_key=os.getenv("TRANSFER_KEY"),  # 这个不能少，还不能为空
    base_url="http://127.0.0.1:8000/v1/",
    http_client=DefaultHttpxClient(
        proxies="http://localhost:7890"
    )
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]
)

result = completion.choices[0].message.content

print(result)
