#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   async_openai_api.py
@Time   :   2025/04/08 17:27:58
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   异步，主要支持stream=True情况
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
from dotenv import load_dotenv, find_dotenv
import asyncio
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


load_dotenv(find_dotenv(".env"), override=True)
client = AsyncOpenAI(
    api_key=os.getenv("TRANSFER_KEY"),  # 这个不能少，还不能为空
    base_url="http://127.0.0.1:8000/v1/",
    http_client=DefaultAsyncHttpxClient(
        proxies="http://localhost:7890"
    )
)

async def get_completion():

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ],
        timeout=600,
        max_tokens=4096,
        temperature=0.0,
        stream=True  # 加了就错
    )

    async for chunk in completion:
        print(chunk)

    print()


if __name__ == "__main__":
    asyncio.run(get_completion())