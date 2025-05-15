#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   proxy.py
@Time   :   2025/04/08 11:42:50
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   当我们的接口不符合openai标准接口时，使用fastapi套壳转换
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
from typing import Optional, Dict
from dotenv import find_dotenv, load_dotenv
import json
import requests
import httpx, aiohttp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

load_dotenv(find_dotenv(".env"), override=False)
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_OPENAI_SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World", "AZURE_TENANT_ID": AZURE_TENANT_ID}


@app.get("/items/{item_id}")
async def get_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id}


async def generate_access_token(client_id, client_secret):
    auth_endpoint = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
    }
    response = requests.post(auth_endpoint, data=data)

    if response.ok:
        access_token = response.json()["access_token"]
        return access_token
    else:
        raise Exception(f"Authentication failed: {response.json()}")


async def stream_generator(session, response):
    """以生成器的方式返回chunk，而不是一次性获取所有chunk，再封装
    """
    try:
        # async for chunk in response.content.iter_chunked(1024):
        async for line in response.content:
            decoded_line = line.decode("utf-8")
            logger.info(f"中转站转发： {decoded_line[:10]}")
            yield decoded_line
    except Exception as e:
        logger.error(f"流转发出错：{e}")
        yield "event: error\ndata: [STREAM ERROR]\n\n"
    finally:
        await response.release()  # 确保释放连接
        await session.close()


@app.post("/v1/chat/completions")
async def get_completion(query: Dict, model="gpt-4o-mini"):
    """Call the OpenAI API to get a completion for the given prompt."""
    access_token = await generate_access_token(AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    if access_token is None:
        return None
    subs_key = AZURE_OPENAI_SUBSCRIPTION_KEY
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Ocp-Apim-Subscription-Key" : subs_key,
        "Content-Type": "application/json",
        "User-Agent": ""  # 对aiohttp有效，httpx无效
    }
    env = 'prod' if AZURE_OPENAI_SUBSCRIPTION_KEY=='fa3bfc057f4946f09099ebc6214f564a' else 'uat'
    unified_api = f"https://aiflmapi{env}.unilever.com/openai4/az_openai_{model}_chat"

    # async with aiohttp.ClientSession(headers=headers) as session:
    #     async with session.post(
    #         url=unified_api,
    #         data=json.dumps(query)
    #     ) as response:
    #         status = response.status
    #         if status == 200:
    #             if query.get("stream"):
    #                 gen = stream_generator(response)
    #                 # await response.text()在代理端等待实际服务端的所有结果返回
    #                 # 即使在客户端是chunk形式，但是高并发情况下代理端很容易崩掉
    #                 # return StreamingResponse(await response.text(), media_type="text/event-stream")
    #                 logger.critical(f"response closed")
    #                 return StreamingResponse(gen, media_type="text/event-stream")
    #             else:
    #                 return JSONResponse(await response.json())
    #         else:
    #             text = await response.text()
    #             logger.error(f"OpenAI proxy请求失败: {response.status}, {text}")
    #             raise Exception(f"Request failed: {response.status_code} {response.text}")
    #     logger.critical(f"session closed")
    try:
        session = aiohttp.ClientSession(headers=headers)
        response = await session.post(
            url=unified_api,
            data=json.dumps(query),
            timeout=aiohttp.ClientTimeout(total=600)
        )
        if response.status == 200:
            if query.get("stream"):
                return StreamingResponse(
                    stream_generator(session, response),
                    media_type="text/event-stream"
                )
            else:
                result = await response.json()
                await session.close()
                return JSONResponse(result)
        else:
            text = await response.text()
            await session.close()
            raise Exception(status_code=response.status, detail=text)
    except Exception as e:
        await session.close()
        raise e