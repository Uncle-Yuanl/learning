#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   proxy_upgrade.py
@Time   :   2025/04/23 16:41:51
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   升级版代理
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
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from aiolimiter import AsyncLimiter

from .proxy import generate_access_token


load_dotenv(find_dotenv(".env"), override=False)
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_OPENAI_SUBSCRIPTION_KEY = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
CONCURRENT_LIMIT = 1000  # 限制并发请求数


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # 初始化共享aiohttp session
    app.state.session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=1000,
            keepalive_timeout=60 * 5
        ),
        timeout=aiohttp.ClientTimeout(
            total=60 * 10
        )
    )
    yield

    # shutdown
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


async def stream_generator(response: aiohttp.ClientResponse):
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


@app.post("/v1/chat/completions")
async def get_completion(
    query: Dict,
    request: Request, 
    model: str = "gpt-4o-mini"
) -> Response:
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
    unified_api = f"https://bnlwe-ai03-q-931039-apim-01.azure-api.net/openai4/az_openai_{model}_chat"
    session = request.app.state.session
    data = json.dumps(query)

    async with AsyncLimiter(CONCURRENT_LIMIT):
        if query.get("stream"):
            response = await session.post(
                url=unified_api,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=600)
            )
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            return StreamingResponse(stream_generator(response), media_type="text/event-stream")
        else:
            async with session.post(
                url=unified_api,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                else:
                    return JSONResponse(await response.json())