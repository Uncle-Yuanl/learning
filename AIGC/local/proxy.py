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

    # response = requests.post(
    #     url=unified_api,
    #     headers=headers,
    #     data=json.dumps(query)
    # )
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         url=unified_api,
    #         headers={"Content-Type": "application/json", **headers},
    #         data=json.dumps(query)
    #     )
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(
            url=unified_api,
            data=json.dumps(query)
        ) as response:
            status = response.status
            if status == 200:
                if query.get("stream"):
                    return StreamingResponse(await response.text(), media_type="text/event-stream")
                else:
                    return JSONResponse(await response.json())
            else:
                raise Exception(f"Request failed: {response.status_code} {response.text}")