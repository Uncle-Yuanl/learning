#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   whatis_stream.py
@Time   :   2025/04/09 10:59:51
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   研究一下steam流式处理的本质
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
import uvicorn


app = FastAPI()


def generate_data():
    for i in range(1, 11):
        time.sleep(1)  # 模拟每秒生成一个块的耗时操作
        logger.info(f"prepare to yield {i}")
        yield f"FASTAPI Chunk {i}\n"


def make_data():
    return [f"New FASTAPI Chunk {i}\n" for i in range(10)]


@app.get("/stream")
async def stream_data():
    return StreamingResponse(generate_data(), media_type="application/octet-stream")
    # return StreamingResponse(make_data(), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)