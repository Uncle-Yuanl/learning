#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   asyncio_triton.py
@Time   :   2024/01/11 10:12:24
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   tritonclient异步实例
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import numpy as np
from typing import Union, Dict, Sequence
from transformers import AutoTokenizer, DebertaV2Tokenizer
import asyncio
import ssl
# import tritonclient.http as tritonhttpclient
import tritonclient.http.aio as tritonhttpclient


def make_triton_data(data, max_length=512):
    input_id = tritonhttpclient.InferInput("input_ids", (1, max_length), "INT16")
    input_id.set_data_from_numpy(data["input_ids"].astype(np.int16))
    input_mask = tritonhttpclient.InferInput("attention_mask", (1, max_length), "INT16")
    input_mask.set_data_from_numpy(data["attention_mask"].astype(np.int16))
    inputs = [input_id, input_mask]

    output_logits = tritonhttpclient.InferRequestedOutput(name="logits")
    output_teacher = tritonhttpclient.InferRequestedOutput(name="teachers")
    outputs = [output_logits, output_teacher]

    return inputs, outputs


async def score_triton_wrapper(
    triton_client,
    modelname,
    inputs,
    outputs,
    headers
):
    logger.info(f"{modelname} single infer start")
    result = await triton_client.infer(modelname, inputs, outputs=outputs, headers=headers)
    logger.info(f"{modelname} single infer done")
    return result


async def score_triton_kpi_list(
    inputs,
    outputs
) -> Union[Dict[str, int], Sequence[np.ndarray]]:
    triton_client = tritonhttpclient.InferenceServerClient(
        url='tritonlearning.northeurope.inference.ml.azure.com/',
        verbose=True,
        ssl=True,
        # ssl_context_factory=gevent.ssl._create_default_https_context
        ssl_context=ssl.create_default_context()
    )
    headers = {
        'Content-Type':'application/json', 
        'Authorization': 'Bearer ' + "RDRVwHJ37HqEshDsW9IyF4b3N2Ve4hOl",
        'azureml-model-deployment': "us-cerberus-triton"
    }
    
    # infer kpis
    tasks = []
    for kpi in ["Distinct_Proposition", "Attention_Catching", "Message_Connection"]:
        tasks.append(
            asyncio.ensure_future(
                score_triton_wrapper(
                    triton_client,
                    kpi,
                    inputs,
                    outputs=outputs,
                    headers=headers
                )
            )
        )

    logger.info("gather start.....")
    results = await asyncio.gather(*tasks)
    results = [np.argmax(result.as_numpy("logits")) for result in results]
    logger.info("gather done.....")
    await triton_client.close()

    return results[:-1], results[-1]


async def score_triton_kpi_list_wofuture(
    inputs,
    outputs
) -> Union[Dict[str, int], Sequence[np.ndarray]]:
    triton_client = tritonhttpclient.InferenceServerClient(
        url='tritonlearning.northeurope.inference.ml.azure.com/',
        verbose=True,
        ssl=True,
        # ssl_context_factory=gevent.ssl._create_default_https_context
        ssl_context=ssl.create_default_context()
    )
    headers = {
        'Content-Type':'application/json', 
        'Authorization': 'Bearer ' + "RDRVwHJ37HqEshDsW9IyF4b3N2Ve4hOl",
        'azureml-model-deployment': "us-cerberus-triton"
    }
    
    # infer kpis
    tasks = []
    for kpi in ["Distinct_Proposition", "Attention_Catching", "Message_Connection"]:
        tasks.append(
            score_triton_wrapper(
                triton_client,
                kpi,
                inputs,
                outputs=outputs,
                headers=headers
            )
        )

    logger.info("gather start.....")
    results = await asyncio.gather(*tasks)
    results = [np.argmax(result.as_numpy("logits")) for result in results]
    logger.info("gather done.....")
    await triton_client.close()

    return results[:-1], results[-1]


async def main_list():
    """Also multi asyncio.run
    """
    kpiresults, cvm_result_model = asyncio.run(
        score_triton_kpi_list(
            *make_triton_data(data)
        )
    )

    kpiresults, cvm_result_model = asyncio.run(
        score_triton_kpi_list_wofuture(
            *make_triton_data(data)
        )
    )


async def main_nested():
    pass


if __name__ == "__main__":
    tokenizer = DebertaV2Tokenizer.from_pretrained("/media/data/pretrained_models/deberta-v3-base")
    sentence = "test"
    data = tokenizer(
        sentence, padding="max_length", truncation=True,
        max_length=512, return_tensors="np"
    )
