#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   function_call.py
@Time   :   2024/07/12 12:03:25
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   工具调用，本质调两次对话
            一次判断使用工具
            一次prompt加上工具结果，再生成最终结果
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
import json
import deepl
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, DefaultHttpxClient

load_dotenv(find_dotenv(".env"), override=True)

client = OpenAI(
    api_key=os.getenv("TRANSFER_KEY"),
    base_url=os.getenv("TRANSFER_URL") + "/v1/",
    http_client=DefaultHttpxClient(
        proxies="http://localhost:7890"
    )
)
translator = deepl.Translator(os.getenv("DEEPL_KEY"))

# Example dummy function hard coded to return the translated content
# In production, this could be your backend API or an external API
def get_translate(content, country):
    """Translate the content to the target language"""
    if "german" == country.lower():
        result = translator.translate_text(content, target_lang="DE")
        return json.dumps({"country": country, "translate": result.text})
    elif "french" in country.lower():
        result = translator.translate_text(content, target_lang="FR")
        return json.dumps({"country": country, "translate": result.text})
    else:
        return json.dumps({"country": country, "translate": "unknown"})


def run_conversation():
    # Step 1: send the message and functions to model
    messages = [
        {"role": "user", "content": "Please translate the following sentences to French:\nHello! How are you?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_translate",
                "description": "Translate the content to target language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": ""
                        },
                        "country": {
                            "type": "string",
                            "enum": ["GERMAN", "FRENCH"]
                        }
                    },
                    "required": ["content", "country"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_translate": get_translate,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's 
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]

            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                content=function_args.get("content"),
                country=function_args.get("country")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                }
            )  # extend conversation with function response
        
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages
        )

        return second_response
    

result = run_conversation()
print(result)
print()