import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, DefaultHttpxClient
from openai import OpenAI, DefaultHttpxClient


load_dotenv(find_dotenv(".env"), override=True)
client = OpenAI(
    api_key=os.getenv("TRANSFER_KEY"),
    base_url=os.getenv("TRANSFER_URL") + "/v1/",
    http_client=DefaultHttpxClient(
        proxies="http://localhost:7890"
    )
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]
)

result = completion.choices[0].message

print(result)


# mode_client = OpenAI(
#     api_key=os.getenv("OPENAI_KEY"),
#     base_url="https://api.openai.com/v1/",
#     http_client=DefaultHttpxClient(
#         proxies="http://localhost:7890"
#     )
# )
# # No quota
# moderation = mode_client.moderations.create(
#     input=result.content
# )


# print()