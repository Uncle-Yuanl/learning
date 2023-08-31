# Use a pipeline as a high-level helper
import torch
from transformers import pipeline

model="/media/data/pretrained_models/Llama-2-7b-chat-hf"
pipe = pipeline(
    "text-generation",
    model="/media/data/pretrained_models/Llama-2-7b-chat-hf",
    device="cuda",
    torch_dtype=torch.float16,
)

while True:
    prompt = input("Enter a prompt: \n")
    if prompt == "quit":
        break
    print(f"Here's the LlaMa output: \n")
    print(pipe(prompt, max_length=5000, num_return_sequences=1)[0]['generated_text'])
    print("="*20)

"""
Memory Statistic
----------------
Model: Llama-2-7b-chat-hf
Device: cuda
_________________________________________________________________________
| Float Precision | Model / MiB | Max Response Length | Inference / MiB | 
| --------------- | ----------- | ------------------- | --------------- |
| 32              | OOM         |                     |                 |
| 16              | 13628       | 50                  | 14020           |
| 16              | 13628       | 732                 | 15396           |
| 16              | 13628       | 1000+               | 22424           |
_________________________________________________________________________
Bad prompt:
1. 如何使用文心一言  

Conclusions:
1. Inference Memory 主要取决于生成文本的长度
2. 多轮对话，Memory不会线性增加，但会持续占资源
"""