#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   baseprofiler.py
@Time   :   2025/06/04 14:36:43
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   refer: https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/use-pytorch-profiler-to-realize-performance-analysis-and-troubleshooting-of-large-models#9583c9f024giu
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
os.environ['PYTORCH_JIT'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn
import torch.optim
import torch.profiler as profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

import faulthandler
faulthandler.enable()


# 准备输入数据。本教程中，使用CIFAR10数据集，将其转换为所需格式，并使用DataLoader加载每个批次。
transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
savepath = "/media/data/datasets"
train_set = torchvision.datasets.CIFAR10(root=savepath, train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_set, num_workers=8, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_set, num_workers=8, pin_memory=True, batch_size=512, shuffle=True)

# 创建ResNet模型、损失函数和优化器对象。为了在GPU上运行，将模型和损失转移到GPU设备上。
device = torch.device("cuda")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


# 定义对每批输入数据的训练步骤。
def train(data):
    # inputs, labels = data[0].to(device=device), data[1].to(device=device)
    # 异步传输，在数据从内存到显存的过程中，cpu不挂起
    inputs, labels = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
    # 开启amp
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 使用分析器记录执行事件
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA
    ],
    schedule=profiler.schedule(
        wait=1, warmup=4, active=3, repeat=1
    ),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./resnet18/baseline'),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./resnet18/moreworkers'),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./resnet18/transfereffi_bs512'),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./resnet18/ampmixprecision_bs512'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= (1 + 4 + 3) * 1:
            break
        train(batch_data)
        prof.step()  # 需要在每个步骤上调用此函数以通知分析器步骤边界。