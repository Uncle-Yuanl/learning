#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   grad_flow.py
@Time   :   2023/09/26 09:36:55
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习计算图的梯度流向与聚合
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import torch


def no_detach():
    x = torch.tensor(([1.0]),requires_grad=True)
    y = x**2
    z = 2*y
    w= z**3

    # This is the subpath
    # Do not use detach()
    p = z
    p.retain_grad()
    q = torch.tensor(([2.0]), requires_grad=True)
    pq = p*q

    # register hook
    p.register_hook(lambda grad: print("node p: ", grad))
    x.register_hook(lambda grad: print("node x: ", grad))

    pq.backward(retain_graph=True)
    w.backward()
    print("tensor p: ", p)
    print("tensor w: ", w)
    print("tensor x: ", x)
    print("node p: ", p.grad)
    print("node x: ", x.grad)


def with_detach():
    x = torch.tensor(([1.0]),requires_grad=True)
    y = x**2
    z = 2*y
    w= z**3

    # detach it, so the gradient w.r.t `p` does not effect `z`!
    p = z.detach()
    # p.retain_grad()  # now we can not retrain_grad() on `p` because it's requires_grad=False
    q = torch.tensor(([2.0]), requires_grad=True)
    pq = p*q

    # register hook
    # p.register_hook(lambda grad: print("node p: ", grad))  # can not hook too
    x.register_hook(lambda grad: print("node x: ", grad))

    pq.backward(retain_graph=True)
    w.backward()
    print("tensor p: ", p)
    print("tensor w: ", w)
    print("tensor x: ", x)
    print("node p: ", p.grad)
    print("node x: ", x.grad)


if __name__ == '__main__':
    # no_detach()
    with_detach()