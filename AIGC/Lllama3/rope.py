#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   rope.py
@Time   :   2024/12/10 10:05:17
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习下旋转位置编码
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import torch
import math


def cal_inv_freq(base, dim):
    """计算词向量元素两两分组之后，每组元素对应的旋转角度

    inv_freq: 从1开始递减的正小数，代表频率
    dim维度(index)越低（例如0），inv_freq频率越大，周期短，捕捉短距离局部信息
    dim维度(index)越高（例如128），inv_freq频率越小，周期长，捕捉长距离整体信息
    """
    # 知乎
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # hf
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int16).float().to(device) / dim))
    return inv_freq


def smooth_inv_freq(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192):
    """平滑，可能是为了兼容之前的版本
    """
    inv_freq = cal_inv_freq(500000.0, 4096 // 32)
    old_context_len = original_max_position_embeddings
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    
    # 先简单理解为周期长度吧，本质是什么不知道
    # 类比三角函数周期：T=2π/ω
    # 波长，波长反映正弦波的周期性
    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    
    return inv_freq_llama


def forward(x, inv_freq):
    """计算旋转位置编码的结果
    """
    position_ids = torch.tensor([
        [1] * 141 + list(range(114)),
        [1] * 169 + list(range(86))
    ])
    # 添加batch_size，seq_len维度
    # (batch_size, dim // 2, 1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    # (batch_size, 1, seq_len)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # shape = (batch_size, seq_len, dim // 2)
        # 理解：
        #      1. 多维矩阵积，shape: (x, 1) @ (1, y)  -->  广播积
        #      2. transpose结果(seq_len, dim // 2)，每一行是position_ids[i]乘上所有的inv_freq
        # 例子：
        # a = tensor([[[2],
        #             [4],
        #             [3]]])
        # b = tensor([[[2, 4, 4, 1, 3]]])
        # a @ b = tensor([[[ 4,  8,  8,  2,  6],
        #                  [ 8, 16, 16,  4, 12],
        #                  [ 6, 12, 12,  3,  9]]])
        # transpose:tensor([[[ 4,  8,  6],
        #                    [ 8, 16, 12],
        #                    [ 8, 16, 12],
        #                    [ 2,  4,  3],
        #                    [ 6, 12,  9]]])
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # shape = (batch_size, seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        attention_scaling = 1.0
        cos = cos * attention_scaling
        sin = sin * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(query_states, key_states, position_embeddings, unsqueeze_dim=1):
    """在实际forward中应用旋转位置编码embedding
        f(xm, m) = Wq xm e^{imθ}
        将e^{imθ}用旋转子形式给出

    query_states: 在第一层，只有词嵌入参与的Wqxm
    key_states: 在第一层，只有词嵌入参与的Wkxn
    position_embeddings: 在嵌入层之后就算完固定了，只是每个layer应用一次
    注意：
        1、在第二层就是融合位置之后的结果
        2、shape不一样的
    """
    cos, sin = position_embeddings
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # cos的index就是dim中的[0, 2, 4, ...] + [0, 2, 4, ...]
    # sin的index也是[0, 2, 4, ...] +  [0, 2, 4, ...]，只是.sin()
    # 原始论文矩阵形式：https://cloud.aigonna.com/2024/07/31/844/
    # hf为啥是对半编组：https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/4
    q_embed = (query_states * cos) + (rotate_half(query_states) * sin)
    k_embed = (key_states * cos) + (rotate_half(key_states) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    device = "cpu"
    inv_freq = cal_inv_freq(500000.0, 4096 // 32)