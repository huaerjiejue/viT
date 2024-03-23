#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 21:43
# @Author : ZhangKuo
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int = 12, d_model: int = 768, num_heads: int = 12, d_ffn: int = 3072, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ffn, dropout, activation='gelu')
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
