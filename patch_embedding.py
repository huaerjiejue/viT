#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 18:11
# @Author : ZhangKuo
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """

        :param img_size:
        :param patch_size:
        :param in_chans:
        :param embed_dim: patch_size * patch_size * in_chans
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange(
                "b e (h) (w) -> b (h w) e",
            ),
            nn.LayerNorm(embed_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.num_patches = (img_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(
            torch.randn(self.num_patches + 1, embed_dim)
        )

    def forward(self, x):
        x = self.projection(x)
        b, n, _ = x.shape
        # 将cls_token变成batch_size个
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        # 将cls_token添加到x中，使用dim=1进行拼接，增加要投影的batchs前面
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positional_embedding
        return x
