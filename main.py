#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 22:00
# @Author : ZhangKuo
import torch
import torch.nn as nn
from einops import reduce
from torchinfo import summary

from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder


class Classification(nn.Module):
    def __init__(self, num_classes=1000, emb_size=768):
        super(Classification, self).__init__()
        self.layer = nn.Linear(emb_size, num_classes)
        self.norm = nn.LayerNorm(num_classes)

    def forward(self, x):
        x = reduce(x, "b n e -> b e", "mean")
        out = self.layer(x)
        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self,
        emb_size=768,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        num_layers=12,
        num_heads=12,
        d_ffn=3072,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, emb_size)
        self.transformer_encoder = TransformerEncoder(
            num_layers, emb_size, num_heads, d_ffn, dropout
        )
        self.classification = Classification(num_classes, emb_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classification(x)
        return x


if __name__ == "__main__":
    model = VisionTransformer()
    summary(model, input_size=(2, 3, 224, 224))
