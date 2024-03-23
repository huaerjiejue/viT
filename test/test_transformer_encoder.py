#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 21:53
# @Author : ZhangKuo
import pytest
import torch

from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder


@pytest.fixture
def transformer_encoder():
    return TransformerEncoder()


@pytest.fixture
def patch_embedding():
    return PatchEmbedding()


class TestTransformerEncoder:
    def test_forward(self, transformer_encoder):
        x = torch.randn(2, 197, 768)
        out = transformer_encoder(x)
        assert out.shape == (2, 197, 768)

    def test_connect_with_patch_embedding(self, transformer_encoder, patch_embedding):
        x = torch.randn(2, 3, 224, 224)
        x = patch_embedding(x)
        out = transformer_encoder(x)
        assert out.shape == x.shape == (2, 197, 768)
