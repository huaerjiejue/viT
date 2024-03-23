#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 18:29
# @Author : ZhangKuo

import pytest
import torch
from patch_embedding import PatchEmbedding


@pytest.fixture
def patch_embedding():
    return PatchEmbedding()


class TestPatchEmbedding:
    def test_forward(self, patch_embedding):
        x = torch.randn(2, 3, 224, 224)
        out = patch_embedding(x)
        assert out.shape == (2, 197, 768)
