#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/23 22:13
# @Author : ZhangKuo
import pytest
import torch
import torch.nn as nn

from main import VisionTransformer


@pytest.fixture
def vision_transformer():
    return VisionTransformer()


class TestVisionTransformer:
    def test_forward(self, vision_transformer):
        x = torch.randn(2, 3, 224, 224)
        out = vision_transformer(x)
        assert out.shape == (2, 1000)
        assert isinstance(out, torch.Tensor)

    def test_summary(self, vision_transformer):
        summary = nn.Sequential(vision_transformer, nn.Softmax(dim=-1))
        x = torch.randn(2, 3, 224, 224)
        out = summary(x)
        assert out.shape == (2, 1000)
        assert isinstance(out, torch.Tensor)
