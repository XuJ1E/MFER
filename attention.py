# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/7/6 17:24
# Author     ：XuJ1E
# version    ：python 3.8
# File       : attention.py
"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.key = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, dim)
        )
        self.value = nn.Linear(dim, dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x.clone()
        prem = x.permute(0, 2, 3, 1)
        value = self.value(prem).reshape(b, -1, c)
        key = self.key(prem) * prem
        key = key.reshape(b, -1, c)
        attn = self.attn(x).permute(0, 2, 3, 1)
        attn = self.norm(self.act(attn)).reshape(b, -1, c)
        attn = self.softmax(attn + key) * value
        attn = attn.reshape(b, h, w, c)
        attn = attn.permute(0, 3, 1, 2)
        return attn + shortcut


if __name__ == '__main__':
    x = torch.rand((1, 64, 32, 32))
    attention = Attention(dim=64)
    print(attention(x).shape)