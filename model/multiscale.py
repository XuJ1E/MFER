# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/7/6 17:47
# Author     ：XuJ1E
# version    ：python 3.8
# File       : multi-scale.py
"""
import torch
import torch.nn as nn


class classifier(nn.Module):
    def __init__(self, dim=256, num_class=7, step=1):
        super().__init__()
        self.fc1 = nn.Linear(dim*step, dim*step)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim*step, num_class)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Msc(nn.Module):
    def __init__(self):
        super().__init__()
        self.msc = nn.ModuleList([classifier(step=2), classifier(step=3), classifier(step=4)])

    def forward(self, x):
        output = []
        for i in range(3):
            pred = self.msc[i](x[:, 0:256*i + 512])
            output.append(pred)
        return output


if __name__ == '__main__':
    x = torch.rand((1, 1024))
    model = Msc()
    for i in range(3):
        print(model(x)[i].shape)
