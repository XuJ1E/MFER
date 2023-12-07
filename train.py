import argparse

import timm
import torch
from torch import nn
from torchvision import transforms, datasets
from models.loss_function import LabelSmoothingCrossEntropy, SmoothFunction
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from MFER import convnext_base


parser = argparse.ArgumentParser(description='FER')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--weight_loss', type=float, default=0.001, help='loss weight of feature loss')
parser.add_argument('--bs', type=int, default=32, help='bs')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--epochs', type=int, default=100, help='number of train epoch')
parser.add_argument('--workers', type=int, default=8, help='Number of cpu data loading works')
args = parser.parse_args()


def main():
  
