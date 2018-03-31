import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

class resBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, k=3, s=1, p=1):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, k, stride=s, padding=p)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x
    
class resTransposeBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, k=3, s=1, p=1):
        super(resTransposeBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, k, stride=s, padding=p)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class Encoder(nn.Module):
    def __init__(self, k=3, s=1, p=1, n_res_blocks=16):
        super(Encoder, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.conv1 = nn.Conv2d(3, 64, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_1' + str(i+1), resBlock(in_channels=64, out_channels=64, k=k, s=s, p=p))
        self.conv2 = nn.Conv2d(64, 32, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_2' + str(i+1), resBlock(in_channels=32, out_channels=32, k=k, s=s, p=p))
        self.conv3 = nn.Conv2d(32, 8, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_3' + str(i+1), resBlock(in_channels=8, out_channels=8, k=k, s=s, p=p))
        self.conv4 = nn.Conv2d(8, 1, k, stride=s, padding=p)
    def forward(self, x):
        y = self.conv1(x)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_1'+str(i+1))(y)
        y = self.conv2(y)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_2'+str(i+1))(y)
        y = self.conv3(y)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_3'+str(i+1))(y)
        y = self.conv4(y)
        return y

class Decoder(nn.Module):
    def __init__(self, k=3, s=1, p=1, n_res_blocks=16):
        super(Decoder, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.conv1 = nn.Conv2d(1, 8, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_1' + str(i+1), resBlock(in_channels=8, out_channels=8, k=k, s=s, p=p))
        self.conv2 = nn.Conv2d(8, 32, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_2' + str(i+1), resBlock(in_channels=32, out_channels=32, k=k, s=s, p=p))
        self.conv3 = nn.Conv2d(32, 64, k, stride=s, padding=p)
        for i in range(n_res_blocks):
            self.add_module('residual_block_3' + str(i+1), resBlock(in_channels=64, out_channels=64, k=k, s=s, p=p))
        self.conv4 = nn.Conv2d(64, 3, k, stride=s, padding=p) 
    def forward(self, x):
        y = self.conv1(x)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_1'+str(i+1))(y)
        y = self.conv2(y)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_2'+str(i+1))(y)
        y = self.conv3(y)
        for i in range(self.n_res_blocks):
            y = self.__getattr__('residual_block_3'+str(i+1))(y)
        y = self.conv4(y)
        return y