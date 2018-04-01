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

class VGG19_extractor(nn.Module):
    def __init__(self, cnn):
        super(VGG19_extractor, self).__init__()
        self.features1 = nn.Sequential(*list(cnn.features.children())[:3])
        self.features2 = nn.Sequential(*list(cnn.features.children())[:5])
        self.features3 = nn.Sequential(*list(cnn.features.children())[:12])
    def forward(self, x):
        return self.features1(x), self.features2(x), self.features3(x)

class Encoder(nn.Module):
    def __init__(self, n_res_blocks=5):
        super(Encoder, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_1' + str(i+1), resBlock(in_channels=64, out_channels=64, k=3, s=1, p=1))
        self.conv2 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_2' + str(i+1), resBlock(in_channels=32, out_channels=32, k=3, s=1, p=1))
        self.conv3 = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_3' + str(i+1), resBlock(in_channels=8, out_channels=8, k=3, s=1, p=1))
        self.conv4 = nn.Conv2d(8, 1, 3, stride=1, padding=1)
    
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
    def __init__(self, n_res_blocks=5):
        super(Decoder, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.conv1 = nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_1' + str(i+1), resTransposeBlock(in_channels=8, out_channels=8, k=3, s=1, p=1))
        self.conv2 = nn.ConvTranspose2d(8, 32, 3, stride=1, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_2' + str(i+1), resTransposeBlock(in_channels=32, out_channels=32, k=3, s=1, p=1))
        self.conv3 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        for i in range(n_res_blocks):
            self.add_module('residual_block_3' + str(i+1), resTransposeBlock(in_channels=64, out_channels=64, k=3, s=1, p=1))
        self.conv4 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1)
    
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


class VAE(nn.Module):
    def __init__(self, encoder, decoder, batchsz):
        super(VAE, self).__init__()
        self.E = encoder
        self.D = decoder
        self.batchsz = batchsz
        self._enc_mu = nn.Linear(26*26, 128)
        self._enc_log_sigma = nn.Linear(26*26, 128)
        self._din_layer = nn.Linear(128, 26*26)
        
    def _sample_latent(self, h_enc):
        '''
        Return the latent normal sample z ~ N(mu, sigma^2)
        '''
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick

    def forward(self, x):
        h_enc = self.E(x)
        h_enc = h_enc.view(self.batchsz, 1, -1)
        z = self._sample_latent(h_enc)
        z = self._din_layer(z)
        z = z.view(self.batchsz, 1, 26, 26)
        return self.D(z)

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.E = encoder
        self.D = decoder
    def forward(self, x):
        h_enc = self.E(x)
        return self.D(h_enc)