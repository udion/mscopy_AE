import argparse
import os
import sys
from models import *
from train_ae import *


parser = argparse.ArgumentParser()
parser.add_argument('--Weights', type=str, default='', help="path to weights (to continue training)")

opt = parser.parse_args()
print(opt)

if opt.Weights != '':
	A.load_state_dict(torch.load(opt.Weights))
A = A.cuda()
print(A)

train_ae(A, 'AE_VGG19_MSE.pth', 64)