import argparse
import os
import sys
from models import *
from train_ae import *


parser = argparse.ArgumentParser()
parser.add_argument('--Weights', type=str, default='', help="path to weights (to continue training)")
parser.add_argument('--modelName', type=str, default='myModel', help="Name of the model to be trained, logs will be created using this name")

opt = parser.parse_args()
print(opt)

if opt.Weights != '':
	A.load_state_dict(torch.load(opt.Weights))
A = A.cuda()
print(A)

train_ae(A, opt.modelName, 8)