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

mytransform1 = transforms.Compose(
    [transforms.RandomCrop((101,101)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# functions to show an image
def imshow(img):
    #img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def imshow2(img):
    m1 = torch.min(img)
    m2 = torch.max(img)
    img = (img-m1)/(m2-m1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def save_model(model, model_name):
    try:
        os.makedirs('../saved_models')
    except OSError:
        pass
    torch.save(model.state_dict(), '../saved_models/'+model_name)
    print('model saved at '+'../saved_models/'+model_name)