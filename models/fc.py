from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import operator
from functools import reduce
import numpy as np
import torch.nn.functional as F

from utils.builder import get_builder
from args import args

class tabular(nn.Module):
    def __init__(self, input_shape, output_shape, depth, width, task):
        super(tabular, self).__init__()
        self.builder = get_builder()
        self.arch = self.make_architecture(input_shape, output_shape, depth, width)
        self.net = self.make_layers()
   
    def make_architecture(self, input_shape, output_shape, depth, width):
       arch = np.ones(depth, dtype=int)*width
       arch[0] = input_shape
       arch[-1] = output_shape
       return arch
   
    def make_layers(self):
        layerStack = []
        dd = len(self.arch)
        for i in np.arange(dd-2):
            l = self.builder.conv1x1(self.arch[i], self.arch[i+1])
            layerStack += [l, nn.ReLU()]
        i =  dd-2
        l = self.builder.conv1x1(self.arch[i], self.arch[i+1])
        layerStack += [l]
        
        return nn.Sequential(*layerStack)
    
    def forward(self, x):
        #y = F.linear(self.net(x))
        #print(y.shape)
        return self.net(x[:, :, None, None]).squeeze() #F.linear(self.net(x))

class MLP(nn.Module):
    def __init__(self, width):
        super(MLP, self).__init__()
        self.builder = get_builder()
        self.linear1 = self.builder.conv1x1(784,width)
        self.linear2 = self.builder.conv1x1(width,width)
        self.linear3 = self.builder.conv1x1(width,10)
    
    def forward(self,X):
        X = X.view(X.shape[0], -1)[:, :, None, None]
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return X.squeeze()
 
class MLP_linear(nn.Module):
    def __init__(self, width):
        super(MLP_linear, self).__init__()
        self.builder = get_builder()
        
        self.linear1 = self.builder.conv1x1(784,width)
        self.bn1 = self.builder.batchnorm(width)
        self.linear2 = self.builder.conv1x1(width,width)
        self.bn2 = self.builder.batchnorm(width)
        self.linear3 = self.builder.conv1x1(width,10)
    
    def forward(self,X):
        X = X.view(X.shape[0], -1)[:, :, None, None]
        X = self.bn1(self.linear1(X))
        X = self.bn2(self.linear2(X))
        X = self.linear3(X)
        return X.squeeze()
 