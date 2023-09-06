from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
import numpy as np


class TrackActReLU(nn.Module):
    def __init__(self):
        super(TrackActReLU, self).__init__()
        self.collect_preact = True
        self.avg_preacts = None

    def forward(self, preact):
        if self.collect_preact:
            # Take the mean of the activation over the batch dimension
            self.avg_preacts = preact.mean(0).detach()

        act = F.relu(preact)
        return act


