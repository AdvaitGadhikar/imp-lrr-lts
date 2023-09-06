import os
import pathlib
import random
import shutil
import time
import json
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy, cosine_lr, assign_learning_rate, warmup_lr, constant_lr, multistep_lr
from utils.conv_type import STRConv, STRConvER, ConvER, ConvMask
from utils.conv_type import sparseFunction
from utils.compensation import CompensatePrune

from args import args
from trainer import train, validate, get_preds, hessian_trace, validate_loss

import data
from data import cifar10, imagenet

import models
from models import resnet18
from models import resnet
from models import fc



def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master/'

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.set == 'cifar100':
        model1 = resnet18.ResNet18([3, 32, 32], num_classes=100)
        model2 = resnet18.ResNet18([3, 32, 32], num_classes=100)

    if args.set == 'cifar10':
        model1 = resnet18.ResNet18([3, 32, 32], num_classes=10)
        model2 = resnet18.ResNet18([3, 32, 32], num_classes=10)

    if args.set == 'tiny-imagenet':
        model1 = resnet.ResNet50(num_classes=200)
        model2 = resnet.ResNet50(num_classes=200)
    
    model1 = torch.nn.DataParallel(model1, device_ids=args.multigpu).cuda(args.multigpu[0])
    model2 = torch.nn.DataParallel(model2, device_ids=args.multigpu).cuda(args.multigpu[0])
    
    print('after set gpu')
    model1.load_state_dict(torch.load('{}runs/{}'.format(base_dir, args.linear_mode_model_1)))
    model2.load_state_dict(torch.load('{}runs/{}'.format(base_dir, args.linear_mode_model_2)))

    # Load the mask and apply the mask before averaging
    if args.linear_mode_mask_1 != '':
        mask_list_1 = torch.load('{}runs/{}'.format(base_dir, args.linear_mode_mask_1))
        cnt = 0
        for n, m in model1.named_modules():
            if isinstance(m, ConvMask):
                m.mask = mask_list_1[cnt].to(m.weight.device)
                m.weight.data = m.weight.data * m.mask
                cnt += 1

    if args.linear_mode_mask_2 != '':
        mask_list_2 = torch.load('{}runs/{}'.format(base_dir, args.linear_mode_mask_2))
        cnt = 0
        for n, m in model2.named_modules():
            if isinstance(m, ConvMask):
                m.mask = mask_list_2[cnt].to(m.weight.device)
                m.weight.data = m.weight.data * m.mask
                cnt += 1

    print('loaded the mask and model')
    
    print('The model definition is:')
    
    num_stats = []
    for (n1, m1), (n2, m2) in zip(model1.named_modules(), model2.named_modules()):
        if isinstance(m1, ConvMask):
            num = m1.weight.data.numel()
            sparsity = m1.mask.sum() / num
            s1 = (m1.weight.data * m1.mask).sign()
            s2 = (m2.weight.data * m2.mask).sign() 
            # get only the signs that are same within the mask, and not the zeroed out weights
            sign_sim = (s1 == s2).sum() - (m1.mask == 0).sum()
            num_stats.append((num, sparsity, sign_sim))

    print('saving the experiments as: ', args.name)
    torch.save(num_stats, '{}runs/sign_sim_{}_{}.pt'.format(base_dir, args.linear_mode_model_1, args.linear_mode_model_2))    

def set_gpu(args, model):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
    cudnn.benchmark = True

    return model

if __name__ == "__main__":
    main()
