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
from utils.schedulers import get_policy, cosine_lr, assign_learning_rate, warmup_lr, constant_lr, multistep_lr, multistep_lr_drops, imagenet_lr_drops, cosine_lr_warmup, multistep_lr_warmup, imagenet_lr_drops_warmup
from utils.conv_type import STRConv, STRConvER, ConvER, ConvMask
from utils.conv_type import sparseFunction
from utils.compensation import CompensatePrune
from utils.custom_activation import TrackActReLU

from args import args
from trainer import train, validate, get_preds, hessian_trace, train_kd, train_with_fixed_signs

import data
from data import cifar10, imagenet

import models
from models import resnet18
from models import resnet18c
from models import resnet
from models import fc
from models import resnet20
# from models import resnet_ima


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

    ### Modifications for speeding up training    
    torch.backends.cuda.matmul.allow_tf32 = True

    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    # model = get_model(args)
    if args.set == 'cifar10':
        if args.resnet_type == 'small-dense':
            model = resnet18.ResNetWidth18([3, 32, 32], num_classes=10, width = args.width)
        elif args.resnet_type == 'small-dense-inc':
            model = resnet18.ResNetWidth18Inc([3, 32, 32], num_classes=10, width = args.width)
        elif args.resnet_type == 'small-sub':
            model = resnet18.ResNetWidthSub18([3, 32, 32], num_classes=10, width = args.width)
        elif args.resnet_type == 'large-c':
            model = resnet18c.ResNet18C([3, 32, 32], num_classes=10)
        elif args.resnet_type == 'res20':
            model = resnet20.ResNet20([3, 32, 32], num_classes=10)
            print(model)
        else:
            model = resnet18.ResNet18([3, 32, 32], num_classes=10)
            if args.kd:
                teacher = resnet18.ResNet18([3, 32, 32], num_classes=10)

    if args.set == 'cifar100':
        if args.resnet_type == 'small-dense':
            model = resnet18.ResNetWidth18([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'small-dense-inc':
            model = resnet18.ResNetWidth18Inc([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'small-sub':
            model = resnet18.ResNetWidthSub18([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'res50':
            model = resnet18.ResNet50([3, 32, 32], num_classes=100)
        elif args.resnet_type == 'large-c':
            model = resnet18c.ResNet18C([3, 32, 32], num_classes=100)
        else:
            model = resnet18.ResNet18([3, 32, 32], num_classes=100)
            if args.kd:
                teacher = resnet18.ResNet18([3, 32, 32], num_classes=100)

    if args.set == 'tiny-imagenet':
        if args.resnet_type == 'res18':
            model = resnet.ResNet18(num_classes=200)
        else:
            model = resnet.ResNet50(num_classes=200)
    if args.set == 'imagenet':
        model = resnet.ResNet50()
    if args.set == 'heart':
        model = fc.tabular(input_shape=6, output_shape=2, depth=6, width=512, task='classification')
    if args.set == 'mnist':
        hidden = 256
        if args.resnet_type == 'linear':
            model = fc.MLP_linear(hidden)
        else:
            model = fc.MLP(hidden)

    #Setting the base directory for running experiments
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master/'
        
    if args.er_sparse_method == 'uniform':
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(args.er_sparse_init)

        print(args.er_sparse_init)

    if args.er_sparse_method == 'ERK':
        sparsity_list = []
        num_params_list = []
        total_params = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
                num_params_list.append(m.weight.numel())
                total_params += m.weight.numel()
        
        num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
        num_params_to_keep = total_params * args.er_sparse_init
        C = num_params_to_keep / num_params_kept
        sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]
        print(sparsity_list)
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)
        

    if args.er_sparse_method == 'balanced':
        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = args.er_sparse_init * total_params / l

        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)

        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)

    if args.er_sparse_method == 'pyramidal':

        num_params = 0
        layerwise_params = []
        for name, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                num_params += m.weight.numel()
                layerwise_params.append(m.weight.numel())
        layerwise_params = layerwise_params[::-1]
        layerwise_params.append(-num_params*args.er_sparse_init)
        roots = np.roots(np.array(layerwise_params))
        print('Roots of np solver', roots)
        for r in roots:
            if r < 1 and r > 0 and np.imag(r) == 0:
                print('root identified: ', r)
                layerwise_sparsities = np.arange(1, len(layerwise_params) + 1)
                sparsity_list = np.real(r) ** layerwise_sparsities
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        
        print(layerwise_sparsities)
    
    if args.er_sparse_method == 'str-induced-er':
        with open(base_dir + 'runs/layerwise_sparsity/' + args.er_sparsity_file) as f:
           data = json.load(f)
        sparsity_list = list(data.values())
        alpha = sparsity_list[-1] / args.er_sparse_init
        sparsity_list = [s / alpha for s in sparsity_list[:-1]]
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)
        


    
    print('student before')
    print(model.state_dict().keys())
    model = set_gpu(args, model)
    print('student after')
    print(model.state_dict().keys())
    if args.kd:
        print('Teacher before')
        args.gpu = None
        print(teacher.state_dict().keys())
        teacher = set_gpu(args, teacher)
        print('Teacher after ')
        print(teacher.state_dict().keys())

    print('The model definition is:')
    print(model)
    total_num_narrow = 0
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_num_narrow += m.weight.numel()
    print('num params', total_num_narrow)
    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)

    
    resnet18_params = 11164352

    optimizer = get_optimizer(args, model)
    # data = get_dataset(args)
    if args.set == 'cifar10':  
        data = cifar10.CIFAR10(args)
    if args.set == 'cifar100':
        data = cifar10.CIFAR100(args)
    if args.set == 'tiny-imagenet':
        data = cifar10.TinyImagenet(args)        
        # data = imagenet.FFCVTinyImageNet(args)

    if args.set == 'imagenet':
        # data = imagenet.ImageNet(args)
        # FFCVImageNet
        data = imagenet.FFCVImageNet(args)

    if args.set == 'heart':
        data = cifar10.HeartData(args)
    if args.set == 'mnist':
        data = cifar10.MNIST(args)


    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # if args.resume:
    #     best_acc1 = resume(args, model, optimizer)

    # Evaulation of a model
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )


    # prune method
    def prune_flow(model, thresh):
        total_num = 0
        total_den = 0
        
        for name, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                print('Pruning for every layer of ConvMask')
                score = (m.mask.to(m.weight.device) * m.weight).detach().abs_().view(m.weight.shape[0], -1)
                max_weights, _ = score.max(dim=1)
                score = score / max_weights[:, None]
    
                new_mask = (score >= thresh)
                m.mask = new_mask.view(m.weight.shape)
                print('New Density: ', (new_mask.sum() / new_mask.numel()).item())
                print('Mask Density of Layer: ', ((m.mask == 1).sum() / m.mask.to(m.weight.device).numel()).item())
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
        print('Overall model density = ', total_num / total_den)
        return model
    

    # random prune iterative
    def prune_random(model, density):
        
        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = density * total_params / l

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                curr_nz = (m.mask == 1).sum()
                target_nz = int(sparsity_list[cnt] * m.mask.numel())
                print(curr_nz, target_nz, m.mask.numel())
                diff = target_nz / (curr_nz - target_nz) 
                if diff > 0 and diff < 1:
                    m.mask = torch.where(m.mask == 0, m.mask, torch.empty(m.mask.shape).bernoulli_(diff))
                cnt+=1
        return model
    

    def prune_mag(model, density):
        score_list = {}
        for n, m in model.named_modules():
            # torch.cat([torch.flatten(v) for v in self.scores.values()])
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after magnitude pruning at current iteration = ', total_num / total_den)
        return model

    def prune_random_balanced(model, density):

        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = density * total_params / l
        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (balanced) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_uniform(model, density):

        total_num = 0
        total_den = 0
        
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()
                global_scores = torch.flatten(score)
                k = int((1 - density) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score.to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                

        print('Overall model density after random global (uniform) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_erk(model, density):

        sparsity_list = []
        num_params_list = []
        total_params = 0
        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
                num_params_list.append(m.weight.numel())
                total_params += m.weight.numel()
        
        num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
        num_params_to_keep = total_params * density
        C = num_params_to_keep / num_params_kept
        print('Factor: ', C)
        sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (ERK) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_global(model, density):

        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        total_num = 0
        total_den = 0
        cnt = 0
        if not k < 1:
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()
                    cnt += 1

        print('Overall model density after random global pruning at current iteration = ', total_num / total_den)
        return model

    def prune_snip(model, trainloader, loss, density):

        for i, (images, target) in enumerate(trainloader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True).long()
            model.zero_grad()
            output = model(images)
            loss(output, target).backward()
            break
        
        score_list = {}
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()
        
        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after snip pruning at current iteration = ', total_num / total_den)
        return model


    def prune_synflow(model, trainloader, density):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(trainloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(args.gpu)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        score_list = {}
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * m.weight.grad * m.weight).detach().abs_()
        
        model.zero_grad()

        nonlinearize(model, signs)

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after synflow pruning at current iteration = ', total_num / total_den)
        return model

    def perturb_signs(model, perturb_ratio):
        # This function randomly flips the signs of perturb_ratio fraction of weights uniformly in each layer
        for n, m in model.named_modules():
            if isinstance(m, ConvMask):
                sign = torch.where(m.mask == 1, torch.where(torch.ones_like(m.mask).bernoulli_(perturb_ratio) == 1, -1, 1), 0).to(m.mask.device)
                m.weight.data = sign * m.weight.data
        return model

    def dead_neurons(model):
        dead_list = []
        dead_mask_list_in = []
        dead_mask_list_out = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                idxs = (m.mask.sum(dim=(1,2,3)) == 0).nonzero(as_tuple=True)
                dead_mask_in = torch.where(m.mask.sum(dim=(1,2,3)) == 0, 1, 0)
                dead_mask_list_in.append(dead_mask_in)

                dead_mask_out = torch.where(m.mask.sum(dim=(0,2,3)) == 0, 1, 0)
                dead_mask_list_out.append(dead_mask_out)

                num_dead = idxs[0].numel()
                dead_ratio = num_dead/m.mask.shape[0]
                dead_list.append((num_dead, dead_ratio))
            
        return dead_list, dead_mask_list_in, dead_mask_list_out

    def prune_dead(model):
        dead_idx_in = []
        dead_idx_out = []
        num_in = 0
        num_out = 0
        print('if there are any dead neurons in the network, all the incoming and outgoing weights to it are zeroed out')
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                in_idxs = (m.mask.sum(dim=(1,2,3)) == 0).nonzero(as_tuple=True)
                dead_idx_in.append(in_idxs)
                num_in += in_idxs[0].shape[0]
                out_idxs = (m.mask.sum(dim=(0,2,3)) == 0).nonzero(as_tuple=True)
                dead_idx_out.append(out_idxs)
                num_out += out_idxs[0].shape[0]

        print('Num in and num out: ', num_in, num_out)
        # Removing dead incoming neurons
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if l == 0:
                    idx = dead_idx_in[l]
                    m.mask[idx, :, :, :] = 0
                elif l == len(dead_idx_in)-1:
                    pass
                else:
                    idx = dead_idx_in[l]
                    m.mask[idx, :, :, :] = 0
                    idx_out = dead_idx_in[l-1]
                    m.mask[:, idx, :, :] = 0

                l += 1

        # Removing dead outgoing neurons
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if l == 0:
                    idx = dead_idx_out[l]
                    m.mask[:, idx, :, :] = 0
                elif l == len(dead_idx_out)-1:
                    pass
                else:
                    idx = dead_idx_out[l]
                    m.mask[:,idx, :, :] = 0
                    idx_out = dead_idx_out[l-1]
                    m.mask[idx, :, :, :] = 0

                l += 1

        return model

    def prune_neuron_random_balanced(model):
        print('Random Structured Pruning with balanced sparsity initiated')
        num_neurons = []
        idxs = []
        prune_ratio = args.structured_prune_ratio
        # This prune ratio will prune away a percent of neurons in each layer at every prune step
        print('identifying random neurons')
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print(n, m.weight.shape, int(m.weight.shape[0] * prune_ratio))
                num = int(m.weight.shape[0] * prune_ratio)
                num_neurons.append(m.weight.shape[0])
                idx = torch.randperm(m.weight.shape[0])                            
                print('Index layer: ', idx[:num])
                idxs.append(idx[:num])
        
        print('pruning these neurons to zero')
        l = 0
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print('layer name: ', n)
                if l == 0:
                    # does not prune the first and last layer and maintains their width if flag is set to True
                    if not args.fix_first_last_structured:
                        idx = idxs[l]
                        m.mask[idx, :, :, :] = 0

                elif l == len(idxs)-1:
                    if not args.fix_first_last_structured:
                        idx_out = idxs[l-1]
                        m.mask[:, idx_out, :, :] = 0
                
                else:
                    idx = idxs[l]
                    m.mask[idx, :, :, :] = 0
                    idx_out = idxs[l-1]
                    m.mask[:, idx_out, :, :] = 0

                l += 1
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

        print('Pruned neurons in each layer as per their size and now the post pruning density is: ', total_num / total_den)
        return model

    def prune_neuron_random(model):
        num_neurons = []
        idxs = []
        num = 1
        print('identifying random neurons')
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print(n, m.weight.shape)
                num_neurons.append(m.weight.shape[0])
                idx = torch.randperm(m.weight.shape[0])                
                # while m.mask.sum(dim=(1,2,3))[idx[:num]].any() != 0:
                #     idx = torch.randperm(m.weight.shape[0])                
                print('Index layer: ', idx[:num])
                idxs.append(idx[:num])
        
        print('pruning these neurons to zero')
        l = 0
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print('layer name: ', n)
                if l == 0:
                    # does not prune the first and last layer and maintains their width if flag is set to True
                    if not args.fix_first_last_structured:
                        idx = idxs[l]
                        m.mask[idx, :, :, :] = 0

                elif l == len(idxs)-1:
                    if not args.fix_first_last_structured:
                        idx_out = idxs[l-1]
                        m.mask[:, idx_out, :, :] = 0
                
                else:
                    idx = idxs[l]
                    m.mask[idx, :, :, :] = 0
                    idx_out = idxs[l-1]
                    m.mask[:, idx_out, :, :] = 0

                l += 1
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

        print('Pruned one neuron in each layer and now the post pruning density is: ', total_num / total_den)
        return model

    def prune_neuron_sequential(model):
        # define number of neurons in a class and then prune 
        # within those neurons in each layer unstructured till they are completely removed
        num_neurons = []
        prune_ratio = args.structured_prune_ratio
        # remove edges from a fixed set of neurons in every layer
        idxs = []
        num = 1
        print('identifying random neurons')
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print(n, m.weight.shape)
                idx = torch.randperm(m.weight.shape[0])
                idxs.append(idx[:int(prune_ratio * m.weight.shape[0])])

                # while m.mask.sum(dim=(1,2,3))[idx[:num]].any() != 0:
                #     idx = torch.randperm(m.weight.shape[0])                
                print('Index layer: ', idx[:num])
                idxs.append(idx[:num])
        
        
        print('pruning these neurons to zero')
        l = 0
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)) and not ('downsample' in n):
                print('layer name: ', n)
                if l == 0:
                    # does not prune the first and last layer and maintains their width if flag is set to True
                    if not args.fix_first_last_structured:
                        idx = idxs[l]
                        m.mask[idx, :, :, :] = 0

                elif l == len(idxs)-1:
                    if not args.fix_first_last_structured:
                        idx_out = idxs[l-1]
                        m.mask[:, idx_out, :, :] = 0
                
                else:
                    idx = idxs[l]
                    m.mask[idx, :, :, :] = 0
                    idx_out = idxs[l-1]
                    m.mask[:, idx_out, :, :] = 0

                l += 1
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

        print('Pruned one neuron sequential in each layer and now the post pruning density is: ', total_num / total_den)
        return model
    


    def prune_mask_ref(model, level, name):
        # Loading a mask at every level
        # ref_mask = 'runs/mask_cifar-imp-rewind-save-every-0.1-seed-4_'
        base_dir = ''
        if args.expt_setup == 'cispa':
            base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
        if args.expt_setup == 'juwels':
            base_dir = '/p/project/hai_efficientml/STR-master/'
            
        ref_mask = base_dir + 'runs/mask_' + str(name) + '_'
        mask_list = torch.load(ref_mask + str(level) + '.pt')

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                
                m.mask = mask_list[cnt]
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1
    
        print('Overall model density after loading IMP mask = ', total_num / total_den)
        return model


    def integrate_bn_params(model):
        '''
        The aim is to incorporate the scaling of the BN parameter in the weights of the previous layer.
        Since we do not have a bias in the weight, we modify the bias of the BN layer to account for this integration.
        '''
        bn_params = []
        eps = 1e-5
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                gamma, beta, mu, var = m.weight.clone(), m.bias.clone(), m.running_mean, m.running_var
                bn_params.append([gamma, beta, mu, var])
                m.weight.data.fill_(1)
                m.bias.data = beta - (gamma * mu / torch.sqrt(var + eps))
                m.running_mean.zero_()
                m.running_var.fill_(1)
        
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, ConvMask) and (cnt < len(bn_params)):
                gamma, beta, mu, var = bn_params[cnt][0], bn_params[cnt][1], bn_params[cnt][2], bn_params[cnt][3] 
                m.weight.data = m.weight.data * gamma[:, None, None, None] / torch.sqrt(var[:, None, None, None] + eps)
                cnt += 1
        print('integrated the batch norm parameters in the Conv Weights, but kept the Bias of the BN')

        return model


    def prune_mag_integrated_bn(model, density, include_gamma=False):
        ### This function first integrates the BN parameters in the weights and then uses these corrected weights as the scores for magnitude prunining
        ### In effect, we are only scaling the weights by the factor gamma. It should be gamma / sd, but this introduces further problems. Hence we start with simply multiplying by gamma.
        
        bn_params = []
        eps = 1e-5
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                gamma, beta, mu, var = m.weight.clone(), m.bias.clone(), m.running_mean, m.running_var
                bn_params.append([gamma, beta, mu, var])
        
        score_list = {}
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if cnt < len(bn_params):
                    gamma = bn_params[cnt][0]
                    var = bn_params[cnt][3]
                    if include_gamma:
                        score_list[n] = (m.mask.to(m.weight.device) * m.weight * gamma[:, None, None , None ] / torch.sqrt(eps + var[:, None, None, None])).detach().abs_()
                    else:
                        score_list[n] = (m.mask.to(m.weight.device) * m.weight * gamma[:, None, None , None ]).detach().abs_()
                    cnt += 1
                else: 
                    score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()
                    cnt += 1

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after magnitude pruning with integrated BN gamma in scores at current iteration = ', total_num / total_den)
        return model

    ### Define a new pruner, which prunes based on the value of the BN parameter gamma, and removes filters (channels with the lowest )
    # def prune_bn_struct(model, density):
    #     score_list = {}
    #     cnt = 0
    #     for n, m in model.named_modules():
    #         if isinstance(m, (nn.BatchNorm2d)):
    #             score_list[n] = m.weight.detach().abs_()

                
    #     global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    #     k = int((1 - density) * global_scores.numel())
    #     threshold, _ = torch.kthvalue(global_scores, k)
    ######
    ## Set all the conv parameters to non trainable, and only train the BN parameters
    #####
    if args.train_only_bn:
        for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    m.weight.requires_grad = False
        print('Setting all the weights of the conv layers to not require gradient i.e. freezing them')
    
    if args.train_only_bn_and_linear:
        for n, m in model.named_modules():
                if isinstance(m, (ConvMask)) and ('linear' not in n):
                    m.weight.requires_grad = False
        print('Freezing all conv layers, only training BN and last linear layeer')
    # Iterative Prune Train before final train
    
    warmup_scheduler = warmup_lr(optimizer, args)
    val_acc_total = []

    ############## 
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master/'
        
    # warmup training
    if not args.warmup:
        args.warmup_epochs = 0
    curr_epoch = 0
    if args.warmup:
        print('Warm Up training for the model')
        for epoch in range(args.warmup_epochs):
            warmup_scheduler(epoch)
            lr = get_lr(optimizer)
            print('The curent learning rate is: ', lr)

            start_train = time.time()
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )
            train_time.update((time.time() - start_train) / 60)
            # evaluate on validation set
            start_validation = time.time()
            acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            
            val_acc_total.append(acc1)
            curr_epoch += 1

    # save the model and the optimizer
    torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
    torch.save(optimizer.state_dict(),"{}runs/optimizer_{}.pt".format(base_dir, args.name))
    # torch.save(val_acc_total, base_dir + 'runs/val_acc_'+ args.name + '.pt')

    hessian_list = []

    ####################
    # Load a model and mask from a given checkpoint
    if args.load_before_prune:
        print('Loading model from: ', args.load_model_name, args.load_mask_name)
        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)
        if args.fix_sign_and_train:
            init_sign_list = []
            for n, m in model.named_modules():
                # Only looking at conv weight signs for now, not at BN parameter signs
                if isinstance(m, ConvMask):
                    init_sign_list.append(m.weight.sign())
        
    
    # Loads only the model, mask will be all ones
    if args.load_only_model:
        print('Loading ONLY model from: ', args.load_model_name)
        base_dir = ''
        if args.expt_setup == 'cispa':
            base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
        if args.expt_setup == 'juwels':
            base_dir = '/p/project/hai_efficientml/STR-master/'

        model_name = base_dir + 'runs/' + args.load_model_name
        model.load_state_dict(torch.load(model_name))
        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
        ####################

    ### Loading only the mask of a reference model for any ablations
    if args.load_only_mask:
        print('Loading ONLY mask: ', args.load_mask_name)
        mask_name = base_dir + 'runs/' + args.load_mask_name
        mask_list = torch.load(mask_name)
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, ConvMask):
                m.mask = mask_list[cnt].to(m.weight.device)
                cnt += 1
        torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
        


    # Loads only the model, mask will be all ones and also loads the sign for the initialized model
    if args.load_model_mask_and_sign:
        print('Loading ONLY model and mask from: ', args.load_model_name, args.load_mask_name)
        base_dir = ''
        if args.expt_setup == 'cispa':
            base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
        if args.expt_setup == 'juwels':
            base_dir = '/p/project/hai_efficientml/STR-master/'

        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)

        print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
        sign_list = torch.load("{}runs/sign_list_{}.pt".format(base_dir, args.load_expt_sign))
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                cnt += 1

        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_sign_changed_init.pt".format(base_dir, args.name))

    ######
    if args.load_shuffled_model_mask_and_sign:
        print('Loading model and mask from, and shuffling the model weights: ', args.load_model_name, args.load_mask_name)
        base_dir = ''
        if args.expt_setup == 'cispa':
            base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
        if args.expt_setup == 'juwels':
            base_dir = '/p/project/hai_efficientml/STR-master/'

        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)

        print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
        sign_list = torch.load("{}runs/sign_list_{}.pt".format(base_dir, args.load_expt_sign))
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                curr_shape = m.weight.shape
                idx = torch.where(m.mask == 1)
                buf = m.weight.data.clone()
                shuffle = buf[idx]
                perm = torch.randperm(shuffle.numel())
                shuffle = shuffle[perm]
                buf[idx] = shuffle
                
                m.weight.data = buf.abs_() * sign_list[cnt].to(m.weight.device)
                cnt += 1

        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_model_shuffled_sign_changed_init.pt".format(base_dir, args.name))

    ##### Loading a teacher model for self knowledge distillation
    if args.kd:
        teacher = resume_from_checkpoint(teacher, args.teacher_model, args.teacher_mask)
        kd_train_loss = []

    if args.hessian:
        trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
        hessian_list.append(trace)
    
    if args.lbfgs_compensate:
        lbfgs_compensate = CompensatePrune(model)

    ### Level for pruning
    level = 0
    if args.resume_training_from_level:
            print('resuming model and mask from: ', args.load_model_name, args.load_mask_name)

            model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)
            val_acc_total = torch.load("{}runs/val_acc_{}.pt".format(base_dir, args.name))
            print('Loaded val acc list from previous checkpoint to continue training')
            level = args.resume_level
            print('Level is: ', level)

    torch.save(val_acc_total, base_dir + 'runs/val_acc_'+ args.name + '.pt')
    ###

    if args.conv_type == 'ConvMask':
         
        print('Threshold list: ', args.threshold_list)
        for threshold in args.threshold_list:
            assign_learning_rate(optimizer, args.lr)
            if args.prune_scheduler == 'cosine':
                prune_scheduler = cosine_lr(optimizer, args.final_prune_epoch, args)
            if args.prune_scheduler == 'step-warmup':
                prune_scheduler = multistep_lr_warmup(optimizer, args)
            if args.prune_scheduler == 'cosine-warmup':
                prune_scheduler = cosine_lr_warmup(optimizer, args.final_prune_epoch, args)
            if args.prune_scheduler == 'constant':
                prune_scheduler = constant_lr(optimizer, args)
                assign_learning_rate(optimizer, args.constant_prune_lr)
                print('Learning rate during pruning is constant at: ', args.constant_prune_lr)
            if args.prune_scheduler == 'step':
                prune_scheduler = multistep_lr(optimizer, args)
            if args.prune_scheduler == 'step-drops':
                prune_scheduler = multistep_lr_drops(optimizer, args)
                print('Step drops LR scheduler')
            if args.prune_scheduler == 'imagenet-step':
                prune_scheduler = imagenet_lr_drops(optimizer, args)
                print('ImageNet Step drops LR scheduler')
            if args.prune_scheduler == 'imagenet-step-warmup':
                prune_scheduler = imagenet_lr_drops_warmup(optimizer, args)
                print('ImageNet Step drops with warmup every cycle LR scheduler')

            for epoch in range(args.final_prune_epoch):
                prune_scheduler(epoch)
                lr = get_lr(optimizer)
                print('The curent learning rate is: ', lr)

                #####
                if args.perturb_signs:
                    if (epoch % args.perturb_sign_every) == args.perturb_sign_every-1:
                        print('Perturbing signs of model while training')
                        model = perturb_signs(model, args.perturb_sign_ratio)
                #####

                start_train = time.time()
                train_acc1, train_acc5 = train(
                    data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
                )
                train_time.update((time.time() - start_train) / 60)

                # evaluate on validation set
                start_validation = time.time()
                acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
                validation_time.update((time.time() - start_validation) / 60)
                val_acc_total.append(acc1)
                curr_epoch += 1
                if args.hessian:
                    trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
                    hessian_list.append(trace)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            ###
            if args.integrate_bn:
                print('Integrating the BN parameters in the model weights')
                model = integrate_bn_params(model)
            ###
            if args.pruner == 'neuron-flow':
                print('Pruning at threshold: ', threshold)
                model = prune_flow(model, threshold) 
            
            if args.pruner == 'random-balanced':
                print('Pruning randomly with balanced at density: ', threshold)
                model = prune_random_balanced(model, threshold)
            
            if args.pruner == 'random-global':
                print('Pruning globally randomly at density: ', threshold)
                model = prune_random_global(model, threshold) 
            
            if args.pruner == 'random-erk':
                print('Pruning with random (ERK) with density: ', threshold)
                model = prune_random_erk(model, threshold)

            if args.pruner == 'random-uniform':
                print('Pruning with random (Uniform) with density: ', threshold)
                model = prune_random_uniform(model, threshold) 

            if args.pruner == 'mag':
                print('Pruning by magnitude at density: ', threshold)
                model = prune_mag(model, threshold)    

            if args.pruner == 'snip':
                print('Pruning by SNIP at density: ', threshold)
                model = prune_snip(model, data.train_loader, criterion, threshold)     
            
            if args.pruner == 'synflow':
                print('Pruning by SynFlow at density: ', threshold)
                model = prune_synflow(model, data.train_loader, threshold)     
            
            if args.pruner == 'mask-ref':
                print('Pruning by loading an equivalent mask from a different experiment')
                model = prune_mask_ref(model, level, args.target_expt_name)

            if args.pruner == 'mag-with-bn':
                print('Pruning with magnitude corrected with BN gamma')
                model = prune_mag_integrated_bn(model, threshold)
            
            # compensation with LBFGS
            if args.lbfgs_compensate:
                print('compensation with LBFGS')
                lbfgs_compensate.step(model)

            mask_list = []
            for name, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    mask_list.append(m.mask)
            torch.save(mask_list, '{}runs/mask_{}_{}.pt'.format(base_dir, args.name, level))
            torch.save(model.state_dict(),"{}runs/model_{}_{}.pt".format(base_dir, args.name, level))
            level += 1

            torch.save(val_acc_total, base_dir + 'runs/val_acc_'+ args.name + '.pt')

            if args.reset_weights:
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('Weights of the model reset to initialization weights')
                
                if args.reset_weight_sign:
                    print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
                    sign_list = torch.load("{}runs/sign_list_{}_{}.pt".format(base_dir, args.load_expt_sign, level))
                    cnt = 0
                    for n, m in model.named_modules():
                        if isinstance(m, (ConvMask)):
                            m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                            cnt += 1
            
            if args.reset_shuffled_amplitude:

                sign_list = []
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        sign_list.append(m.weight.sign())

                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('Resetting only the amplitude during IMP and shuffling it, while keeping the signs!!!')

                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        buf = m.weight.data.clone()
                        perm = torch.randperm(buf.numel())
                        buf = buf.view(-1)[perm].view(buf.shape)
                        m.weight.data = buf.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

            if args.reset_only_bn:
                print('resetting BN parameters and the optimizer')
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                # original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                original_weights = {k:v for (k,v) in original_dict.items() if ('bn' in k) or ('downsample.1' in k)}
                print('resets the BN parameters along with the running mean and variance to the init values')
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
            
            if args.reset_only_weights_not_bn:
                print('resetting weights of the model and the optimizer, but not resetting the BN parameters')

                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                # original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                original_weights = {k:v for (k,v) in original_dict.items() if ('weight' in k) and (('conv' in k) or ('linear' in k) or ('downsample.0.' in k))}
                print('resets the only the weights to the initial values')
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
            

            if args.reset_only_bn_amplitude:
                print('resetting only BN parameter amplitudes and the optimizer')

                sign_list = []
                for n, m in model.named_modules():
                    if isinstance(m, (nn.BatchNorm2d)):
                        sign_list.append(m.weight.sign())

                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                # original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                original_weights = {k:v for (k,v) in original_dict.items() if ('bn' in k) or ('downsample.1' in k)}
                
                print('resets the BN parameters along with the running mean and variance to the init values')
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (nn.BatchNorm2d)):
                        m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
               

            if args.reset_weight_amplitude:

                sign_list = []
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        sign_list.append(m.weight.sign())

                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('Resetting only the amplitude during IMP, while keeping the signs!!!')

                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

            if args.reset_only_weight_sign:
                # Get the signs from the initial model.
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                sign_list = []
                for n in original_dict.keys():
                    if 'weight' in n and(('conv' in n) or ('linear' in n) or ('downsample.0.' in n)):
                        sign_list.append(original_dict[n].sign())

                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('resetting only the signs of the weights to the signs of IMP init and continuing LRR')
                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

    # save the mask of the sparse structure
    mask_list = []
    total_num = 0
    total_den = 0
    for name, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            mask_list.append(m.mask)
            total_num += m.mask.sum()
            total_den += m.mask.numel()
    print('Density before full training is: ', total_num / total_den)
    torch.save(mask_list, '{}runs/mask_{}.pt'.format(base_dir, args.name))


    # Start training
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    

    # Final training for the pruned network follows a cosine LR schedule
    assign_learning_rate(optimizer, args.lr)
    if args.prune_scheduler == 'cosine':
        prune_scheduler = cosine_lr(optimizer, args.final_prune_epoch, args)
    if args.prune_scheduler == 'step':
        prune_scheduler = multistep_lr(optimizer, args)
    if args.prune_scheduler == 'step-drops':
        prune_scheduler = multistep_lr_drops(optimizer, args)
    if args.prune_scheduler == 'imagenet-step':
        prune_scheduler = imagenet_lr_drops(optimizer, args)
        print('ImageNet Step drops LR scheduler')
    if args.prune_scheduler == 'step-warmup':
        prune_scheduler = multistep_lr_warmup(optimizer, args)
    if args.prune_scheduler == 'cosine-warmup':
        prune_scheduler = cosine_lr_warmup(optimizer, args.final_prune_epoch, args)
    if args.prune_scheduler == 'imagenet-step-warmup':
        prune_scheduler = imagenet_lr_drops_warmup(optimizer, args)
        print('ImageNet Step drops with warmup every cycle LR scheduler')

    level = 'final'

    if args.track_bn_running_stats:
        curr_mean_list = []
        running_mean_list = []
        curr_var_list = []
        running_var_list = []

    for epoch in range(args.start_epoch, args.epochs):
        prune_scheduler(epoch)
        cur_lr = get_lr(optimizer)
        print('The curent learning rate is: ', cur_lr)

        # train for one epoch
        start_train = time.time()

        if args.perturb_signs:
            if (epoch % args.perturb_sign_every) == args.perturb_sign_every - 1:
                print('Perturbing signs of model while training')
                model = perturb_signs(model, args.perturb_sign_ratio)

        if args.kd:
            print('Knowledge distillation epoch: ', epoch)
            train_acc1, train_acc5, kd_loss = train_kd(
                data.train_loader, model, teacher, optimizer, epoch, args, writer=writer)
            kd_train_loss.append(kd_loss)

        elif args.fix_sign_and_train:
            print('Training with weight signs fixed to initial values')
            train_acc1, train_acc5 = train_with_fixed_signs(
                init_sign_list, data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )

        else: 
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )
        train_time.update((time.time() - start_train) / 60)

        
        ## Tracking the running BN stats on the training data
        if args.track_bn_running_stats:
            ####
            cnt = 0
            for n, m in model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    # if m.running_mean is not None:
                    if args.bn_layer_idx == cnt:
                        curr_mean_list.append(m.curr_mean)
                        running_mean_list.append(m.running_mean)
                        curr_var_list.append(m.curr_var)
                        running_var_list.append(m.running_var)
                    cnt += 1

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)
        val_acc_total.append(acc1)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        curr_epoch += 1
        
        # get activation after validation
        if args.track_activation:
            get_activations(model, epoch, level, args)

        if args.hessian:
            trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
            hessian_list.append(trace)
        
        

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()
        torch.save(val_acc_total, base_dir + 'runs/val_acc_'+ args.name + '.pt')    
        if args.kd:
            torch.save(kd_train_loss, base_dir + 'runs/kd_train_loss_'+ args.name + '.pt')    

        # Storing sparsity and threshold statistics for STRConv models
        if args.conv_type == 'STRConv' or args.conv_type == 'STRConvER' or args.conv_type == 'ConvER' or args.conv_type == 'ConvMask':
            count = 0
            sum_sparse = 0.0
            for n, m in model.named_modules():
                if isinstance(m, (STRConv, STRConvER, ConvER, ConvMask)):
                    sparsity, total_params, thresh = m.getSparsity()
                    writer.add_scalar("sparsity/{}".format(n), sparsity, epoch)
                    writer.add_scalar("thresh/{}".format(n), thresh, epoch)
                    sum_sparse += int(((100 - sparsity) / 100) * total_params)
                    count += total_params
            total_sparsity = 100 - (100 * sum_sparse / count)
            writer.add_scalar("sparsity/total", total_sparsity, epoch)
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    torch.save(hessian_list, '{}runs/hessian_{}.pt'.format(base_dir, args.name))
    torch.save(model.state_dict(),"{}runs/model_{}_trained.pt".format(base_dir, args.name))
    if args.track_bn_running_stats:
        bn_stats = [
            curr_mean_list,
            running_mean_list,
            curr_var_list,
            running_var_list
            ]
        torch.save(bn_stats, "{}runs/bn_stats_{}_{}.pt".format(base_dir, args.name, args.bn_type))

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
        sparsity=total_sparsity,
    )

    

    if args.conv_type == "STRConv" or args.conv_type == 'STRConvER' or args.conv_type == 'ConvER' or args.conv_type == 'ConvMask':
        json_data = {}
        json_thres = {}
        for n, m in model.named_modules():
            if isinstance(m, (STRConv, STRConvER, ConvER, ConvMask)):
                sparsity = m.getSparsity()
                json_data[n] = sparsity[0]
                sum_sparse += int(((100 - sparsity[0]) / 100) * sparsity[1])
                count += sparsity[1]
                json_thres[n] = sparsity[2]
        json_data["total"] = 100 - (100 * sum_sparse / count)
        if not os.path.exists("runs/layerwise_sparsity"):
            os.mkdir("runs/layerwise_sparsity")
        if not os.path.exists("runs/layerwise_threshold"):
            os.mkdir("runs/layerwise_threshold")
        with open("runs/layerwise_sparsity/{}.json".format(args.name), "w") as f:
            json.dump(json_data, f)
        with open("runs/layerwise_threshold/{}.json".format(args.name), "w") as f:
            json.dump(json_thres, f)


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


def resume_from_checkpoint(model, model_name, mask_name):
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master/'

    model_name = base_dir + 'runs/' + model_name
    mask_name = base_dir + 'runs/' + mask_name
    model.load_state_dict(torch.load(model_name))
    mask_list = torch.load(mask_name)
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, ConvMask):
            m.mask = mask_list[cnt].to(m.weight.device)
            cnt += 1

    return model




def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()

        if not args.ignore_pretrained_weights:

            pretrained_final = {
                k: v
                for k, v in pretrained.items()
                if (k in model_state_dict and v.size() == model_state_dict[k].size())
            }

            if args.conv_type != "STRConv":
                for k, v in pretrained.items():
                    if 'sparseThreshold' in k:
                        wkey = k.split('sparse')[0] + 'weight'
                        weight = pretrained[wkey]
                        pretrained_final[wkey] = sparseFunction(weight, v)

            model_state_dict.update(pretrained_final)
            model.load_state_dict(model_state_dict)

        # Using the budgets of STR models for other models like DNW and GMP
        if args.use_budget:
            budget = {}
            for k, v in pretrained.items():
                if 'sparseThreshold' in k:
                    wkey = k.split('sparse')[0] + 'weight'
                    weight = pretrained[wkey]
                    sparse_weight = sparseFunction(weight, v)
                    budget[wkey] = (sparse_weight.abs() > 0).float().mean().item()

            for n, m in model.named_modules():
                if hasattr(m, 'set_prune_rate'):
                    pr = 1 - budget[n + '.weight']
                    m.set_prune_rate(pr)
                    print('set prune rate', n, pr)


    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

# Write a custom train loop and data loader to get activations sample wise and class wise for full filter and mean filter

def get_activations(model, epoch, level, args):
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master/'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master/'

    activation_list = []
    for n, m in model.named_modules():
        if isinstance(m, TrackActReLU):
            activation_list.append(m.avg_preacts)
    torch.save(activation_list, '{}runs/activations/activation_{}_{}_{}.pt'.format(base_dir, args.name, level, epoch))

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):

    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    

        

    # applying sparsity to the network
    if args.conv_type != "DenseConv":

        print(f"==> Setting prune rate of network to {args.prune_rate}")

        def _sparsity(m):
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(args.prune_rate)

        model.apply(_sparsity)

    # freezing the weights if we are only doing mask training
    if args.freeze_weights:
        print(f"=> Freezing model weights")

        def _freeze(m):
            if hasattr(m, "mask"):
                m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False

        model.apply(_freeze)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        sparse_thresh = [v for n, v in parameters if ("sparseThreshold" in n) and v.requires_grad]
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        # rest_params = [v for n, v in parameters if ("bn" not in n) and ('sparseThreshold' not in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {
                    "params": sparse_thresh,
                    "weight_decay": args.st_decay if args.st_decay is not None else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    filename = args.result_dir + '.csv'
    base_dir = ''
    if args.expt_setup == 'cispa':
        base_dir = '/home/c01adga/CISPA-projects/sparse_lottery-2022/STR-master'
    if args.expt_setup == 'juwels':
        base_dir = '/p/project/hai_efficientml/STR-master'

    results = pathlib.Path(base_dir, "runs", filename)
    
    if not results.exists():
        with open(results, 'w', newline=''):

            results.write_text(
                "Date Finished, "
                "Base Config, "
                "Name, "
                "Prune Rate, "
                "Current Val Top 1, "
                "Current Val Top 5, "
                "Best Val Top 1, "
                "Best Val Top 5, "
                "Best Train Top 1, "
                "Best Train Top 5,"
                "Sparsity\n"
            )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f},"
                "{sparsity:.04f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
