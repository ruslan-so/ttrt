import argparse
import os
import random
import shutil
import time
import warnings
import importlib
import re
import datetime
import sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from modules import lowerprecision as lp
import numpy as np
from utilities import scale_fn

module_names = {
    'imagenet' : 'models_imagenet',
    'cifar' : 'models_cifar',
    'cifardecem' : 'models_cifardecem',
    'cifarcentum' : 'models_cifarcentum',
    'less_data' : 'models_less_data'
    }

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
                    # choices=model_names,
                    # help='model architecture: ' +
                    #    ' | '.join(model_names) +
                    #    ' (default: resnet18)')
parser.add_argument('-d', '--dataset', default='cifar', metavar='DATASET')
parser.add_argument('-f', '--freezeout', default=False, metavar='FREEZEOUT')
parser.add_argument('--lp', '--low-precision', default=False, metavar='LOWPRECISION')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

def main():
    args = parser.parse_args()

    # output hyperparameters values
    print('# Model: ' + str(args.arch))
    print('# Dataset: ' + str(args.dataset))
    print('# Batch size: ' + str(args.batch_size))
    print('# Freezeout: ' + str(args.freezeout))
    print('# Low precision: ' + str(args.lp))
    print('# Epochs: ' + str(args.epochs))
    print('# Initial learning rate: ' + str(args.lr))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    #global log_file
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # get dataset name and model name
    arg_dataset = module_names[args.dataset]
    arg_model = args.arch

    # remove model version (e.g. vgg11 -> vgg)
    model_type = (re.match(r"[a-zA-Z]+", arg_model)).group(0)
    module_name = arg_dataset + "." + model_type
    print("# module_name: " + module_name)

    # import the module for specified dataset
    module = importlib.import_module(module_name)

    # get class 
    cnn = getattr(module, arg_model)
    print(cnn)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = cnn()
    else:
        print("# model requested: '{}'".format(args.arch))
        model = cnn()

    print("# printing out the model")
    print(model)

    #model -> to lower precision
    if args.lp:
        model.half()
        print("# model is low precision")
    else:
        print("# model is full precision")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.freezeout:
        first = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                if first:
                    optimizer = torch.optim.SGD([{'params': param, 'param_name': name,
                        'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}])
                    first = False
                else:
                    optimizer.add_param_group({'params': param, 'param_name': name,
                        'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay})

        lr_schedule = init_lr(model, optimizer, args)
        print_lr_schedule(lr_schedule)
        print_model_parameters(model)
        print_optimizer_groups_columns(optimizer)
        print_optimizer_groups_rows(optimizer)

    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    '''
    # compose log file name
    log_file = args.arch + "_" + args.dataset + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".out"

    # write hyperparameters to log file
    with open(log_file, 'a') as f:
        f.write('# Learning rate: ' + str(args.lr) + '\n')
        f.write('# Criterion: ' + str(criterion) + '\n')
        f.write('# Optimizer: ' + str(optimizer.__class__.__name__) + '\n')
        for hp in optimizer.param_groups[0]:
            if hp != 'params':    #skip params
                f.write('#\t' + hp + ' ' + str(optimizer.param_groups[0][hp]) +'\n')

    '''

    # output hyperparameters values
    print('# Model: ' + str(args.arch))
    print('# Dataset: ' + str(args.dataset))
    print('# Freezeout: ' + str(args.freezeout))
    print('# Low precision: ' + str(args.lp))
    print('# Initial learning rate: ' + str(args.lr))
    print('# Criterion: ' + str(criterion))
    print('# Optimizer: ' + str(optimizer.__class__.__name__))
    for hp in optimizer.param_groups[0]:
        if hp != 'params':    #skip params
            print('#\t' + hp + ' ' + str(optimizer.param_groups[0][hp]))

    # Data loading code
    # CIFAR10 data loading and normalization
    if args.dataset == 'cifardecem':
        print("CIFAR10 loading and normalization")
        print('==> Preparing data..')

        if args.lp:
            print("# CIFAR10 / low precision")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                lp.ToTensor(),
                lp.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                lp.ToTensor(),
                lp.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        else:
            print("# CIFAR10 / full precision")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data-decem', train=True, download=True, transform=transform_train)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

        val_dataset = torchvision.datasets.CIFAR10(root='./data-decem', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)

        if args.evaluate:
            validate(val_loader, model, criterion, args)
            return

    # CIFAR 100 data loading and normalization
    elif args.dataset == 'cifarcentum':
        print("CIFAR100 loading and normalization")
        print('==> Preparing data..')

        if args.lp:
            print("# CIFAR100 / low precision")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                lp.ToTensor(),
                lp.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

            transform_test = transforms.Compose([
                lp.ToTensor(),
                lp.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

        else:
            print("# CIFAR100 / full precision")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

        train_dataset = torchvision.datasets.CIFAR100(root='./data-centum', train=True, download=True, transform=transform_train)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

        val_dataset = torchvision.datasets.CIFAR100(root='./data-centum', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)

        if args.evaluate:
            validate(val_loader, model, criterion, args)
            return

    # IMAGENET or LESS_DATA data loading and normalization
    elif (args.dataset == 'imagenet') or (args.dataset == 'less_data'):

        print("From IMAGENET dataset:")
        print('==> Preparing data..')

        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        if args.dataset == 'imagenet':

            print("IMAGENET (224x224) loading and normalization")

            resize_dim = 256
            crop_dim = 224

        else:

            print("IMAGENET-LESS_DATA (112x112) loading and normalization")
            print("# resize 128x128 and centercrop 112x112")

            resize_dim = 128
            crop_dim = 112

        print("# resized to " + str(resize_dim))
        print("# cropped to " + str(crop_dim))

        if args.lp:
            print("#--> training with low precision")
            normalize = lp.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.RandomCrop(crop_dim),
                    transforms.RandomHorizontalFlip(),
                    lp.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.CenterCrop(crop_dim),
                    lp.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            if args.evaluate:
                validate(val_loader, model, criterion, args)
                return

        else:   # full precision
            print("#--> training with full precision")
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.RandomCrop(crop_dim),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(resize_dim),
                    transforms.CenterCrop(crop_dim),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            if args.evaluate:
                validate(val_loader, model, criterion, args)
                return

    else:
        print("missing DATASET requested")
        print("Exiting.")
        exit()

    print('#--> Training data: <--#')
    print('#-->>')

    # set the timer for total training(only) time 
    total_training = 0

    for epoch in range(args.start_epoch, args.epochs):
        print("EPOCH " + str(epoch))
        # set the timer for one epoch execution time
        epoch_start = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.freezeout:
            adjust_lrs(model, optimizer, lr_schedule, epoch, args)
            #print_model_parameters(model)
            #print_optimizer_groups_rows(optimizer)
            print("")
        else:
            adjust_learning_rate(optimizer, epoch, args)

        # set the timer for training(only) time for one epoch 
        training_start = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # calc training time
        training_time = time.time() - training_start
        total_training = total_training + training_time
        print("## epoch[" + str(epoch) + "] training(only) time: " + str(training_time))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

        epoch_time = time.time() - epoch_start
        print("### epoch[" + str(epoch) + "] execution time: " + str(epoch_time))

    print("### Training complete:")
    if args.freezeout:
        print_model_parameters(model)
        print_optimizer_groups_columns(optimizer)
        print_optimizer_groups_rows(optimizer)

    print("#### total training(only) time: " + str(total_training))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    print("# Switched to train mode...")

    optimizerzerograd_time = 0
    lossbackward_time = 0
    optimizerstep_time = 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        # and measure each step's elapsed time
        ozg = time.time()
        optimizer.zero_grad()
        optimizerzerograd_time += time.time() - ozg

        lbw = time.time()
        loss.backward()
        lossbackward_time += time.time() - lbw

        opt = time.time()
        optimizer.step()
        optimizerstep_time += time.time() - opt

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print("## e[" + str(epoch) + "] optimizer.zero_grad (sum) time: " + str(optimizerzerograd_time))
    print("## e[" + str(epoch) + "]       loss.backward (sum) time: " + str(lossbackward_time))
    print("## e[" + str(epoch) + "]      optimizer.step (sum) time: " + str(optimizerstep_time))


def validate(val_loader, model, criterion, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    print("# Switched to evaluate mode...")

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        #with open(log_file, 'a') as f:
        #    f.write('{top1.avg:.3f}\n'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    #disabled saving state

    #torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')
    return

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def print_lr_schedule(lr_sch):
    for key in lr_sch:
        print(key)
        print(lr_sch[key])


def print_model_parameters(model):
    print("*** Model named parameters and requires_grad:")
    for name, param in model.named_parameters():
        print("name: %30s  req_grad: %5s " % (name, param.requires_grad))

def print_optimizer_groups_columns(optimizer):

    print("*** Optimizer groups, parameters and req_grads")
    for group in optimizer.param_groups:
        for hp in group:
            if hp == 'params':
                for t in group[hp]:
                    print("# %20s:  %30s" % ('requires_grad', t.requires_grad))
            else:
                print("# %20s:  %30s" % (hp, group[hp]))
        print()

def print_optimizer_groups_rows(optimizer):
    print("*** Optimizer group lrs")
    for i,group in enumerate(optimizer.param_groups):
        print("# group: %2i,   name: %30s,   req_grad: %6s   lr: %20.18f,   mmm: %7.5f,   weight_decay: %6.1f,   damp: %6.1f,   nesterov: %6s"
                  % (i, group['param_name'], group['params'][0].requires_grad, group['lr'],\
                        group['momentum'], group['weight_decay'], group['dampening'], group['nesterov']))

def init_lr(model, optimizer, args):

    how_scale = 'cubic'
    t_0 = 0.8
    scale_lr = False

    param_count = sum(1 for e in optimizer.param_groups)
    freeze_count = sum(1 for e in optimizer.param_groups if not (('classifier' in e['param_name']) or ('fc' in e['param_name'])))

    # debug
    print("PARAM TOTAL COUNT: " + str(param_count))
    print("PARAM TO FREEZE COUNT: " + str(freeze_count))
    print()

    lr_sch = {}
    for i, group in enumerate(optimizer.param_groups):
        key = optimizer.param_groups[i]['param_name']
        lr_sch[key] = [args.lr]

        lr_ratio = scale_fn[how_scale](t_0 + (1 - t_0) * float(i) / param_count)

        max_j = args.epochs * lr_ratio
        lr = args.lr / lr_ratio if scale_lr else args.lr

        for epoch in range(1, args.epochs):
            next_lr = (((args.lr * 0.5)/lr_ratio) * (1+np.cos(np.pi*epoch/max_j)) + args.lr*0.01)\
                                                     if scale_lr else ((args.lr * 0.5) * (1+np.cos(np.pi*epoch/max_j)) + args.lr*0.01)
            if max_j > epoch:
                lr_sch[key].append(next_lr)
            else:
                lr_sch[key].append(next_lr if (('classifier' in key) or ('fc' in key)) else 0)

    return lr_sch

def adjust_lrs(model, optimizer, lr_sch, epoch, args):

    for name, param in model.named_parameters():
        for group in optimizer.param_groups:
            for p in group:
                if (p == 'param_name' and group['param_name'] == name):

                    if lr_sch[name][epoch] == 0:
                        print('REMOVING: ' + str(name))
                        for t in group['params']:
                            t.requires_grad = False
                        optimizer.param_groups.remove(group)

    for i, group in enumerate(optimizer.param_groups):
        param_name = optimizer.param_groups[i]['param_name']
        # debugging (2 lines)
        if lr_sch[param_name][epoch] == 0:
            print("removing " + str(param_name))
        else:
            print("i: %3i, name: %30s  changing lr from: %20.18f   to: %20.18f"
                   % (i, group['param_name'], group['lr'], lr_sch[param_name][epoch]))
            optimizer.param_groups[i]['lr'] = lr_sch[param_name][epoch]


# added: different adjustment rate for cifar
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr * (0.1 ** (epoch // (args.epochs / 3)))

    '''
    if (args.dataset == 'imagenet') or (args.dataset == 'less_data'):
        lr = args.lr * (0.1 ** (epoch // 30))
    # elif args.dataset == 'cifar':
    else:
        #lr = args.lr * (0.1 ** (epoch // 120))
        lr = args.lr * (0.1 ** (epoch // 80))
    '''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("# current learning rate: " + str(lr))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    time_zero = time.time()

    main()

    print("##### Total run time: " + str(time.time() - time_zero))
