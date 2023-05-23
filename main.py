# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
import os
import numpy as np
import torch
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from engine import evaluate_swig, train_one_epoch
from models import build_model
from pathlib import Path
from data_loader import ImSituVerbGender
import torchvision.transforms as transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Set grounded situation recognition transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_verb', type=int, default = 211)
    # Loss coefficients
    parser.add_argument('--img_loss_coef', default=1, type=float)
    parser.add_argument('--mask_loss_coef', default=0, type=float)

    # Dataset parameters
    parser.add_argument('--annotation_dir', type=str,
            default='./data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = './data/of500_images_resized',
            help='image directory')
    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training')
    parser.add_argument('--gender_balanced', action='store_true',
            help='use balanced subset for training, ratio will be 1/2/3')
    parser.add_argument('--batch_balanced', action='store_true',
            help='in every batch, gender balanced')

    parser.add_argument('--no_image', action='store_true',
            help='do not load image in dataloaders')
    
    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)

    # Etc...
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--inference', default=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', type=str,help='path for saving log files')
    parser.add_argument('--device', default='cuda', 
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--saved_model', default='gsrtr_checkpoint.pth',
                        help='path where saved model is')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

  # image preprocessing
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # Data samplers.
    train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'train', transform = train_transform)

    val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)

    
    # build model
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    # optimizer & LR scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset sampler

    if args.distributed:
        sampler_train = DistributedSampler(train_data)
        sampler_val = DistributedSampler(val_data, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_data)
        sampler_val = torch.utils.data.SequentialSampler(val_data)


    output_dir = Path(args.output_dir)
    # dataset loader
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(train_data, num_workers=args.num_workers,
                                batch_sampler=batch_sampler_train,pin_memory = True)
    data_loader_val = DataLoader(val_data, num_workers=args.num_workers,
                                drop_last=False, sampler=sampler_val, pin_memory = True)

    
    # use saved model for evaluation (using dev set or test set)
    if args.test:
        checkpoint = torch.load(args.saved_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        data_loader = data_loader_val

        test_stats = evaluate_swig(model, criterion, data_loader, device, args.output_dir)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return None

    # train model
    print("Start training")
    start_time = time.time()
    max_test_mean_acc = 10
    for epoch in range(args.start_epoch, args.epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, 
                                      device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        # evaluate
        test_stats = evaluate_swig(model, criterion, data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_mean_acc_unscaled'] > max_test_mean_acc:
                max_test_mean_acc = log_stats['test_mean_acc_unscaled']
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'lr_scheduler': lr_scheduler.state_dict(),
                                      'epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GSRTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)