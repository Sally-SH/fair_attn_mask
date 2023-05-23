# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import util.misc as utils
from typing import Iterable
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
import numpy as np
import torch.nn.functional as F
import cv2

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for images, targets, genders, image_ids in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        images = images.to(device)
        targets = targets.to(device)
        # model output & calculate loss
        outputs, _ = model(images)
        loss_dict = criterion(targets, outputs)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_swig(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    for images, targets, genders, image_ids in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        images = images.to(device)
        targets = targets.to(device)

        # model output & calculate loss
        outputs, _ = model(images)
        loss_dict = criterion(targets, outputs, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


def train_one_epoch_mask(args, mask_model, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for images, targets, genders, image_ids in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        images = images.to(device)
        targets = targets.to(device)
        masked_images = mask_image(args, mask_model, images)
        masked_images = masked_images.to(device)

        # model output & calculate loss
        cat_img = torch.cat([images,masked_images], dim=0)
        outputs, _ = model(cat_img)
        b = targets.shape[0]
        img_outputs = outputs[:b,...]
        mask_outputs = outputs[b:,...]

        loss_dict = criterion(img_outputs=img_outputs, mask_outputs=mask_outputs, targets=targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_fair_mask(args, mask_model, model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    for images, targets, genders, image_ids in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        images = images.to(device)
        targets = targets.to(device)
        masked_images = mask_image(args, mask_model, images)
        masked_images = masked_images.to(device)

        # model output & calculate loss
        # img_outputs, _ = model(images)
        cat_img = torch.cat([images,masked_images], dim=0)
        outputs, _ = model(cat_img)
        b = targets.shape[0]
        img_outputs = outputs[:b,...]
        mask_outputs = outputs[b:,...]
        # mask_outputs, _ = model(masked_images)
        loss_dict = criterion(img_outputs=img_outputs, mask_outputs=mask_outputs, targets=targets, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats



def mask_image(args, mask_model, images):
    to_tensor = transforms.ToTensor()
    preds, attentions = mask_model(images.cuda())
    preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1)[0]
    att_mat = torch.stack(attentions)
    att_mat = att_mat.cpu().detach()

    att_mat = reduce(att_mat, 'l b hd h w -> l b h w', 'mean')
    residual_att = torch.eye(att_mat.size(2))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    masks = rearrange(v[:, 0, 1:],'b (h w) -> b h w', h=grid_size, w=grid_size)
    masks = masks.detach().numpy()

    images = rearrange(images, 'b c h w -> b h w c')

    masked_imgs = []

    for i in range(images.shape[0]):
        img = images[i].cpu()
        mask = masks[i]
        if args.mask_mode == 'pixel':
            mask = cv2.resize(mask / mask.max(), (img.shape[1], img.shape[0]))
            flatten_mask = mask.reshape(-1)
            flatten_mask = np.sort(flatten_mask)
            # select the top mask_ratio% attention value
            mask_val = flatten_mask[int((1-args.mask_ratio*0.01)*len(flatten_mask))]
            mask = mask[...,np.newaxis]
            # mask top mask_ratio% attention area
            masked_img = np.where(np.repeat(mask, 3, axis=2)>mask_val, 0, img)
            masked_imgs.append(to_tensor(masked_img))
        else:
            mask = mask/mask.max()
            flatten_mask = mask.reshape(-1)
            flatten_mask = np.sort(flatten_mask)
            # select the top mask_ratio% attention value
            mask_val = flatten_mask[int((1-args.mask_ratio*0.01)*len(flatten_mask))]
            mask = np.repeat(np.repeat(mask,args.patch_size,0),args.patch_size,1)
            img_mask = np.where(mask > mask_val, 0, 1)
            img_mask = img_mask[...,np.newaxis]
            mask = mask[...,np.newaxis]
            # mask top mask_ratio% attention area
            masked_img = img*img_mask
            masked_imgs.append(to_tensor(masked_img))
    
    masked_images = torch.stack(masked_imgs)

    return masked_images