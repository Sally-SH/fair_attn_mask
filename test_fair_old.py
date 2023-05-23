import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm
from einops import rearrange, reduce, repeat

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from pytorch_transformers.optimization import WarmupCosineSchedule

from data_loader import ImSituVerbGender
from model import VisionTransformer
from logger import Logger

verb_id_map = pickle.load(open('./data/verb_id.map', 'rb'))
verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
            help='path for saving checkpoints')
    parser.add_argument('--mask_dir', type=str,
            help='path to bias model')

    parser.add_argument('--ratio', type=str,
            default = '0')
    parser.add_argument('--num_verb', type=int,
            default = 211)

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

    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--batch_size', type=int, default=5)

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--mask_mode', type=str, default='pixel',
                     help='pixel or patch')
    parser.add_argument('--patch_size', type=int, default=16,
                     help='patch size')
    parser.add_argument('--mask_ratio', type=int, default=10,
                     help='Percentage to mask the image')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # image preprocessing
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # Data samplers.
    val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)

    # build the models
    channel = 3
    patch_size = 16
    d_model = 64
    n_layers = 12
    n_head = 8
    ff_dim = 256
    dropout_rate = 0.1
    mask_output_dim = 2
    fair_output_dim = args.num_verb
    img_size = (val_loader.dataset[0][0].shape[0], val_loader.dataset[0][0].shape[1])
    fair_model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, fair_output_dim)
    fair_model = fair_model.cuda()
    mask_model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, mask_output_dim)
    mask_model = mask_model.cuda()
    fair_model.eval()
    mask_model.eval()

    if os.path.isfile(os.path.join('./checkpoints', args.save_dir, 'model_best.pth.tar')):
        print("=> loading checkpoint '{}'".format(args.save_dir))
        checkpoint = torch.load(os.path.join('./checkpoints', args.save_dir, 'model_best.pth.tar'))
        args.start_epoch = checkpoint['epoch']
        best_performance = checkpoint['best_performance']
        fair_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        print("best performance : {}".format(best_performance))
    else:
        print("=> no checkpoint found at '{}'".format(args.save_dir))
        return None

    args.mask_dir = os.path.join('./checkpoints', args.mask_dir)
    if os.path.isfile(os.path.join(args.mask_dir, 'model_best.pth.tar')):
        print("=> loading mask model '{}'".format(args.mask_dir))
        checkpoint = torch.load(os.path.join(args.mask_dir, 'model_best.pth.tar'))
        mask_model.load_state_dict(checkpoint['state_dict'])
        mask_model.eval()

    visualize_result(args, mask_model, fair_model, val_loader)
    
def visualize_result(args, mask_model, fair_model, val_loader):
    
    denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    # set the eval mode
    for batch_idx, (images, targets, genders, image_ids) in enumerate(val_loader):
        # Set mini-batch dataset
        masked_images = mask_image(args, mask_model, images)
        masked_images = denormalize(masked_images)
        masked_images = rearrange(masked_images, 'b c h w -> b h w c')
        masked_images = masked_images.detach().numpy()

        # forward, Backward and Optimize
        _, orig_attn = fair_model(images.cuda())

        att_mat = torch.stack(orig_attn)
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
        attentions = rearrange(v[:, 0, 1:],'b (h w) -> b h w', h=grid_size, w=grid_size)
        attentions = attentions.detach().numpy()

        images = denormalize(images)
        images = rearrange(images, 'b c h w -> b h w c')

        for i in range(images.shape[0]):
            img = images[i]
            attn = attentions[i]
            mask = masked_images[i]

            if args.mask_mode == 'pixel':
                attn = cv2.resize(attn / attn.max(), (img.shape[1], img.shape[0]))
                flatten_mask = attn.reshape(-1)
                flatten_mask = np.sort(flatten_mask)
                # select the top mask_ratio% attention value
                mask_val = flatten_mask[int((1-args.mask_ratio*0.01)*len(flatten_mask))]
                attn = attn[...,np.newaxis]
            else:
                attn = attn/attn.max()
                flatten_mask = attn.reshape(-1)
                flatten_mask = np.sort(flatten_mask)
                # select the top mask_ratio% attention value
                mask_val = flatten_mask[int((1-args.mask_ratio*0.01)*len(flatten_mask))]
                attn = np.repeat(np.repeat(attn,args.patch_size,0),args.patch_size,1)
                img_mask = np.where(attn > mask_val, 0, 1)
                img_mask = img_mask[...,np.newaxis]
                attn = attn[...,np.newaxis]

            heatmap = cv2.applyColorMap(np.uint8(225*attn), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
            result = heatmap + np.float32(img)
            result = result / np.max(result)
            result = np.uint8(255*result)

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

            ax1.set_title('Original')
            ax2.set_title('Attention Mask')
            ax3.set_title('Attention Map')
            _ = ax1.imshow(img)
            _ = ax2.imshow(mask)
            _ = ax3.imshow(result)
            plt.savefig('img_{}.png'.format(i))

        break


def mask_image(args, mask_model, images):
    to_tensor = torchvision.transforms.ToTensor()
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

if __name__ == '__main__':
    main()
