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
    parser.add_argument('--log_dir', type=str,
            help='path for saving log files')
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

    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--mask_mode', type=str, default='pixel',
                     help='pixel or patch')
    parser.add_argument('--patch_size', type=int, default=32,
                     help='patch size')
    parser.add_argument('--mask_ratio', type=int, default=10,
                     help='Percentage to mask the image')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model save directory
    args.save_dir = os.path.join('./checkpoints', args.save_dir)
    # if os.path.exists(args.save_dir) and not args.resume:
    #     print('Path {} exists! and not resuming'.format(args.save_dir))
    #     return
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # create log save directory for train and val
    args.log_dir = os.path.join('./logs', args.log_dir)
    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir): os.makedirs(val_log_dir)
    train_logger = Logger(train_log_dir)
    val_logger = Logger(val_log_dir)

    #save all hyper-parameters for training
    with open(os.path.join(args.log_dir, "arguments.txt"), "a") as f:
        f.write(str(args)+'\n')

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

    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)

    # build the models
    channel = 3
    patch_size = 8
    d_model = 512
    n_layers = 6
    n_head = 8
    ff_dim = 2048
    dropout_rate = 0.15
    mask_output_dim = 2
    fair_output_dim = args.num_verb
    img_size = (val_loader.dataset[0][0].shape[0], val_loader.dataset[0][0].shape[1])
    fair_model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, fair_output_dim)
    fair_model = fair_model.cuda()
    mask_model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, mask_output_dim)
    mask_model = mask_model.cuda()

    # build optimizer and scheduler
    optimizer = optim.SGD(fair_model.parameters(),args.learning_rate,0.9)
    scheduler = WarmupCosineSchedule(optimizer, args.num_epochs // 5, args.num_epochs)
    mask_criterion = nn.CrossEntropyLoss(reduction='elementwise_mean').cuda()
    orig_criterion = nn.CrossEntropyLoss(reduction='elementwise_mean').cuda()

    best_performance = 0
    if args.resume:
        if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint '{}'".format(args.save_dir))
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            fair_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))

    # args.mask_dir = os.path.join('./models', args.mask_dir)
    # if os.path.isfile(os.path.join(args.mask_dir, 'model_best.pth.tar')):
    #     print("=> loading mask model '{}'".format(args.mask_dir))
    #     checkpoint = torch.load(os.path.join(args.mask_dir, 'model_best.pth.tar'))
    #     mask_model.load_state_dict(checkpoint['state_dict'])
    #     mask_model.eval()
        
        
    print('before training, evaluate the model')
    test(args, 0, fair_model, mask_model, mask_criterion, orig_criterion, val_loader, val_logger, logging=False)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, fair_model, mask_model, mask_criterion, orig_criterion, train_loader, optimizer, scheduler,\
                train_logger, logging = True)
        current_performance = test(args, epoch, fair_model, mask_model, mask_criterion, orig_criterion, val_loader, \
                val_logger, logging = True)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': fair_model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

        # at the end of every run, save the model
        if epoch == args.num_epochs:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%s.pth.tar' % str(args.num_epochs)))

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))


def train(args, epoch, fair_model, mask_model, mask_criterion, orig_criterion, train_loader, optimizer, scheduler,\
    train_logger, logging=True):
    fair_model.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset) # number of images
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    res = list()

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):

        if args.batch_balanced:
            man_idx = genders[:, 0].nonzero().squeeze()
            if len(man_idx.size()) == 0: man_idx = man_idx.unsqueeze(0)
            woman_idx = genders[:, 1].nonzero().squeeze()
            if len(woman_idx.size()) == 0: woman_idx = woman_idx.unsqueeze(0)
            selected_num = min(len(man_idx), len(woman_idx))

            if selected_num < args.batch_size / 2:
                continue # skip the batch if the selected num is too small
            else:
                selected_num = args.batch_size / 2
                selected_idx = torch.cat((man_idx[:selected_num], woman_idx[:selected_num]), 0)

            images = torch.index_select(images, 0, selected_idx)
            targets = torch.index_select(targets, 0, selected_idx)
            genders = torch.index_select(genders, 0, selected_idx)

        # set mini-batch dataset
        # masked_images = mask_image(args, mask_model, images)
        # masked_images = masked_images.cuda()
        targets = targets.cuda()

        # forward, Backward and Optimize
        # mask_preds, _ = fair_model(masked_images)
        orig_preds, _ = fair_model(images.cuda())

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        # mask_loss = mask_criterion(mask_preds, targets.max(1, keepdim=False)[1])
        orig_loss = orig_criterion(orig_preds, targets.max(1, keepdim=False)[1])
        mask_para = 1
        orig_para = 1
        # loss = mask_para*mask_loss + orig_para*orig_loss
        loss = orig_loss
        loss_logger.update(loss.item())

        orig_preds = F.softmax(orig_preds, dim=1)
        preds_max = orig_preds.max(1, keepdim=False)[1]
        # mask_preds = F.softmax(mask_preds, dim=1)
        # preds_max = mask_preds.max(1, keepdim=True)[1]
        # print("orig pred max {}".format(preds_max))

        # save the exact preds (binary)
        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(orig_preds.size())
        # preds_exact = tensor.new_zeros(mask_preds.size())
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1
        
        res.append((image_ids, orig_preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))
        # res.append((image_ids, mask_preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)
        
    # compute mean average precision score for verb classifier
    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_genders = torch.cat([entry[3] for entry in res], 0)
    total_preds_exact = torch.cat([entry[4] for entry in res], 0)

    # compute f1 score (no threshold as we simple take the max for multi-classification)
    task_f1_score = f1_score(total_targets.numpy(), total_preds_exact.numpy(), average = 'macro')

    man_idx = total_genders[:, 0].nonzero().squeeze()
    woman_idx = total_genders[:, 1].nonzero().squeeze()

    preds_man = torch.index_select(total_preds, 0, man_idx)
    preds_woman = torch.index_select(total_preds, 0, woman_idx)
    targets_man = torch.index_select(total_targets, 0, man_idx)
    targets_woman = torch.index_select(total_targets, 0, woman_idx)

    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
    meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')

    if logging:
        train_logger.scalar_summary('loss', loss_logger.avg, epoch)
        train_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        train_logger.scalar_summary('meanAP', meanAP, epoch)
        train_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        train_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Train epoch  : {}, task_f1_score, {:.2f} meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, task_f1_score, meanAP, meanAP_man, meanAP_woman))
    for name, child in fair_model.named_children():
            for param in child.parameters():
                print(name, param)


def test(args, epoch, fair_model, mask_model, mask_criterion, orig_criterion, val_loader, val_logger, logging=True):

    # set the eval mode
    fair_model.eval()
    nProcessed = 0
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        # masked_images = mask_image(args, mask_model, images)
        # masked_images = masked_images.cuda()
        targets = targets.cuda()

        # forward, Backward and Optimize
        # mask_preds, _ = fair_model(masked_images)
        orig_preds, _ = fair_model(images.cuda())

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        # mask_loss = mask_criterion(mask_preds, targets.max(1, keepdim=False)[1])
        orig_loss = orig_criterion(orig_preds, targets.max(1, keepdim=False)[1])
        mask_para = 1
        orig_para = 1
        # loss = mask_para*mask_loss + orig_para*orig_loss
        loss = orig_loss
        loss_logger.update(loss.item())

        orig_preds = F.softmax(orig_preds, dim=1)
        preds_max = orig_preds.max(1, keepdim=False)[1]
        # mask_preds = F.softmax(mask_preds, dim=1)
        # preds_max = mask_preds.max(1, keepdim=True)[1]
        print("orig pred max {}".format(preds_max))


        # save the exact preds (binary)
        tensor = torch.tensor((), dtype=torch.float64)
        preds_exact = tensor.new_zeros(orig_preds.size())
        # preds_exact = tensor.new_zeros(mask_preds.size())
        for idx, item in enumerate(preds_max):
            preds_exact[idx, item] = 1
        
        # res.append((image_ids, orig_preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))
        res.append((image_ids, orig_preds.detach().cpu(), targets.detach().cpu(), genders, preds_exact))

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    # compute mean average precision score for verb classifier
    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_genders = torch.cat([entry[3] for entry in res], 0)
    total_preds_exact = torch.cat([entry[4] for entry in res], 0)

    task_f1_score = f1_score(total_targets.numpy(), total_preds_exact.numpy(), average = 'macro')

    man_idx = total_genders[:, 0].nonzero().squeeze()
    woman_idx = total_genders[:, 1].nonzero().squeeze()

    preds_man = torch.index_select(total_preds, 0, man_idx)
    preds_woman = torch.index_select(total_preds, 0, woman_idx)
    targets_man = torch.index_select(total_targets, 0, man_idx)
    targets_woman = torch.index_select(total_targets, 0, woman_idx)

    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
    meanAP_man = average_precision_score(targets_man.numpy(), preds_man.numpy(), average='macro')
    meanAP_woman = average_precision_score(targets_woman.numpy(), preds_woman.numpy(), average='macro')

    if logging:
        val_logger.scalar_summary('loss', loss_logger.avg, epoch)
        val_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        val_logger.scalar_summary('meanAP', meanAP, epoch)
        val_logger.scalar_summary('meanAP_man', meanAP_man, epoch)
        val_logger.scalar_summary('meanAP_woman', meanAP_woman, epoch)

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Val epoch  : {}, task_f1_score, {:.2f} meanAP: {:.2f}, meanAP_man: {:.2f}, meanAP_woman: {:.2f}'.format( \
        epoch, task_f1_score, meanAP, meanAP_man, meanAP_woman))

    return task_f1_score

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

if __name__ == '__main__':
    main()
