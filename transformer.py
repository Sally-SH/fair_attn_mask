import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm
from einops import reduce

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from pytorch_transformers.optimization import WarmupCosineSchedule

from data_loader import ImSituVerbGender, get_loader
from model import VisionTransformer, ResNet
from logger import Logger


verb_id_map = pickle.load(open('./data_imsitu/verb_id.map', 'rb'))
verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./tf_models/',
            help='path for saving checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
            help='path for saving log files')

    parser.add_argument('--ratio', type=str,
            default = '0')
    parser.add_argument('--dataset', type=str, default='celeba', choices=['imsitu', 'celeba'])

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
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    if args.dataset == 'imsitu':
        parser.add_argument('--num_verb', type=int,
                default = 211)
        parser.add_argument('--annotation_dir', type=str,
                default='./data_imsitu',
                help='annotation files path')
        parser.add_argument('--image_dir',
                default = './data_imsitu/of500_images_resized',
                help='image directory')
    else:
        parser.add_argument('--target_att', default=['Gray_Hair', 'Male'],
                            help='list of target attributes and gender(male or not)')
        parser.add_argument('--num_att', type=int,
                default = 2,
                help='number of target attribute')
        parser.add_argument('--annotation_dir', type=str,
                default='./data_celeba/list_attr_celeba.txt',
                help='annotation files path')
        parser.add_argument('--image_dir',
                default = './data_celeba/images',
                help='image directory') # 202599
        parser.add_argument('--train_val_split', default=0.9,
                help='Percentage of train data among total datasets, ex) train 90%, val 10%')
    args = parser.parse_args()
    


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model save directory
    args.save_dir = os.path.join('./models', args.save_dir)
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
        #transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # Data samplers.
    if args.dataset == 'imsitu':
        train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir, split = 'train', transform = train_transform)

        val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir, split = 'val', transform = val_transform)
        
        # Data loaders / batch assemblers.
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)

        val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)
        test = test_imsitu
        train = train_imsitu
        
    else: # CelebA
        train_loader = get_loader(args.image_dir, args.annotation_dir, args.target_att,
                                   args.crop_size, args.image_size, args.batch_size,
                                   'CelebA', 'Train', args.train_val_split, num_workers=6) # 20261 images
        val_loader = get_loader(args.image_dir, args.annotation_dir, args.target_att,
                                   args.crop_size, args.image_size, args.batch_size,
                                   'CelebA', 'Test', args.train_val_split, num_workers=4) # 182338
        test = test_celeba
        train = train_celeba


    # build the models
    channel = 3
    patch_size = 16
    d_model = 256#512
    n_layers = 6
    n_head = 8
    ff_dim = 128#2048
    dropout_rate = 0.15
    output_dim = args.num_verb if args.dataset=='imsitu' else args.num_att
    img_size = (val_loader.dataset[0][0].shape[0], val_loader.dataset[0][0].shape[1])
    model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, output_dim)
    #model = ResNet(BasicBlock, [2,2,2,2])
    model = model.cuda()
    print('Model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))

    # build optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # optimizer = optim.SGD(model.parameters(),args.learning_rate,0.9)
    # scheduler = WarmupCosineSchedule(optimizer, args.num_epochs // 5, args.num_epochs)
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    best_performance = 0
    if args.resume:
        if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint '{}'".format(args.save_dir))
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))


    print('before training, evaluate the model')
    test(args, 0, model, criterion, val_loader, val_logger, logging=False)


    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, model, criterion, train_loader, optimizer, scheduler,\
                train_logger, logging = True)
        current_performance = test(args, epoch, model, criterion, val_loader, \
                val_logger, logging = True)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'scheduler': scheduler.state_dict()}
        save_checkpoint(args, model_state, is_best, epoch, best_performance, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

        # at the end of every run, save the model
        if epoch == args.num_epochs:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%s.pth.tar' % str(args.num_epochs)))
            
    visualize_att(model,val_loader.dataset)

def save_checkpoint(args, state, is_best, epoch, best_performance, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, f'model_best_epoch{epoch}_acc{best_performance}.pth.tar'))


def train_imsitu(args, epoch, model, criterion, train_loader, optimizer, scheduler,\
    train_logger, logging=True):
    model.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset) # number of images 24301
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):

        # set mini-batch dataset
        images = images.cuda() # [16, 3, 224, 224]
        genders = genders.cuda() # [16, 2]
        targets = targets.cuda() # [16, 211]

        # forward, Backward and Optimize
        preds, _ = model(images) # preds : [16, 211]
        #preds = model(images)

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        loss = criterion(preds, targets.max(1, keepdim=False)[1]) # criterion([16,211], [16,])
        loss_logger.update(loss.item())

        preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1) # 가장 높게 예측된 ith label (softmax값)
        preds_list += preds.tolist()
        truth_list += targets.cpu().max(1, keepdim=False)[1].numpy().tolist() # 가장 높

        if batch_idx > 0 and len(preds_list) > 0:
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            total_genders = genders.cpu()

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    man_idx = total_genders[:, 0].nonzero().squeeze() # [num_man,]
    woman_idx = total_genders[:, 1].nonzero().squeeze() # [num_woman,]
    acc = accuracy_score(truth_list, preds_list)

    scheduler.step()

    if logging:
        train_logger.scalar_summary('loss', loss_logger.avg, epoch)
        train_logger.scalar_summary('acc', acc, epoch)
        train_logger.scalar_summary('lr', scheduler.get_last_lr()[0], epoch)

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Train epoch  : {}, acc: {:.2f}'.format(epoch, acc))

def test_imsitu(args, epoch, model, criterion, val_loader, val_logger, logging=True):

    # set the eval mode
    model.eval()
    nProcessed = 0
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # constrain epoch size

        # Set mini-batch dataset
        images = images.cuda()
        genders = genders.cuda()
        targets = targets.cuda()

        # Forward, Backward and Optimize
        preds, _ = model(images)
        #preds = model(images)

        loss = criterion(preds, targets.max(1, keepdim=False)[1])
        loss_logger.update(loss.item())

        preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1)
        preds_list += preds.tolist()
        truth_list += targets.cpu().max(1, keepdim=False)[1].numpy().tolist()
        if batch_idx > 0 and len(preds_list) > 0:
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            total_genders = genders.cpu()

        # print("pred max {}".format(preds))
        # print("targeta {}".format(targets.cpu().max(1, keepdim=False)[1].numpy().tolist()))


        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    # compute accuracy
    man_idx = total_genders[:, 0].nonzero().squeeze() # [4457,]
    woman_idx = total_genders[:, 1].nonzero().squeeze() # [3273,]
    acc = accuracy_score(truth_list, preds_list)

    if logging:
        val_logger.scalar_summary('loss', loss_logger.avg, epoch)
        val_logger.scalar_summary('acc', acc, epoch)


    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Val epoch  : {}, acc: {:.2f}'.format(epoch, acc))

    return acc

def train_celeba(args, epoch, model, criterion, train_loader, optimizer, scheduler,\
    train_logger, logging=True):
    model.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset) # number of images : 1999
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets) in enumerate(t):

        # set mini-batch dataset
        images = images.cuda() # [16, 3, 224, 224]
        targets = targets.cuda() # default [16, 2] : first column is for Gray_Hair, second column is for gender(man=1, woman=0)
        genders = targets[:,1].cuda()
        targets = targets[:,0].cuda() # [16]
        ## one hot encoded gender
        zero = torch.zeros(genders.size(0), 2).cuda()
        genders = zero.scatter_(1, genders.view(genders.size(0),1).long(), 1)
        # one hot encoded target
        targets = zero.scatter_(1, targets.view(targets.size(0),1).long(), 1)

        # forward, Backward and Optimize
        preds, _ = model(images) # preds : [16, 211]
        #preds = model(images)

        # compute loss and add softmax to preds (crossentropy loss integrates softmax)
        loss = criterion(preds, targets.max(1, keepdim=False)[1]) # criterion([16,211], [16,])
        loss_logger.update(loss.item())

        preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1) # 가장 높게 예측된 ith label (softmax값)
        preds_list += preds.tolist()
        truth_list += targets.cpu().max(1, keepdim=False)[1].numpy().tolist() # 가장 높

        if batch_idx > 0 and len(preds_list) > 0:
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            total_genders = genders.cpu()

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    man_idx = total_genders[:, 0].nonzero().squeeze() # [num_man,]
    woman_idx = total_genders[:, 1].nonzero().squeeze() # [num_woman,]
    acc = accuracy_score(truth_list, preds_list)

    scheduler.step()

    if logging:
        train_logger.scalar_summary('loss', loss_logger.avg, epoch)
        train_logger.scalar_summary('acc', acc, epoch)
        train_logger.scalar_summary('lr', scheduler.get_last_lr()[0], epoch)

    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Train epoch  : {}, acc: {:.2f}'.format(epoch, acc))

def test_celeba(args, epoch, model, criterion, val_loader, val_logger, logging=True):

    # set the eval mode
    model.eval()
    nProcessed = 0
    loss_logger = AverageMeter()
    preds_list, truth_list = [], []

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for batch_idx, (images, targets) in enumerate(t):

        # set mini-batch dataset
        images = images.cuda() # [16, 3, 224, 224]
        targets = targets.cuda() # default [16, 2] : first column is for Gray_Hair, second column is for gender(man=1, woman=0)
        genders = targets[:,1].cuda()
        targets = targets[:,0].cuda() # [16]
        ## one hot encoded gender
        zero = torch.zeros(genders.size(0), 2).cuda()
        genders = zero.scatter_(1, genders.view(genders.size(0),1).long(), 1)
        # one hot encoded target
        targets = zero.scatter_(1, targets.view(targets.size(0),1).long(), 1)

        # Forward, Backward and Optimize
        preds, _ = model(images)
        #preds = model(images)

        loss = criterion(preds, targets.max(1, keepdim=False)[1])
        loss_logger.update(loss.item())

        preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1)
        preds_list += preds.tolist()
        truth_list += targets.cpu().max(1, keepdim=False)[1].numpy().tolist()
        if batch_idx > 0 and len(preds_list) > 0:
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            total_genders = genders.cpu()

        # print("pred max {}".format(preds))
        # print("targeta {}".format(targets.cpu().max(1, keepdim=False)[1].numpy().tolist()))


        # Print log info
        nProcessed += len(images)
        t.set_postfix(loss = loss_logger.avg, completed = nProcessed)

    # compute accuracy
    man_idx = total_genders[:, 0].nonzero().squeeze() # [4457,]
    woman_idx = total_genders[:, 1].nonzero().squeeze() # [3273,]
    acc = accuracy_score(truth_list, preds_list)

    if logging:
        val_logger.scalar_summary('loss', loss_logger.avg, epoch)
        val_logger.scalar_summary('acc', acc, epoch)


    print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
    print('Val epoch  : {}, acc: {:.2f}'.format(epoch, acc))

    return acc



def visualize_att(model,testdata):
    denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    idxs = np.random.choice(range(len(testdata)), 5, False)
    for idx in idxs:
        img = testdata[idx][0]
        _, attentions = model(img.unsqueeze(0).cuda())
        att_mat = torch.stack(attentions).squeeze(1)
        att_mat = att_mat.cpu().detach()

        att_mat = torch.mean(att_mat, dim=1)

        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (img.shape[2], img.shape[1]))
        img = denormalize(img)
        img = np.transpose(img,(1,2,0))
        
        heatmap = cv2.applyColorMap(np.uint8(225*mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        result = heatmap + np.float32(img)
        result = result / np.max(result)
        result = np.uint8(255*result)

        mask = mask[...,np.newaxis]

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

        ax1.set_title('Original')
        ax2.set_title('Attention Mask')
        ax3.set_title('Attention Map')
        _ = ax1.imshow(img)
        _ = ax2.imshow(mask)
        _ = ax3.imshow(result)
        plt.savefig('img_{}.png'.format(idx))

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