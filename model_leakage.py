import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, DistributedSampler

import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse, operator, collections
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from tqdm import tqdm as tqdm

from data_loader import ImSituVerbGender, ImSituVerbGenderFeature
from models.classifier import GenderClassifier
import util.misc as utils
from engine import evaluate_swig, train_one_epoch
from models import build_model

####### data preparation #########################
verb_id_map = pickle.load(open('./data/verb_id.map', 'rb'))
verb2id = verb_id_map['verb2id']
id2verb = verb_id_map['id2verb']


def generate_image_feature(split, image_features_path, data_loader, encoder):

    targets = list()
    genders = list()
    image_ids = list()
    potentials = list()
    
    t = tqdm(data_loader)

    for ind, (images_, targets_, genders_, image_ids_) in enumerate(t):
        images_ = images_.cuda()
        preds, _ = encoder(images_)
        potentials.append(preds.detach().cpu())
        targets.append(targets_.cpu())
        genders.append(genders_.cpu())
        image_ids.append(image_ids_.cpu())

    targets = torch.cat(targets, 0)
    genders = torch.cat(genders, 0)
    image_ids = torch.cat(image_ids, 0)
    potentials = torch.cat(potentials, 0)

    torch.save(targets, os.path.join(image_features_path, '{}_targets.pth'.format(split)))
    torch.save(genders, os.path.join(image_features_path, '{}_genders.pth'.format(split)))
    torch.save(image_ids, os.path.join(image_features_path, '{}_image_ids.pth'.format(split)))
    torch.save(potentials, os.path.join(image_features_path, '{}_potentials.pth'.format(split)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', default='./vanila/checkpotints.pth',
                        help='path where saved model is')
    parser.add_argument('--num_rounds', type=int,
            default = 2)

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

    ## training setting for attacker
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
            help='attacker learning rate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hid_size', type=int, default=300,
            help='linear layer dimension for attacker')
    parser.add_argument('--attacker_capacity', type=int, default=300,
            help='linear layer dimension for attacker')
    parser.add_argument('--attacker_dropout', type=float, default=0.2,
            help='parameter for dropout layter in attacker')

    # Etc..
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', 
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='leakage',
                        help='path where to save, empty for no saving')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--inference', default=True)
    parser.add_argument('--fair', default=False)

    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    args = parser.parse_args()

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    
    device = torch.device(args.device)

    #Build the encoder from adv model
    encoder, criterion = build_model(args)
    encoder.to(device)
    model_path = args.saved_dir
    
    if os.path.isfile(model_path):
        print("=> loading encoder from '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    encoder.eval()

    val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'val', transform = test_transform)
    data_loader_val = DataLoader(val_data, batch_size = args.batch_size, \
            shuffle = False, num_workers = 4,pin_memory = True)

    test_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'test', transform = test_transform)
    data_loader_test = DataLoader(test_data, batch_size = args.batch_size, \
            shuffle = False, num_workers = 4,pin_memory = True)

    output_dir = os.path.join(os.path.split(model_path)[0],args.output_dir)
    os.makedirs(output_dir,exist_ok=True)
    log_path = os.path.join(output_dir,"log.txt")

    print('val set performance:')
    val_stats = evaluate_swig(encoder, criterion, data_loader_val, device, args.output_dir)
    val_log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}
    if args.output_dir and utils.is_main_process():
            with open(log_path, "a") as f:
                f.write(json.dumps(val_log_stats) + "\n")

    print('test set performance:')
    test_stats = evaluate_swig(encoder, criterion, data_loader_test, device, args.output_dir)
    test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    if args.output_dir and utils.is_main_process():
            with open(log_path, "a") as f:
                f.write(json.dumps(test_log_stats) + "\n")

    acc_list = {}
    acc_list['potential'] = []

    args.gender_balanced = True
    for i in range(args.num_rounds):

        train_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir,split = 'train', transform = train_transform)
        data_loader_train = DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        # Data samplers for val set.
        val_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'val', transform = test_transform)
        data_loader_val = DataLoader(val_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)
        
        # Data samplers for test set.
        test_data = ImSituVerbGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'test', transform = test_transform)
        data_loader_test = DataLoader(test_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)
        
        image_features_path = os.path.join(output_dir, 'image_features')
        if not os.path.exists(image_features_path):
            os.makedirs(image_features_path)

        # get image features from encoder
        generate_image_feature('train', image_features_path, data_loader_train, encoder)
        generate_image_feature('val', image_features_path, data_loader_val, encoder)
        generate_image_feature('test', image_features_path, data_loader_test, encoder)

        train_data = ImSituVerbGenderFeature(args, image_features_path, split = 'train')
        data_loader_train = DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)
        
        val_data = ImSituVerbGenderFeature(args, image_features_path, split = 'val')
        data_loader_val = DataLoader(val_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)
        
        test_data = ImSituVerbGenderFeature(args, image_features_path, split = 'test')
        data_loader_test = DataLoader(test_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)
        

        model_save_dir = output_dir

        for feature_type in acc_list.keys():

            #import pdb
            #pdb.set_trace()

            attacker = GenderClassifier(args, args.num_verb)

            attacker = attacker.cuda()

            optimizer = optim.Adam(attacker.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

            train_attacker(args.num_epochs, optimizer, attacker, encoder, data_loader_train, data_loader_val, \
               model_save_dir, feature_type)

            # evaluate best attacker on balanced test split
            best_attacker = torch.load(model_save_dir + '/best_attacker.pth.tar')
            attacker.load_state_dict(best_attacker['state_dict'])
            _, val_acc = epoch_pass(0, data_loader_val, attacker, encoder, None, False, feature_type)
            val_acc = 0.5 + abs(val_acc - 0.5)
            _, test_acc = epoch_pass(0, data_loader_test, attacker, encoder, None, False, feature_type)
            test_acc = 0.5 + abs(test_acc - 0.5)
            acc_list[feature_type].append(test_acc)
            print('round {} feature type: {}, test acc: {}, val acc: {}'.format(i, feature_type, test_acc, val_acc))

    for feature_type in acc_list.keys():
        print(acc_list[feature_type], np.std(np.array(acc_list[feature_type])))
        print('{} average leakage: {}'.format(feature_type, np.mean(np.array(acc_list[feature_type]))))
        if args.output_dir and utils.is_main_process():
            with open(log_path, "a") as f:
                f.write( '{} average leakage: {}\n'.format(feature_type, np.mean(np.array(acc_list[feature_type]))))

def train_attacker(num_epochs, optimizer, attacker, encoder, train_loader, test_loader, model_save_dir, feature_type, print_every=500):
    # training setting
    encoder.eval()
    attacker.train()

    train_acc_arr, train_loss_arr = [], []
    dev_acc_arr, dev_loss_arr = [], []
    best_model_epoch = 1
    best_score = 0.0

    for epoch in range(1, num_epochs + 1):

        # train
        loss, train_task_acc = epoch_pass(epoch, train_loader, attacker, encoder, optimizer, True, feature_type, print_every)
        train_acc_arr.append(train_task_acc)
        train_loss_arr.append(loss)
        if epoch % 10 == 0:
          print('train, {0}, adv acc: {1:.2f}'.format(epoch, train_task_acc*100))

        # dev
        loss, dev_task_acc = epoch_pass(epoch, test_loader, attacker, encoder, optimizer, False, feature_type, print_every)
        dev_acc_arr.append(dev_task_acc)
        dev_loss_arr.append(loss)
        if epoch % 10 == 0:
          print('dev, {0}, adv acc: {1:.2f}'.format(epoch, dev_task_acc*100))

        if dev_task_acc > best_score:
            best_score = dev_task_acc
            best_model_epoch = epoch
            torch.save({'epoch': epoch, 'state_dict': attacker.state_dict()}, model_save_dir + '/best_attacker.pth.tar')

        if epoch % 10 == 0:
          print('current best dev score: {:.2f}'.format(best_score*100))


def epoch_pass(epoch, data_loader, attacker, encoder, optimizer, training, feature_type, print_every=500):

    t_loss = 0.0
    preds, truth = [], []
    n_processed = 0

    if training:
        attacker.train()
    else:
        attacker.eval()

    t = tqdm(data_loader)

    for ind, (targets, genders, image_ids, potentials) in enumerate(t):

        features = potentials.float().cuda()
        adv_pred = attacker(features)
        loss = F.cross_entropy(adv_pred, genders.cuda().max(1, keepdim=False)[1], reduction='mean')

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        preds += adv_pred.tolist()
        truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(targets)

        acc_score = accuracy_score(truth, preds)

    return t_loss / n_processed, acc_score

if __name__ == '__main__':
    main()