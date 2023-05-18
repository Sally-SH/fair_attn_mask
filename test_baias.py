from email.policy import default
import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm
from einops import reduce

import torch, torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms


from data_loader import ImSituVerbGender
from model import VisionTransformer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=str,
			help='path for saving checkpoints')
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

	parser.add_argument('--learning_rate', type=float, default=3e-2)
	parser.add_argument('--finetune', action='store_true')
	parser.add_argument('--num_epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=32)

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

	# Data loaders / batch assemblers.
	val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
			shuffle = False, num_workers = 4, pin_memory = True)

	# build the models
	channel = 3
	patch_size = args.patch_size
	d_model = 64
	n_layers = 12
	n_head = 8
	ff_dim = 256
	dropout_rate = 0.1
	output_dim = 2
	img_size = (val_loader.dataset[0][0].shape[0], val_loader.dataset[0][0].shape[1])

	model = VisionTransformer(channel, img_size, patch_size, d_model, n_layers, n_head, ff_dim, dropout_rate, output_dim)
	model = model.cuda()
	model.eval()

	if os.path.isfile(os.path.join('./models', args.save_dir, 'model_best.pth.tar')):
		print("=> loading checkpoint '{}'".format(args.save_dir))
		checkpoint = torch.load(os.path.join('./models', args.save_dir, 'model_best.pth.tar'))
		args.start_epoch = checkpoint['epoch']
		best_performance = checkpoint['best_performance']
		model.load_state_dict(checkpoint['state_dict'])
		print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
		print("best performance : {}".format(best_performance))
	else:
		print("=> no checkpoint found at '{}'".format(args.save_dir))
		return None

	visualize_att(args, model,val_loader.dataset)

def visualize_att(args, model,testdata):
    denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    idxs = np.random.choice(range(len(testdata)), 5, False)
    genders = ['male', 'female']
    for idx in idxs:
        img = testdata[idx][0]
        label = np.argmax(testdata[idx][2].cpu().detach().numpy())
        pred, attentions = model(img.unsqueeze(0).cuda())
        pred = np.argmax(F.softmax(pred, dim=1).cpu().detach().numpy(), axis=1)[0]
        
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

        img = denormalize(img)
        img = np.transpose(img,(1,2,0))

        if args.mask_mode == 'pixel':
            mask = cv2.resize(mask / mask.max(), (img.shape[1], img.shape[0]))
            flatten_mask = mask.reshape(-1)
            flatten_mask = np.sort(flatten_mask)
			# select the top mask_ratio% attention value
            mask_val = flatten_mask[int((1-args.mask_ratio*0.01)*len(flatten_mask))]
            mask = mask[...,np.newaxis]
            # mask top mask_ratio% attention area
            masked_img = np.where(np.repeat(mask, 3, axis=2)>mask_val, 0, img)
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

        heatmap = cv2.applyColorMap(np.uint8(225*mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        result = heatmap + np.float32(img)
        result = result / np.max(result)
        result = np.uint8(255*result)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(12, 12))

        ax1.set_title('Original')
        ax2.set_title('Attention Mask')
        ax3.set_title('Attention Map')
        ax4.set_title('Masked image')
        _ = ax1.imshow(img)
        _ = ax2.imshow(mask)
        _ = ax3.imshow(result)
        _ = ax4.imshow(masked_img)
        plt.savefig('img_{}.png'.format(idx))
        
        print("idx {}\n pred : {} // ground : {}".format(idx,genders[pred],genders[label]))
        
def test(model, val_loader):

	# set the eval mode
	model.eval()
	nProcessed = 0
	nVal = len(val_loader.dataset) # number of images
	preds_list, truth_list = [], []

	t = tqdm(val_loader)
	for batch_idx, (images, targets, genders, image_ids) in enumerate(t):

		# Set mini-batch dataset
		images = images.cuda()
		genders = genders.cuda()

		# Forward, Backward and Optimize
		preds, _ = model(images)

		preds = np.argmax(F.softmax(preds, dim=1).cpu().detach().numpy(), axis=1)
		preds_list += preds.tolist()
		truth_list += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()
		if batch_idx > 0 and len(preds_list) > 0:
			total_genders = torch.cat((total_genders, genders.cpu()), 0)
		else:
			total_genders = genders.cpu()

		# Print log info
		nProcessed += len(images)

	# compute accuracy
	man_idx = total_genders[:, 0].nonzero().squeeze()
	woman_idx = total_genders[:, 1].nonzero().squeeze()
	acc = accuracy_score(truth_list, preds_list)


	print('man size: {} woman size: {}'.format(len(man_idx), len(woman_idx)))
	print('acc: {:.2f}'.format(acc))

	return acc

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

