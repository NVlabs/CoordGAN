# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# To view a copy of this license, visit
# https://github.com/NVlabs/CoordGAN/blob/main/LICENSE
#
# Written by Jiteng Mu
# -------------------------------------------------------------------------

import os
import sys
import argparse
import glob
import re
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

import imageio
import numpy as np
import cv2
from PIL import Image

import model
import loss
from utils import AverageMeter, swap, convert_to_coord_format


def cal_lpips(trg, pred, lpips):
    loss = lpips(trg, pred)
    return loss

def cal_identity(trg, pred, id_loss):
    loss, _, _ = id_loss(pred, trg, trg)
    return loss


@torch.no_grad()
def generator_samples(args, g_ema, num_pairs=100, generate_real_imgs=True, device='cuda'):

    if generate_real_imgs:
        img_paths = sorted(glob.glob(os.path.join(args.real_path, '*')))
        img_paths = img_paths[:num_pairs]
        for i in tqdm(range(len(img_paths)//2)):
            img_path1 = img_paths[i*2]
            img1 = Image.open(img_path1).convert('RGB')
            img1 = torchvision.transforms.functional.resize(img1, (args.size, args.size))
            img1 = torch.Tensor(np.array(img1)).permute(2,0,1).contiguous().unsqueeze(0)
            img1 = (img1/255-0.5)/0.5
    
            img_path2 = img_paths[i*2+1]
            img2 = Image.open(img_path2).convert('RGB')
            img2 = torchvision.transforms.functional.resize(img2, (args.size, args.size))
            img2 = torch.Tensor(np.array(img2)).permute(2,0,1).contiguous().unsqueeze(0)
            img2 = (img2/255-0.5)/0.5
    
            input_img = torch.cat((img1, img2), dim=0).to(device)
            for j in range(2):
                torchvision.utils.save_image(input_img[j, :, :, :],
                                           os.path.join(args.output_path, 'input', '%s.png' % str(i * 2 + j).zfill(5)), range=(-1, 1),
                                           normalize=True)

    # lpips
    lpips_loss = loss.LPIPS().to(device).eval()

    # arc face
    id_loss = loss.IDLoss().to(device).eval()

    swap_lpips_meter = AverageMeter()
    struc_swap_lpips_meter = AverageMeter()
    swap_id_meter = AverageMeter()
    struc_swap_id_meter = AverageMeter()

    for i in tqdm(range(num_pairs//2)):
        if args.test_arch=='coordgan':
            coords = convert_to_coord_format(2, args.size, args.size, device, integer_values=args.coords_integer_values)
            z_style = torch.randn(2, args.latent_dim, device=device)
            z_struc = torch.randn(2, args.latent_dim, device=device)
            rec_img, warp_coords, vis_coords = g_ema([z_style], [z_struc], cur_size=args.size, mode='vis', input_is_latent=False)
            swap_style = swap([z_style])
            swap_img, warp_coords, vis_coords = g_ema([swap_style], [z_struc], cur_size=args.size, mode='vis', input_is_latent=False)


        swap_lpips_score = cal_lpips(rec_img, swap_img, lpips_loss)
        swap_lpips_meter.update(swap_lpips_score)
        struc_swap_lpips_score = cal_lpips(rec_img, torch.flip(swap_img, dims=(0,)), lpips_loss)
        struc_swap_lpips_meter.update(struc_swap_lpips_score)

        if args.dataset=='celebA':
            swap_id_score = cal_identity(rec_img, swap_img, id_loss)
            swap_id_meter.update(swap_id_score)
            struc_swap_id_score = cal_identity(rec_img, torch.flip(swap_img, dims=(0,)), id_loss)
            struc_swap_id_meter.update(struc_swap_id_score)

        # save images
        for j in range(2):
            torchvision.utils.save_image(rec_img[j, :, :, :],
                                       os.path.join(args.output_path, 'samples', '%s.png' % str(i * 2 + j).zfill(5)), range=(-1, 1),
                                       normalize=True)

            torchvision.utils.save_image(swap_img[j, :, :, :],
                                       os.path.join(args.output_path, 'samples_swap', '%s_swap.png' % str(i * 2 + j).zfill(5)), range=(-1, 1),
                                       normalize=True)
    print('swap_lpips:', swap_lpips_meter.avg)
    print('struc_swap_lpips:', struc_swap_lpips_meter.avg)
    if args.dataset=='celebA':
        print('swap_id:', swap_id_meter.avg)
        print('struc_swap_id:', struc_swap_id_meter.avg)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    # test
    parser.add_argument('--ckpt', type=str, default=None)

    # our Generator params
    parser.add_argument('--Generator', type=str, default='CoordGAN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--n_mlp', type=int, default=8)

    # dataset
    parser.add_argument('--dataset', type=str, default='celebA')
    parser.add_argument('--num_workers', type=int, default=32)

    # fid
    parser.add_argument('--fid_sample', type=int, default=10000)
    parser.add_argument('--test_arch', type=str, default='coordgan')

    args = parser.parse_args()

    # define architecture
    if args.test_arch=='coordgan':

        path = re.split('/', args.ckpt)[:-1]
        iters = re.split('/', args.ckpt)[-1][:-3]
        args.output_path = os.path.join(*path, iters)
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'input'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'samples_swap'), exist_ok=True)
        print('output dir', args.output_path)

        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt)

        # load generator
        Generator = getattr(model, args.Generator)
        print('Generator', Generator)
        generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent_dim, n_mlp=args.n_mlp,
                        channel_multiplier=args.channel_multiplier
                        ).to(device)
        generator.load_state_dict(ckpt['g_ema'], strict=False)
        generator.eval()

    else:
        raise Exception ('not implemented')

    # load input images / fid images
    if args.dataset=='stanfordcar': 
        args.real_path = '/data/jiteng/raid/cars_train/'
    if args.dataset=='afhq-cat':
        args.real_path = '/data/jiteng/raid/afhq/train/cat'
    if args.dataset=='celebA':
        args.real_path = '/data/jiteng/raid/CelebAMask-HQ/CelebA-HQ-img/'

    generator_samples(args, generator, num_pairs=args.fid_sample)

