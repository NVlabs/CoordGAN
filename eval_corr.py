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

import argparse
import math
import random
import os
import re
import glob
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
#from torch.utils import data
import torch.distributed as dist
from torchvision import transforms
import torchvision
from torchvision import utils as torchvision_utils

import model
from utils import convert_to_coord_format, AverageMeter, swap
from seg_utils import load_file_datasetgan, load_file_CelebAHQMask, inter_and_union, seg2img




def seg_transfer(args, inp_img, style_enc, struc_enc, \
                        g_ema, inp_seg, device, num_cls, i):

    inp_seg_argmax = torch.argmax(inp_seg, dim=1).unsqueeze(1).repeat(1,3,1,1)
    inp_seg_argmax[inp_seg_argmax==17]=0
    inp_seg_argmax[inp_seg_argmax==25]=0

    z_style = style_enc
    z_struc = struc_enc

    real, warp, vis = g_ema([z_style], [z_struc], mode='vis', input_is_latent=args.enc_input_is_latent)

    if real.shape[-1]!=vis.shape[-1]:
        vis = F.interpolate(vis, [real.shape[-2], real.shape[-1]])
    if inp_img.shape[-1]!=vis.shape[-1]:
        inp_img = F.interpolate(inp_img, [real.shape[-2], real.shape[-1]])

    sample = real.unsqueeze(1)
    vis = vis.unsqueeze(1)
    sample = torch.cat((vis, sample), dim=1)
    sample = torch.cat((inp_img.unsqueeze(1), sample), dim=1)


    vis_swap=True
    if vis_swap==True:
        swap_style = swap([z_style])
        swap_real, swap_warp, swap_vis = g_ema([swap_style], [z_struc], \
                                        mode='vis', input_is_latent=args.enc_input_is_latent)
        if real.shape[-1]!=swap_vis.shape[-1]:
            swap_vis = F.interpolate(swap_vis, [real.shape[-2], real.shape[-1]])
        sample = torch.cat((sample, swap_vis.unsqueeze(1)), dim=1)
        sample = torch.cat((sample, swap_real.unsqueeze(1)), dim=1)

    seg_transfer = True
    if seg_transfer==True:
        seg1, seg2, seg12, seg21, inter, union = get_seg_transfer_img(real, warp, inp_seg)
        _,H,W = seg1.shape
        real_12_seg = torch.cat((seg1.unsqueeze(1), seg21.unsqueeze(1)), dim=1 ).view(-1,H,W).long()
        real_21_seg = torch.cat((seg12.unsqueeze(1), seg2.unsqueeze(1)), dim=1 ).view(-1,H,W).long()
        real_12_seg = seg2img(args, real_12_seg.cpu(), num_cls)/255*2-1
        real_21_seg = seg2img(args, real_21_seg.cpu(), num_cls)/255*2-1
        real_12_seg = torch.from_numpy(real_12_seg).permute(0,3,1,2).contiguous().to(device)
        real_21_seg = torch.from_numpy(real_21_seg).permute(0,3,1,2).contiguous().to(device)
        sample = torch.cat((sample, real_12_seg.unsqueeze(1)), dim=1)
        sample = torch.cat((sample, real_21_seg.unsqueeze(1)), dim=1)

    sample = sample.view(-1, 3, args.size, args.size)

    return sample, inter, union


def get_seg_transfer_img(real, warp, inp_seg=None, mode='soft', temp=0.01):

    if inp_seg is None:
        pred_logits = real.clone()
        pred = real
        num_cls = pred_logits.shape[1]
    else:
        pred_logits = inp_seg
        pred = torch.argmax(pred_logits, dim=1)
        num_cls = pred_logits.shape[1]

    assert warp.shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [warp.shape[0] // 2, 2] + list(warp.shape[1:])
    warp = warp.view(*new_shape)

    new_shape = [pred.shape[0] // 2, 2] + list(pred.shape[1:])
    pred = pred.view(*new_shape)

    new_shape = [pred_logits.shape[0] // 2, 2] + list(pred_logits.shape[1:])
    pred_logits = pred_logits.view(*new_shape) #(B//2, 2, C, H, W)
    B,N,C,H,W = pred_logits.shape
    pred_logits = pred_logits.permute(0,1,3,4,2).contiguous().view(B,N,-1,C)

    B,N,C,H,W = warp.shape
    warp = warp.permute(0,1,3,4,2).contiguous()\
                   .view(B,N,-1,C)
    diff = warp[:,0].unsqueeze(2) - warp[:,1].unsqueeze(1)
    diff = torch.norm(diff, dim=3)

    if mode=='soft':
        aff12 = F.softmax(- diff / temp, dim = 2)
        aff21 = F.softmax(- diff / temp, dim = 1).permute(0,2,1)
        out12 = torch.einsum('bik,bkj->bij', aff12, pred_logits[:,1].contiguous())
        out21 = torch.einsum('bik,bkj->bij', aff21, pred_logits[:,0].contiguous())

    elif mode=='hard':
        cor12 = torch.argmin(diff, dim=2) # B//2, H*W
        cor21 = torch.argmin(diff, dim=1) # B//2, H*W
        out12 = pred_logits[:,1].reshape(-1, num_cls)[cor12.view(-1)] 
        out21 = pred_logits[:,0].reshape(-1, num_cls)[cor21.view(-1)] 
        
    out12 = out12.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()
    out21 = out21.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()

    if inp_seg is None:
        pred1 = pred[:,0] 
        pred2 = pred[:,1] 
        return pred1, pred2, out12, out21
    else:
        out12 = torch.argmax(out12, dim=1)
        out21 = torch.argmax(out21, dim=1)
        pred1 = pred[:,0] 
        pred2 = pred[:,1] 
        inter2, union2 = inter_and_union(out21.cpu().numpy(), pred2.cpu().numpy(), num_cls)
        return pred1, pred2, out12, out21, inter2, union2



def test_seg_transfer(args, data, g_ema, encoder, num_cls, device):

    with torch.no_grad():
        # inputs
        inp_img = data[0][0]
        inp_seg = data[0][1]
        for img, seg in data[1:]: 
            inp_img = torch.cat((inp_img,img), dim=0)
            inp_seg = torch.cat((inp_seg,seg), dim=0)
        inp_img = inp_img.to(device)
        inp_seg = inp_seg.to(device)

        mean_iou = []
        test_idx = list(np.arange(args.num_ref_img, len(data)))
        for base_idx in tqdm(range(args.num_ref_img)):

            inter_meter = AverageMeter()
            union_meter = AverageMeter()

            # test pairs
            for i in test_idx:
                img = inp_img[[base_idx, i]]
                seg = inp_seg[[base_idx, i]]

                struc_enc, style_enc, _ = encoder(img)
 
                sample, inter, union = seg_transfer(args, img, style_enc, struc_enc, \
                                                        g_ema, seg, device, num_cls, i)

                inter_meter.update(inter)
                union_meter.update(union)
    
                torchvision_utils.save_image(
                    sample,
                    os.path.join(args.output_dir, f'{str(i).zfill(6)}_{str(base_idx)}_{str(i)}.png'),
                    nrow=int(sample.shape[0]/2),
                    normalize=True,
                    range=(-1, 1),
                )

            iou = inter_meter.sum / (union_meter.sum + 1e-10)
    
            if args.segdataset=='datasetgan-face-34':
                iou_no_hair = list(iou)
                iou_no_hair.pop(16)
                iou_no_hair.pop(23)
                iou_no_hair = np.array(iou_no_hair)
                mean_iou.append(iou_no_hair)
                print('Reference image {} exclude hair/neck IoU: {}'.format(base_idx, iou_no_hair.mean() * 100))
            else:
                mean_iou.append(iou)
                print('Reference image {} IoU: {}'.format(base_idx, iou.mean() * 100))
        print('Final IOU: ', np.mean(mean_iou) * 100)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    # Encoder params
    parser.add_argument('--Encoder', type=str, default='Encoder')
    parser.add_argument('--enc_input_is_latent', type=str2bool, default=True)

    # Generator params
    parser.add_argument('--Generator', type=str, default='CoordGAN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--n_mlp', type=int, default=8)

    # dataset
    parser.add_argument('--segdataset', type=str, default='datasetgan-face-34')
    parser.add_argument('--num_ref_img', type=int, default=5)
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    # make dirs
    path = re.split('/', args.ckpt)[:-2]
    iters = re.split('/', args.ckpt)[-1][:-3]
    args.output_dir = os.path.join(*path, 'seg_transfer_mask_enc', iters)
    os.makedirs(args.output_dir, exist_ok=True)
    print('save label transfer results at: ', args.output_dir)

    # load model
    Encoder = getattr(model, args.Encoder)
    Generator = getattr(model, args.Generator)

    encoder = Encoder(args).to(device)
    g_ema = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, channel_multiplier=args.channel_multiplier,
                      ).to(device)

    if args.ckpt is not None:
        print('load checkpoint: ', args.ckpt)
        ckpt = torch.load(args.ckpt)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        encoder.load_state_dict(ckpt['e'], strict=False)
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)
    encoder.eval()
    g_ema.eval()

    # load data
    if args.segdataset=='datasetgan-face-34':
        num_cls = 34
        base_path = 'data/datasetgan'
        data = load_file_datasetgan(args, base_path, num_cls)
    elif args.segdataset=='datasetgan-car-20':
        num_cls = 20
        base_path = 'data/datasetgan'
        data = load_file_datasetgan(args, base_path, num_cls)
    elif args.segdataset=='celebA-7':
        base_path = 'data/CelebAMask-HQ'
        num_cls = 7
        data = load_file_CelebAHQMask(base_path, num_cls)

    test_seg_transfer(args, data, g_ema, encoder, num_cls, device)

