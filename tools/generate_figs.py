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
sys.path.append('../')
import math

import torch
import torch.nn as nn
import scipy.misc
from collections import OrderedDict
import os
import re
import glob
import pickle
import json

from PIL import Image
import numpy as np

import torch.optim as optim
import argparse
import imageio

import model
import loss
import torchvision
from torch.nn import functional as F

from torchvision import transforms
from tqdm import tqdm



@torch.no_grad()
def generator_figs(args, g_ema, num_pairs=10, device='cuda'):

    torch.manual_seed(args.seed)
    all_style = torch.randn(num_pairs, args.latent, device=device)
    all_struc = torch.randn(num_pairs, args.latent, device=device)
    for struc_id in tqdm(range(num_pairs)):
        for style_id in tqdm(range(num_pairs)):
            z_style = all_style[style_id].unsqueeze(0)
            z_struc = all_struc[struc_id].unsqueeze(0)
            if args.test_arch=='coordgan':
                rec_img, warp_coords, vis_coords = g_ema([z_style], [z_struc], mode='vis', input_is_latent=False)

                # save images
                torchvision.utils.save_image(rec_img,
                                            os.path.join(args.output_path, 'figs_samples', 'struc%s_style%s.png' % (str(struc_id).zfill(2), str(style_id).zfill(2))), range=(-1, 1),
                                            normalize=True)

            if args.test_arch=='DiagonalGAN':
                step = int(math.log(args.size, 2)) - 2

                z_style = g_ema.style(z_style)
                z_struc = g_ema.pix(z_struc)

                source_pix = []
                source_style = []
                for ip in range(18):
                    source_pix.append(z_struc)
                    source_style.append(z_style)

                rec_img = generator.generator(
                        source_style, source_pix,step=step,alpha=1,eval_mode=True)

                # save images
                torchvision.utils.save_image(rec_img,
                                            os.path.join(args.output_path, 'figs_samples', 'struc%s_style%s.png' % (str(struc_id).zfill(2), str(style_id).zfill(2))), range=(-1, 1),
                                            normalize=True)

        if args.test_arch=='coordgan':
            # save images
            vis_coords = F.interpolate(vis_coords, [512,512], mode='bilinear')
            torchvision.utils.save_image(vis_coords,
                                        os.path.join(args.output_path, 'figs_coords', 'struc%s.png' % str(struc_id).zfill(2)), range=(-1, 1),
                                        normalize=True)

            # img_path = os.path.join(args.output_path, 'figs_samples', 'struc%s_style%s.png' % (str(struc_id).zfill(2), str(style_id).zfill(2))) 
            # coord_mask = get_coord_masks(args, rec_img, vis_coords)
            # torchvision.utils.save_image(coord_mask,
            #                             os.path.join(args.output_path, 'coords', 'struc%s_mask.png' % str(struc_id).zfill(2)), range=(-1, 1),
            #                             normalize=True)

def get_coord_masks(img, coord, image_path=None):

    if args.dataset=='celebA':
        num_cls = 34
        seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                                num_classes=num_cls, aux_loss=None).to(device)
        seg_model.load_state_dict(torch.load('datasetgan_checkpoints/model/face_512/our.pth')['model_state_dict'])
        seg_model.eval()


    elif args.dataset=='stanfordcar':
        num_cls = 20
        seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                                num_classes=num_cls, aux_loss=None).to(device)
        seg_model.load_state_dict(torch.load('datasetgan_checkpoints/model/car/our.pth')['model_state_dict'])
        seg_model.eval()

    elif args.dataset=='afhq-cat':

        # facebook
        # Some basic setup:
        # Setup detectron2 logger
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()
        
        # import some common libraries
        import numpy as np
        import os, json, cv2, random
        
        # import some common detectron2 utilities
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog

        img = cv2.imread(img_path)

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(img)

        mask = outputs["instances"].pred_masks
        c = outputs["instances"].pred_classes
        c_idx = torch.where(c==15)
        mask = mask[c_idx]
        mask = mask.unsqueeze(1).repeat(1,3,1,1)


    pred_logits = seg_model(img.to(device))["out"]
    seg_mask = torch.argmax(pred_logits, dim=1).cpu().unsqueeze(1).repeat(1,3,1,1)
    seg_mask[seg_mask==17]=0
    seg_mask[seg_mask==25]=0

    seg_mask[seg_mask==0] = -1
    seg_mask[seg_mask>0] = 1
    seg_mask = seg_mask.float()
    coord_mask = coord.clone()
    coord_mask = (coord_mask + 1 )/2
    coord_mask[seg_mask<0] = 0.3*coord_mask[seg_mask<0]
    coord_mask = coord_mask*2-1

    return coord_mask


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()

    # test
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)

    # our Generator params
    parser.add_argument('--Generator', type=str, default='CoordGAN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--mixing', type=float, default=0.)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--Encoder', type=str, default='Encoder')

    parser.add_argument('--dataset', type=str, default='celebA')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--test_arch', type=str, default='coordgan')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.test_arch=='coordgan':
        path = re.split('/', args.ckpt)[:-1]
        iters = re.split('/', args.ckpt)[-1][:-3]
        print(path, iters)
        args.output_path = os.path.join(*path, iters)
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'figs_samples'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'figs_coords'), exist_ok=True)
        print('output dir', args.output_path)

        # load generator
        Generator = getattr(model, args.Generator)
        print('Generator', Generator)
        generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                        activation=args.activation, channel_multiplier=args.channel_multiplier,
                        ).to(device)

        if args.ckpt is not None:
            print('load model:', args.ckpt)
            ckpt = torch.load(args.ckpt)
            generator.load_state_dict(ckpt['g_ema'], strict=False)
            generator.eval()


#    elif args.test_arch=='DiagonalGAN':
#
#        iters = re.split('/', args.ckpt)[-1][:-6]
#        args.output_path = os.path.join(path, iters)
#        os.makedirs(args.output_path, exist_ok=True)
#        os.makedirs(os.path.join(args.output_path, 'samples'), exist_ok=True)
#        print('output dir', args.output_path)
# 
#        from baseline.DiagonalGAN.model import StyledGenerator
#        generator = StyledGenerator(512).to(device)
#        generator.load_state_dict(torch.load(args.ckpt)['g_running'])
#        generator.eval().cuda()

    else:
        raise Exception ('not implemented')

    generator_figs(args, generator, args.num_samples)

