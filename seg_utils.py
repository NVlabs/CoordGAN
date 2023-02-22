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

import glob
import os

import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from PIL import Image

from tqdm import tqdm

############### plot utils #################

face_class = ['background', 'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix',
              'head***ear***lobule', 'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris',
              'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
              'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
              'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
              'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
              'head***philtrum', 'head***temple', 'head***wrinkles']

car_20_class = ['background', 'back_bumper', 'bumper', 'car_body', 'car_lights', 'door', 'fender','grilles','handles',
                'hoods', 'licensePlate', 'mirror','roof', 'running_boards', 'tailLight','tire', 'trunk_lids','wheelhub', 'window', 'windshield']


celebA_7_class = ['background', 'mouth', 'eye', 'brow', 'ear', 'nose', 'face']


def seg2img(args, seg, num_cls):

    face_palette = [  1.0000,  1.0000 , 1.0000,
                  0.4420,  0.5100 , 0.4234,
                  0.8562,  0.9537 , 0.3188,
                  0.2405,  0.4699 , 0.9918,
                  0.8434,  0.9329  ,0.7544,
                  0.3748,  0.7917 , 0.3256,
                  0.0190,  0.4943 , 0.3782,
                  0.7461 , 0.0137 , 0.5684,
                  0.1644,  0.2402 , 0.7324,
                  0.0200 , 0.4379 , 0.4100,
                  0.5853 , 0.8880 , 0.6137,
                  0.7991 , 0.9132 , 0.9720,
                  0.6816 , 0.6237  ,0.8562,
                  0.9981 , 0.4692 , 0.3849,
                  0.5351 , 0.8242 , 0.2731,
                  0.1747 , 0.3626 , 0.8345,
                  0.5323 , 0.6668 , 0.4922,
                  0.2122 , 0.3483 , 0.4707,
                  0.6844,  0.1238 , 0.1452,
                  0.3882 , 0.4664 , 0.1003,
                  0.2296,  0.0401 , 0.3030,
                  0.5751 , 0.5467 , 0.9835,
                  0.1308 , 0.9628,  0.0777,
                  0.2849  ,0.1846 , 0.2625,
                  0.9764 , 0.9420 , 0.6628,
                  0.3893 , 0.4456 , 0.6433,
                  0.8705 , 0.3957 , 0.0963,
                  0.6117 , 0.9702 , 0.0247,
                  0.3668 , 0.6694 , 0.3117,
                  0.6451 , 0.7302,  0.9542,
                  0.6171 , 0.1097,  0.9053,
                  0.3377 , 0.4950,  0.7284,
                  0.1655,  0.9254,  0.6557,
                  0.9450  ,0.6721,  0.6162]

    face_palette = [int(item * 255) for item in face_palette]

    car_20_palette =[ 255,  255,  255, # 0 background
      238,  229,  102,# 1 back_bumper
      0, 0, 0,# 2 bumper
      124,  99 , 34, # 3 car
      193 , 127,  15,# 4 car_lights
      248  ,213 , 42, # 5 door
      220  ,147 , 77, # 6 fender
      99 , 83  , 3, # 7 grilles
      116 , 116 , 138,  # 8 handles
      200  ,226 , 37, # 9 hoods
      225 , 184 , 161, # 10 licensePlate
      142 , 172  ,248, # 11 mirror
      153 , 112 , 146, # 12 roof
      38  ,112 , 254, # 13 running_boards
      229 , 30  ,141, # 14 tailLight
      52 , 83  ,84, # 15 tire
      194 , 87 , 125, # 16 trunk_lids
      225,  96  ,18,  # 17 wheelhub
      31 , 102 , 211, # 18 window
      104 , 131 , 101# 19 windshield
             ]


    celebA_7_palette =[ 255,  255,  255, # 0 background
        200,  226,  37,# 1 back_bumper
        31 , 102 , 211,# 2 bumper
        124,  99 , 34, # 3 car
        193 , 127,  15,# 4 car_lights
        248  ,213 , 42, # 5 door
    #       104  ,131 , 101, # 6 fender
        153, 112, 146,
                    ]


    if args.segdataset=='datasetgan-face-34':
        palette = face_palette
    if args.segdataset=='datasetgan-car-20':
        palette = car_20_palette
    if args.segdataset=='celebA-7':
        palette = celebA_7_palette
    img = np.array([ colorize_mask(s.numpy(), palette) for s in seg ])
    return img


def inter_and_union(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class-1, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class-1, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class-1, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return (area_inter, area_union)

def seg2onehot(seg, num_cls):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    mask = [seg == i for i in range(num_cls)]
    return np.array(mask).astype(np.uint8)

def onehot2seg(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    seg = np.argmax(mask, axis=0).astype(np.uint8)
    return seg

def colorize_mask(mask, palette):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


############### load datasetgan data #################

def load_file_datasetgan(args, base_path, num_cls, transfer_mask=False, img_dim=128, seg_dim=128):

    data = []

    if args.segdataset=='datasetgan-face-34':
        path_train = os.path.join(base_path, 'our_train/face_processed/')
        path_test = os.path.join(base_path, 'our_test/face_34_class/')

    if args.segdataset=='datasetgan-car-20':
        path_train = os.path.join(base_path, 'our_train/car_processed/')
        path_test = os.path.join(base_path, 'our_test/car_20_class/')

    img_paths_ref = sorted(glob.glob(os.path.join(path_train, '*.jpg'))) \
                + sorted(glob.glob(os.path.join(path_train, '*.png')))
    seg_paths_ref = sorted(glob.glob(os.path.join(path_train, '*.npy')))
    img_paths = sorted(glob.glob(os.path.join(path_test, '*.jpg'))) \
                + sorted(glob.glob(os.path.join(path_test, '*.png')))
    seg_paths = sorted(glob.glob(os.path.join(path_test, '*.npy')))

    for i, (img_path, seg_path) in enumerate(zip(img_paths_ref, seg_paths_ref)):
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.functional.resize(img, (img_dim, img_dim))
        img = torch.Tensor(np.array(img)).permute(2,0,1).contiguous().unsqueeze(0)
        img = (img/255-0.5)/0.5
        seg = np.load(seg_path)
    
        seg = torch.Tensor(seg2onehot(seg, num_cls)).unsqueeze(0)
        seg = F.interpolate(seg, [seg_dim, seg_dim], mode='bilinear')
        data.append([img, seg])

    for i, (img_path, seg_path) in enumerate(zip(img_paths, seg_paths)):
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.functional.resize(img, (img_dim, img_dim))
        img = torch.Tensor(np.array(img)).permute(2,0,1).contiguous().unsqueeze(0)
        img = (img/255-0.5)/0.5
        seg = np.load(seg_path)

        seg = torch.Tensor(seg2onehot(seg, num_cls)).unsqueeze(0)
        seg = F.interpolate(seg, [seg_dim, seg_dim], mode='bilinear')
        data.append([img, seg])

    return data


############### load CelebAMask-HQ data #################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def load_file_CelebAHQMask(base_path, num_cls, img_dim=128, seg_dim=128):
    dataset_img= make_dataset(os.path.join(base_path, 'CelebA-HQ-img'))
    dataset_seg = make_dataset(os.path.join(base_path, 'CelebAMask-HQ-mask-anno'))

    dataset_img = sorted(dataset_img)
    dataset_seg = sorted(dataset_seg)

    # define part segs and merge labels to 7 classes
    parts = ['_skin', '_u_lip', '_l_lip', '_l_eye', '_r_eye', '_l_brow', '_r_brow', \
    '_l_ear', '_r_ear', '_nose',  '_mouth']
    lbls = {}
    lbls['_u_lip'] = 1
    lbls['_l_lip'] = 1
    lbls['_mouth'] = 1
    lbls['_l_eye'] = 2
    lbls['_r_eye'] = 2
    lbls['_l_brow'] = 3
    lbls['_r_brow'] = 3
    lbls['_l_ear'] = 4
    lbls['_r_ear'] = 4
    lbls['_nose'] = 5
    lbls['_skin'] = 6

    # load images and seg 
    l = [1,3,5,6,8] + list(range(100,200)) # define reference and target image indices
    data_img = [dataset_img[i] for i in l]
    data = []
    for img_path in tqdm(data_img):
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.functional.resize(img, (img_dim, img_dim))
        img = torch.Tensor(np.array(img)).permute(2,0,1).contiguous().unsqueeze(0)
        img = (img/255-0.5)/0.5
        img_idx = os.path.basename(img_path)[:-4]
        seg = load_seg_CelebAHQMask(img_idx, dataset_seg, parts, lbls)
        seg = torch.Tensor(seg2onehot(seg, num_cls)).unsqueeze(0)
        seg = F.interpolate(seg, [seg_dim, seg_dim], mode='bilinear')
        data.append([img, seg])
    return data

def load_seg_CelebAHQMask(img_idx, dataset_seg, parts, lbls):
    seg_img = np.zeros((512,512)).astype(np.uint8)
    seg_paths = [s for s in dataset_seg if str(img_idx).zfill(5) in s]
    dir_path = os.path.dirname(seg_paths[0])
    seg = {}
    for part in parts:
        seg[part] = None
    for part in parts:
        part_path = os.path.join(dir_path, str(img_idx).zfill(5)+part+'.png')
        if os.path.exists(part_path):
            seg[part] = np.array(Image.open(part_path))[:,:,0]>0
            seg_img[seg[part]] = lbls[part]
    return seg_img
