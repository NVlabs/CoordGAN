# -------------------------------------------------------------------------
# Creative Commons — Attribution-NonCommercial-ShareAlike 4.0 International — CC BY-NC-SA 4.0
# See CC-BY-NC-SA-4.0.md for a human-readable summary of (and not a substitute for) the license.
#
# Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way  that suggests the licensor endorses you or your use.
#
# NonCommercial — You may not use the material for commercial purposes.
#
# ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
#
# No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
#
# Modified by Jiteng Mu
# -------------------------------------------------------------------------



__all__ = [
           'ImageFolder'
           ]

from io import BytesIO
import math

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

import torchvision
import re
import scipy.io

from PIL import Image
import os
import os.path

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


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(Dataset):

    def __init__(self, root, transform, resolution=128, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.resolution = resolution

        self.length = len(self.imgs)
        self.transform = transform

    def __getitem__(self, index):

        path = self.imgs[index]
        img = self.loader(path)

        filename = re.split('/', path)[-2] +'/' + re.split('/', path)[-1]

        img = torchvision.transforms.functional.resize(img, (self.resolution, self.resolution))

        img = self.transform(img)

        return img

    def __len__(self):
        return self.length

