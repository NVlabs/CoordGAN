# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
# Copyright (c) 2021, Multimodal Lab @ Samsung AI Center Moscow
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Jiteng Mu
# ------------------------------------------------------------------------- 

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import utils
import torchvision

import os


def swap(x):
    """ Swaps (or mixes) the ordering of the minibatch """
    x = x[0]
    shape = x.shape
    assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [shape[0] // 2, 2] + list(shape[1:])
    x = x.view(*new_shape)
    x = torch.flip(x, [1])
    return x.view(*shape)


def swap_img(x):
    """ Swaps (or mixes) the ordering of the minibatch """
    shape = x.shape
    assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [shape[0] // 2, 2] + list(shape[1:])
    x = x.view(*new_shape)
    x = torch.flip(x, [1])
    return x.view(*shape)

# eval_fid.py
class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01



def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1)


