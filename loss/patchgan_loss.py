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

"""https://github.com/taesungp/swapping-autoencoder-pytorch/blob/76879fdaa4a4e8e49b3881e6e4691dbcae438992/models/swapping_autoencoder_model.py"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_random_crop(x, target_size, scale_range=(1/8,1/4), num_crops=8, return_rect=False):
    # build grid
    B = x.size(0) * num_crops
    flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
    unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
    unit_grid_y = unit_grid_x.transpose(1, 2)
    unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)

    #crops = []
    x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
    #for i in range(num_crops):
    scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
    sampling_grid = unit_grid * scale + offset
    crop = F.grid_sample(x, sampling_grid, align_corners=False)
    #crops.append(crop)
    #crop = torch.stack(crops, dim=1)
    crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

    return crop


def get_random_crops(args, x):
    """ Make random crops.
        Corresponds to the yellow and blue random crops of Figure 2.
    """
    crops = apply_random_crop(
        x, args.patch_size,
        (args.patch_min_scale, args.patch_max_scale),
        num_crops=args.patch_num_crops
    )
    return crops


def compute_patch_discriminator_losses(args, real, mix, Dpatch):
    if isinstance(Dpatch, nn.parallel.DistributedDataParallel):
        Dpatch = Dpatch.module

    losses = {}

    real_feat = Dpatch.extract_features(
        get_random_crops(args, real),
        aggregate=True,
    )
    target_feat = Dpatch.extract_features(get_random_crops(args, real))
    mix_feat = Dpatch.extract_features(get_random_crops(args, mix))

    losses_PatchD_real = gan_loss(
        Dpatch.discriminate_features(real_feat, target_feat),
        should_be_classified_as_real=True,
    )

    losses_PatchD_mix = gan_loss(
        Dpatch.discriminate_features(real_feat, mix_feat),
        should_be_classified_as_real=False,
    )

    losses = losses_PatchD_real + losses_PatchD_mix
    return losses


def compute_patch_generator_losses(args, real, mix, Dpatch):
    if isinstance(Dpatch, nn.parallel.DistributedDataParallel):
        Dpatch = Dpatch.module
    real_feat = Dpatch.extract_features(
        get_random_crops(args, real),
        aggregate=True).detach()
    mix_feat = Dpatch.extract_features(get_random_crops(args, mix))

    losses = gan_loss(
        Dpatch.discriminate_features(real_feat, mix_feat),
        should_be_classified_as_real=True,
    )
    
    return losses


def d_patch_r1_loss(args, real, Dpatch):

    if isinstance(Dpatch, nn.parallel.DistributedDataParallel):
        Dpatch = Dpatch.module

    real_crop = get_random_crops(args, real).detach()
    real_crop.requires_grad = True
    target_crop = get_random_crops(args, real).detach()
    target_crop.requires_grad = True

    real_feat = Dpatch.extract_features(
        real_crop,
        aggregate=True)
    target_feat = Dpatch.extract_features(target_crop)
    pred_real_patch = Dpatch.discriminate_features(
        real_feat, target_feat
    ).sum()

    grad_real, grad_target = torch.autograd.grad(
        outputs=pred_real_patch,
        inputs=[real_crop, target_crop],
        create_graph=True,
        retain_graph=True,
    )

    dims = list(range(1, grad_real.ndim))
    grad_penalty = grad_real.pow(2).sum(dims) + \
        grad_target.pow(2).sum(dims) 
    return grad_penalty.mean()



def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1).mean()
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1).mean()


