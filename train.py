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

import argparse
import math
import random
import os
from tqdm import tqdm
import json
import git

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms
from torchvision import utils as torchvision_utils
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import ImageFolder
from distributed import get_rank, synchronize, reduce_loss_dict
from utils import convert_to_coord_format, swap
import vis_utils
import loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def fix_warping_grad(model, flag=False):
    for n,p in model.named_parameters():
        if "foldnet" in n or "struc" in n:
            p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, patch_discriminator=None, percep_loss=None, style_loss=None, seg_model=None, patch_discriminator_cor=None):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    #mean_path_length = 0
    d_loss = torch.tensor(0.0, device=device)
    real_pred = torch.tensor(0.0, device=device)
    fake_pred = torch.tensor(0.0, device=device)
    d_gan_loss = torch.tensor(0.0, device=device)
    d_patch_loss = torch.tensor(0.0, device=device)
    r1_loss = torch.tensor(0.0, device=device)
    r1_patch_loss = torch.tensor(0.0, device=device)
    g_cham_loss = torch.tensor(0.0, device=device)
    g_patch_loss = torch.tensor(0.0, device=device)
    g_perc_loss = torch.tensor(0.0, device=device)
    g_rgb_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        if patch_discriminator is not None:
            d_patch_module = patch_discriminator.module

    else:
        g_module = generator
        d_module = discriminator
        if patch_discriminator is not None:
            d_patch_module = patch_discriminator
        if patch_discriminator_cor is not None:
            d_patch_cor_module = patch_discriminator_cor

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z_style = torch.randn(args.n_sample, args.latent, device=device)
    sample_z_struc = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        # progressive training
        if args.progressive_training:
            if i <= args.backbone_num_iters:
                cur_size = 128
                alpha = 1
            #elif i <= args.backbone_num_iters+args.upsample_num_iters:
            #    cur_size = 256
            #    alpha = min(1, 1 / args.upsample_warm_num_iters * (i-args.backbone_num_iters))
            else:
                cur_size = 512
                alpha = 1
            #    alpha = min(1, 1 / args.upsample_warm_num_iters * (i-args.backbone_num_iters-args.upsample_num_iters)) 
        else:
            cur_size = args.size
            alpha = 1

        if i > args.iter:
            print('Done!')
            break

        data = next(loader)
        real_img = data.to(device)
        std_coords = convert_to_coord_format(args.n_sample, 128, 128, device,
                                                             integer_values=args.coords_integer_values)

        if cur_size != args.size:
            real_img = F.interpolate(real_img, [cur_size, cur_size], mode='bilinear')

        # train discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        if patch_discriminator is not None:
            requires_grad(patch_discriminator, True)

        style_noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        struc_noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        # generate images
        fake_img, warp_coords, _, _ = generator(style_noise, struc_noise, cur_size, alpha, mode='mid_sup', input_is_latent=False)

        # gan discriminator loss
        fake_pred = discriminator(fake_img, cur_size, alpha)
        real_pred = discriminator(real_img, cur_size, alpha)
        if i>10000:
            d_gan_loss = args.gan_loss_lambda * loss.d_logistic_loss(real_pred, fake_pred)
        else:
            d_gan_loss = max((i/10000), 1/args.gan_loss_lambda) * args.gan_loss_lambda * loss.d_logistic_loss(real_pred, fake_pred)
        loss_dict['d_gan'] = d_gan_loss.clone()
        d_loss = d_gan_loss

        discriminator.zero_grad()

        # generate swap texture images
        swap_style_noise = swap(style_noise)
        swap_style_img, _ = generator([swap_style_noise], struc_noise, cur_size, alpha, input_is_latent=False)
        swap_struc_img = swap([swap_style_img])

        # patchgan discriminator loss
        if args.patch_d_lambda>0 and i>10000:
            d_patch_loss = args.patch_d_lambda * loss.compute_patch_discriminator_losses(args, fake_img, swap_struc_img, patch_discriminator)
            d_loss += d_patch_loss
            patch_discriminator.zero_grad()

        loss_dict['d_patch'] = d_patch_loss.clone()
        loss_dict['d'] = d_loss

        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        # gan | patchgan regularizer
        if d_regularize:
            real_img.requires_grad = True

            # d_r1
            real_pred = discriminator(real_img, cur_size, alpha)
            r1_loss = loss.d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            # d_patch_r1
            if args.patch_d_lambda>0 and i>10000:
                r1_patch_loss = loss.d_patch_r1_loss(args, fake_img, patch_discriminator)
                patch_discriminator.zero_grad()
                (args.patch_r1 / 2 * r1_patch_loss * args.d_reg_every).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss
        loss_dict['r1_patch'] = r1_patch_loss

        # train generator
        requires_grad(generator, True)
        if args.fix_struc_grad:
            fix_warping_grad(generator)
        requires_grad(discriminator, False)
        if patch_discriminator is not None:
            requires_grad(patch_discriminator, False)

        style_noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        struc_noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        # generate images
        fake_img, warp_coords, _, _ = generator(style_noise, struc_noise, cur_size, alpha, mode='mid_sup', input_is_latent=False)

        # gan discriminator loss
        fake_pred = discriminator(fake_img, cur_size, alpha)
        if i>10000:
            g_gan_loss = args.gan_loss_lambda * loss.g_nonsaturating_loss(fake_pred)
        else:
            g_gan_loss =  max((i/10000), 1/args.gan_loss_lambda) * args.gan_loss_lambda * loss.g_nonsaturating_loss(fake_pred)
        loss_dict['g_gan'] = g_gan_loss.clone()
        g_loss = g_gan_loss

        # generate swap texture images
        swap_style_noise = swap(style_noise)
        swap_style_img, _ = generator([swap_style_noise], struc_noise, cur_size, alpha, input_is_latent=False)
        swap_struc_img = swap([swap_style_img])

        # structure swap loss
        if args.patch_d_lambda>0 and i>10000:
            g_patch_loss = args.patch_d_lambda * loss.compute_patch_generator_losses(args, fake_img, swap_struc_img, patch_discriminator)
            g_loss += g_patch_loss

        # texture swap loss
        if args.perc_loss_lambda>0:
            g_perc_loss = min((i/args.warmup_iter), 1) * args.perc_loss_lambda * percep_loss(fake_img, swap_style_img)
            g_loss += g_perc_loss

        # coord warping chamfer loss
        if args.chamfer_lambda>0:
            g_cham_loss = args.chamfer_lambda * loss.warp_chamfer_loss(warp_coords, std_coords)
            g_loss += g_cham_loss

        # warp loss
        if args.rgb_loss_lambda>0 and args.chamfer_lambda>0:
            cor_img = loss.get_cor_img(args, fake_img, warp_coords, 
                                                  temp=args.rgb_temp, mode=args.rgb_mode, down_sample_ratio=args.rgb_down_sample_ratio)
            _,_,H,W = cor_img.shape
            if args.rgb_down_sample_ratio>1 or fake_img.shape[-1]>128:
                fake_img_resize = F.interpolate(fake_img, [H, W], mode='bilinear')
            else:
                fake_img_resize = fake_img
            g_rgb_loss =  min((i/args.warmup_iter), 1) * args.rgb_loss_lambda * percep_loss(cor_img, fake_img_resize.detach())
            g_loss += g_rgb_loss

        # log losses
        loss_dict['g_patch'] = g_patch_loss.clone()
        loss_dict['g_perc'] = g_perc_loss.clone()
        loss_dict['g_cham'] = g_cham_loss.clone()
        loss_dict['g_rgb'] = g_rgb_loss.clone()
        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()

        g_gan_val = loss_reduced['g_gan'].mean().item()
        g_patch_val = loss_reduced['g_patch'].mean().item()
        g_cham_val = loss_reduced['g_cham'].mean().item()
        g_perc_val = loss_reduced['g_perc'].mean().item()
        g_rgb_val = loss_reduced['g_rgb'].mean().item()

        d_gan_val = loss_reduced['d_gan'].mean().item()
        d_patch_val = loss_reduced['d_patch'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        r1_patch_val = loss_reduced['r1_patch'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd_gan: {d_gan_val:.2f}; d_patch: {d_patch_val:.2f}; g_gan: {g_gan_val:.2f}; g_patch: {g_patch_val:.2f}; g_cham: {g_cham_val:.2f}; g_per: {g_perc_val:.2f}; g_rgb: {g_rgb_val:.2f}; '
                )
            )

            if i % 100 == 0:
                writer.add_scalar("Generator", g_loss_val, i)
                writer.add_scalar("Discriminator", d_loss_val, i)
                writer.add_scalar("G_gan", g_gan_val, i)
                writer.add_scalar("D_gan", d_gan_val, i)
                writer.add_scalar("R1", r1_val, i)
                writer.add_scalar("G_patch", g_patch_val, i)
                writer.add_scalar("D_patch", d_patch_val, i)
                writer.add_scalar("R1_patch", r1_patch_val, i)
                writer.add_scalar("G_chamfer", g_cham_val, i)
                writer.add_scalar("G_perc", g_perc_val, i)
                writer.add_scalar("G_rgb", g_rgb_val, i)

            if i % 2000 == 0:
                with torch.no_grad():
                    g_ema.eval()

                    sample_cur = vis_utils.visualization(args, [sample_z_style, sample_z_struc], g_ema, std_coords, cur_size, alpha, device)

                    torchvision_utils.save_image(
                        sample_cur,
                        os.path.join(path, 'configs', args.output_dir, 'images', f'{str(i).zfill(6)}_cur.png'),
                        nrow=int(sample_cur.shape[0]/args.n_sample),
                        normalize=True,
                        range=(-1, 1),
                    )

                    if i==0:
                        torchvision_utils.save_image(
                            real_img,
                            os.path.join(
                                path,
                                f'configs/{args.output_dir}/images/real_patch_{str(i)}_{str(i).zfill(6)}.png'),
                            nrow=int(sample_cur.shape[0]/args.n_sample),
                            normalize=True,
                            range=(-1, 1),
                        )


            if i % (args.save_checkpoint_frequency//2) == 0:
                save_dic = {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    }
                if args.patch_d_lambda>0:
                    save_dic['d_patch'] = d_patch_module.state_dict()

                torch.save(
                    save_dic,
                    os.path.join(
                        path,
                        f'configs/{args.output_dir}/checkpoints/latest.pt'),
                )

                if i % args.save_checkpoint_frequency == 0:

                    torch.save(
                        save_dic,
                        os.path.join(
                            path,
                            f'configs/{args.output_dir}/checkpoints/{str(i).zfill(6)}.pt'),
                    )



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
            help='Load settings from file in json format. Command line options override values in file.')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--progressive_training', type=str2bool, default=False)
    parser.add_argument('--warmup_iter', type=int, default=20000)
    parser.add_argument('--backbone_num_iters', type=int, default=500000)
    parser.add_argument('--fix_struc_grad', type=str2bool, default=False)
    parser.add_argument('--reinit_discriminator', type=str2bool, default=False)

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    # load module
    Generator = getattr(model, args.Generator)
    print('Generator', Generator)
    Discriminator = getattr(model, args.Discriminator)
    print('Discriminator', Discriminator)
    PatchDiscriminator = getattr(model, args.PatchDiscriminator)
    print('PatchDiscriminator', PatchDiscriminator)

    # create log dirs
    path = './'
    os.makedirs(os.path.join(path, 'configs', args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'configs', args.output_dir, 'checkpoints'), exist_ok=True)
    args.logdir = os.path.join(path, 'configs', args.output_dir, 'tensorboard')
    os.makedirs(args.logdir, exist_ok=True)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

        if get_rank() == 0:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            args.githash = sha

            # Optional: support for saving settings into a json file
            with open(os.path.join(path, 'configs', args.output_dir, 'saved_config.yaml'), 'wt') as f:
                json.dump(vars(args), f, indent=4)



    # LPIPS net
    percep_loss = loss.LPIPS().to(device)

    # coordgan generator
    generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                          activation=args.activation, channel_multiplier=args.channel_multiplier,
                          ).to(device)

    # coordgan discriminator
    discriminator = Discriminator(
        size=args.size, channel_multiplier=args.channel_multiplier, n_scales=1, input_size=3,
    ).to(device)

    # coordgan patch discriminator
    if args.patch_d_lambda>0:
        patch_discriminator = PatchDiscriminator(args).to(device)
    else:
        patch_discriminator = None

    # coordgan generator ema
    g_ema = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, channel_multiplier=args.channel_multiplier,
                      ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # gan regularizer schedule
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # optimizer
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    if args.patch_d_lambda>0:
        print("GAN and PatchGAN")
        d_params = list(discriminator.parameters()) + list(patch_discriminator.parameters())
    else:
        print("GAN only")
        d_params = discriminator.parameters()

    d_optim = optim.Adam(
        d_params,
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # load ckpts
    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt,  map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'], strict=False)
        if not args.reinit_discriminator:
            discriminator.load_state_dict(ckpt['d'], strict=False)
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)
        if args.patch_d_lambda>0:
            patch_discriminator.load_state_dict(ckpt['d_patch'])

        del ckpt
        torch.cuda.empty_cache()

    # model distributed training
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if args.patch_d_lambda>0:
            patch_discriminator = nn.parallel.DistributedDataParallel(
                patch_discriminator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    # dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageFolder(args.path, transform=transform, resolution=args.size)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    writer = SummaryWriter(log_dir=args.logdir)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, patch_discriminator=patch_discriminator, percep_loss=percep_loss)
