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
from utils import convert_to_coord_format, swap, swap_img
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

def train(args, loader, generator, g_optim, g_ema, device, percep_loss=None, encoder=None):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss = torch.tensor(0.0, device=device)
    g_perc_loss = torch.tensor(0.0, device=device)
    g_enc_consistency_loss = torch.tensor(0.0, device=device)
    g_rec_loss = torch.tensor(0.0, device=device)
    g_rec_perc_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        e_module = encoder.module
        g_module = generator.module

    else:
        e_module = encoder
        g_module = generator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z_style = torch.randn(args.n_sample, args.latent, device=device)
    sample_z_struc = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        data = next(loader)
        real_img = data.to(device)
        std_coords = convert_to_coord_format(args.n_sample, 128, 128, device,
                                                             integer_values=args.coords_integer_values)

        # train generator
        requires_grad(generator, False)
        requires_grad(encoder, True)

        # generate input image latent
        struc_enc, style_enc, _ = encoder(real_img)
        struc_enc = [struc_enc]
        style_enc = [style_enc]
        # generate input image reconstruction
        fake_img_enc, warp_coords_enc, style_w_enc, struc_w_enc = generator(style_enc, struc_enc, mode='mid_sup', input_is_latent=args.enc_input_is_latent)

        # generate swap texture image for input image
        if args.perc_loss_lambda>0:
            swap_style_enc = swap(style_enc)
            swap_style_fake_img_enc, _ = generator([swap_style_enc], struc_enc, input_is_latent=args.enc_input_is_latent)
            swap_struc_fake_img_enc = swap_img(swap_style_fake_img_enc)

        # generate synthesized image
        style_noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        struc_noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, warp_coords, style_w, struc_w = generator(style_noise, struc_noise, mode='mid_sup', input_is_latent=False)

        g_loss = torch.tensor(0.0, device=device)
        # reconstruction L1 loss
        if args.enc_rec_lambda>0:
            g_rec_loss = args.enc_rec_lambda * F.l1_loss(real_img, fake_img_enc)
            g_loss += g_rec_loss

        # reconstruction LPIPS loss
        if args.enc_rec_perc_lambda>0:
            g_rec_perc_loss = args.enc_rec_perc_lambda * percep_loss(real_img, fake_img_enc)
            g_loss += g_rec_perc_loss

        # latent consistency loss
        if args.enc_consistency_lambda>0:
            struc_enc_, style_enc_, _ = encoder(fake_img)
            #struc_enc_warp_coords = generator.module.w_to_warp_coords(std_coords, [struc_enc_])
            struc_enc_warp_coords = g_module.w_to_warp_coords(std_coords, [struc_enc_])
            g_enc_consistency_loss = args.enc_consistency_lambda * F.l1_loss(struc_w, struc_enc_) \
                                    + args.enc_consistency_lambda * F.l1_loss(warp_coords, struc_enc_warp_coords)
            g_loss += g_enc_consistency_loss

        # texture swap loss
        if args.perc_loss_lambda>0:
            g_perc_loss = args.perc_loss_lambda * percep_loss(real_img, swap_style_fake_img_enc)
            g_loss += g_perc_loss


        # log losses
        loss_dict['g_perc'] = g_perc_loss.clone()
        loss_dict['g_enc_consistency'] = g_enc_consistency_loss.clone()
        loss_dict['g_rec'] = g_rec_loss.clone()
        loss_dict['g_rec_perc'] = g_rec_perc_loss.clone()
        loss_dict['g'] = g_loss

        encoder.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        g_loss_val = loss_reduced['g'].mean().item()
        g_perc_val = loss_reduced['g_perc'].mean().item()
        g_enc_consistency_val = loss_reduced['g_enc_consistency'].mean().item()
        g_rec_val = loss_reduced['g_rec'].mean().item()
        g_rec_perc_val = loss_reduced['g_rec_perc'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'g_per: {g_perc_val:.2f}; g_enc_consis: {g_enc_consistency_val:.2f}; g_rec: {g_rec_val:.2f} ; g_rec_perc: {g_rec_perc_val:.2f} ;'
                )
            )

            if i % 100 == 0:
                writer.add_scalar("Generator", g_loss_val, i)
                writer.add_scalar("G_perc", g_perc_val, i)
                writer.add_scalar("G_rec", g_rec_val, i)
                writer.add_scalar("G_rec_perc", g_rec_perc_val, i)
                writer.add_scalar("G_enc_consistency", g_enc_consistency_val, i)

            if i % 2000 == 0:
                with torch.no_grad():
                    g_ema.eval()


                    sample_cur = vis_utils.visualization(args, [style_enc[0], struc_enc[0]], g_ema, std_coords, cur_size=128, alpha=1, device=device, real_img=real_img, enc_input_is_latent=args.enc_input_is_latent)

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

            if i % args.save_checkpoint_frequency == 0:
                save_dic = {
                        'g': g_module.state_dict(),
                        'e': e_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                    }
                

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

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)


    Generator = getattr(model, args.Generator)
    print('Generator', Generator)
    Encoder = getattr(model, args.Encoder)
    print('Encoder', Encoder)

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

    # coordgan generator
    generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                          activation=args.activation, channel_multiplier=args.channel_multiplier,
                          ).to(device)
    # encoder
    encoder = Encoder(args).to(device)
    g_ema = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, channel_multiplier=args.channel_multiplier,
                      ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # gan regularizer schedule
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    # optimizer
    g_params = list(generator.parameters())
    g_params +=  list(encoder.parameters())
    g_optim = optim.Adam(
        g_params,
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt,  map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g_ema'], strict=False)
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)
        if 'e' in ckpt.keys():
            encoder.load_state_dict(ckpt['e'])

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
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
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

    # lpips net
    percep_loss = loss.LPIPS().to(device)
    train(args, loader, generator, g_optim, g_ema, device, percep_loss=percep_loss, encoder=encoder)
