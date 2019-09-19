import itertools
import os
import time

from datetime import datetime

import numpy as np
import torch
import torchvision.utils as vutils

from torch.optim.lr_scheduler import MultiStepLR

import utils

from model_3dgan import Generator as G
from model_3dgan import Discriminator as D


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Model(object):
    def __init__(self,
                 name,
                 device,
                 data_loader,
                 latent_dim,
                 cube_len):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.latent_dim = latent_dim
        self.cube_len = cube_len
        assert self.name == '3dgan'
        self.netG = G(self.latent_dim, self.cube_len)
        # self.netG.apply(_weights_init)
        self.netG.to(self.device)
        self.netD = D(self.cube_len)
        # self.netD.apply(_weights_init)
        self.netD.to(self.device)
        self.optim_G = None
        self.optim_D = None
        self.scheduler_D = None
        self.criterion = torch.nn.BCELoss()

    @property
    def generator(self):
        return self.netG

    @property
    def discriminator(self):
        return self.netD

    def create_optim(self, g_lr, d_lr, alpha=0.5, beta=0.5):
        self.optim_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=g_lr,
                                        betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(self.netD.parameters(),
                                          lr=d_lr,
                                          betas=(alpha, beta))
        self.scheduler_D = MultiStepLR(self.optim_D, milestones=[500, 1000])

    def train(self,
              epochs,
              d_loss_thresh,
              log_interval=100,
              export_interval=10,
              out_dir='',
              verbose=True):
        self.netG.train()
        self.netD.train()
        total_time = time.time()
        for epoch in range(epochs):
            batch_time = time.time()
            for batch_idx, data in enumerate(self.data_loader):
                data = data.to(self.device)

                batch_size = data.shape[0]
                real_label = torch.Tensor(batch_size).uniform_(0.7, 1.2).to(self.device)
                fake_label = torch.Tensor(batch_size).uniform_(0, 0.3).to(self.device)

                # Train D
                d_real = self.netD(data)
                d_real = d_real.squeeze()
                d_real_loss = self.criterion(d_real, real_label)

                latent = torch.Tensor(batch_size, self.latent_dim).normal_(0, 0.33).to(self.device)
                fake = self.netG(latent)
                d_fake = self.netD(fake.detach())
                d_fake = d_fake.squeeze()
                d_fake_loss = self.criterion(d_fake, fake_label)

                d_loss = d_real_loss + d_fake_loss

                d_real_acc = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acc = torch.le(d_fake.squeeze(), 0.5).float()
                d_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc),0))

                if d_acc <= d_loss_thresh:
                    self.netD.zero_grad()
                    d_loss.backward()
                    self.optim_D.step()

                # Train G
                latent = torch.Tensor(batch_size, self.latent_dim).normal_(0, 0.33).to(self.device)
                fake = self.netG(latent)
                d_fake = self.netD(fake)
                d_fake = d_fake.squeeze()
                g_loss = self.criterion(d_fake, real_label)

                self.netD.zero_grad()
                self.netG.zero_grad()
                g_loss.backward()
                self.optim_G.step()

                if verbose and batch_idx % log_interval == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} acc_D: {:.4f} time: {:.2f}'.format(
                          epoch, batch_idx, len(self.data_loader),
                          d_loss.mean().item(),
                          g_loss.mean().item(),
                          d_acc.mean().item(),
                          time.time() - batch_time))
                    batch_time = time.time()
            if epoch % export_interval == 0:
                samples = fake.cpu().data[:8].squeeze().numpy()
                utils.save_voxels(samples, out_dir, epoch)
                self.save_to(path=out_dir, name=self.name, verbose=False)
            self.scheduler_D.step()
        if verbose:
            print('Total train time: {:.2f}'.format(time.time() - total_time))

    def eval(self,
             batch_size=None):
        self.netG.eval()
        self.netD.eval()
        if batch_size is None:
            batch_size = 8

        with torch.no_grad():
            latent = torch.normal(0, 0.33, size=(batch_size, self.latent_dim)).to(self.device)
            fake = self.netG(latent)
            samples = fake.cpu().data[:8].squeeze().numpy()
            utils.save_voxels(samples, '', 'eval')

    def save_to(self,
                path='',
                name=None,
                verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nSaving models to {}_G.pt and {}_D.pt ...'.format(name, name))
        torch.save(self.netG.state_dict(), os.path.join(path, '{}_G.pt'.format(name)))
        torch.save(self.netD.state_dict(), os.path.join(path, '{}_D.pt'.format(name)))

    def load_from(self,
                  path='',
                  name=None,
                  verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nLoading models from {}_G.pt and {}_D.pt ...'.format(name, name))
        ckpt_G = torch.load(os.path.join(path, '{}_G.pt'.format(name)))
        if isinstance(ckpt_G, dict) and 'state_dict' in ckpt_G:
            self.netG.load_state_dict(ckpt_G['state_dict'], strict=True)
        else:
            self.netG.load_state_dict(ckpt_G, strict=True)
        ckpt_D = torch.load(os.path.join(path, '{}_D.pt'.format(name)))
        if isinstance(ckpt_D, dict) and 'state_dict' in ckpt_D:
            self.netD.load_state_dict(ckpt_D['state_dict'], strict=True)
        else:
            self.netD.load_state_dict(ckpt_D, strict=True)
