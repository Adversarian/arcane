import sys

sys.path.append("PyTorch-StudioGAN")
sys.path.append("PyTorch-StudioGAN/src")

import config
import torch
from models.model import build_G_D

GAN_CONFIGS = {
    "CIFAR10_ADCGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR10/ACGAN-Mod-Big-ADC.yaml",
    "CIFAR10_READCGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR10/ReACGAN-ADC-DiffAug.yaml",
    "CIFAR10_REACGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR10/ReACGAN-DiffAug.yaml",
    "CIFAR100_ADCGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR100/ACGAN-Mod-Big-ADC.yaml",
    "CIFAR100_READCGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR100/ReACGAN-ADC-DiffAug.yaml",
    "CIFAR100_REACGAN": "PyTorch-StudioGAN/src/configs/Experiments/CIFAR100/ReACGAN-DiffAug.yaml",
    "MNIST_ADCGAN": "",  # TODO: Figure this out!
    "MNIST_READCGAN": "",
    "MNIST_REACGAN": "",
    "TIMGNET_ADCGAN": "PyTorch-StudioGAN/src/configs/Experiments/Tiny_ImageNet/ACGAN-Mod-Big-ADC.yaml",
    "TIMGNET_READCGAN": "PyTorch-StudioGAN/src/configs/Experiments/Tiny_ImageNet/ReACGAN-ADC-DiffAug.yaml",
    "TIMGNET_REACGAN": "PyTorch-StudioGAN/src/configs/Experiments/Tiny_ImageNet/ReACGAN-DiffAug.yaml",
}


def load_model(g_ckpt_path, d_ckpt_path, config_name, device="cuda"):
    cfgs = config.Configurations(GAN_CONFIGS[config_name])
    generator, discriminator = build_G_D(cfgs, device=device)
    g_ckpt = torch.load(g_ckpt_path)
    d_ckpt = torch.load(d_ckpt_path)
    generator.load_state_dict(g_ckpt['state_dict'])
    discriminator.load_state_dict(d_ckpt['state_dict'])
    return generator, discriminator, cfgs.MODEL.z_dim
