from pathlib import Path

import yaml
from LGDPM.diffusion import ConditionalUNet, GaussianDiffusion
from LGDPM.utils import Trainer

DIFF_CONFIGS = {
    "CIFAR10_DIFF": "LGDPM/configs/CIFAR10_eval.yaml",
    "CIFAR100_DIFF": "LGDPM/configs/CIFAR100_eval.yaml",
    "MNIST_DIFF": "LGDPM/configs/MNIST_eval.yaml",
    "TIMGNET_DIFF": "LGDPM/configs/Tiny_ImageNet_eval.yaml",
}


def load_model(ckpt_path, config_name):
    config = yaml.load(Path(DIFF_CONFIGS[config_name]).read_text(), Loader=yaml.Loader)
    unet = ConditionalUNet(**config["UNET_ARGS"])

    diffuser = GaussianDiffusion(model=unet, **config["DIFFUSER_ARGS"])

    trainer = Trainer(diffusion_model=diffuser, **config["TRAINER_ARGS"])

    trainer.load_from_path(ckpt_path)

    return trainer.model, trainer.ema.ema_model
