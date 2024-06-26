import argparse
from pathlib import Path

import yaml
from diffusion import ConditionalUNet, GaussianDiffusion
from utils import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training utility for label guided diffusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/CIFAR10.yaml",
        help="Path to config file",
    )
    args = vars(parser.parse_args())
    config = yaml.load(Path(args["config"]).read_text(), Loader=yaml.Loader)

    unet = ConditionalUNet(**config["UNET_ARGS"])

    diffuser = GaussianDiffusion(model=unet, **config["DIFFUSER_ARGS"])

    trainer = Trainer(diffusion_model=diffuser, **config["TRAINER_ARGS"])

    trainer.train()
