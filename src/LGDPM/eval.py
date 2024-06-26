import argparse
import os
from pathlib import Path

import torch
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
    parser.add_argument(
        "--first_milestone", type=int, default=1, help="Number of the first milestone"
    )
    parser.add_argument(
        "--last_milestone", type=int, help="Number of the last milestone."
    )
    parser.add_argument(
        "--milestone_step",
        type=int,
        default=1,
        help="Step size for milestones to evaluate.",
    )
    parser.add_argument(
        "--ddim_sampling_timesteps",
        type=int,
        default=250,
        help="Number of DDIM sampling timesteps. If this is greater than the training noise steps, DDIM is ignored.",
    )
    args = vars(parser.parse_args())
    config = yaml.load(Path(args["config"]).read_text(), Loader=yaml.Loader)

    unet = ConditionalUNet(**config["UNET_ARGS"])

    diffuser = GaussianDiffusion(
        model=unet,
        sampling_timesteps=args["ddim_sampling_timesteps"],
        **config["DIFFUSER_ARGS"],
    )

    trainer = Trainer(diffusion_model=diffuser, **config["TRAINER_ARGS"])

    fid_scores = {}
    for i in range(
        args["first_milestone"], args["last_milestone"] + 1, args["milestone_step"]
    ):
        trainer.load(i)
        fid = trainer.fid_score_with_num_samples()
        fid_scores.update({i: fid})
        print(f"FID score for milestone {i} = {fid}")
    torch.save(
        fid_scores,
        os.path.join(
            f"{trainer.results_folder}",
            f"{trainer.dataset_name}_Robust_FID_scores_{args['first_milestone']}-{args['last_milestone']}.pkl",
        ),
    )
