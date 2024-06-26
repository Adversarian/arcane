import copy
import math
import os
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import numpy as np

import torch
import torch.nn.functional as F

from accelerate import Accelerator

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA

from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import einsum, nn

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms as T, utils
from tqdm.auto import tqdm


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_data(
    batch_size,
    img_size,
    dataset_name,
    dataset_path,
    pin_memory=False,
    num_workers=None,
    augment_horizontal_flip=False,
):
    if num_workers is None:
        num_workers = cpu_count()
    train_transforms = T.Compose(
        [
            T.Resize(img_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ]
    )

    if dataset_name in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        train_dataset = datasets.__dict__[dataset_name](
            root=dataset_path, train=True, transform=train_transforms
        )
    else:
        train_dataset = datasets.ImageFolder(dataset_path, transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset_name,
        dataset_path,
        num_classes,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        calculate_fid=True,
        num_samples_for_fid=50000,
        pin_memory=True,
        num_workers=None,
        inception_block_idx=2048,
    ):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision="fp16" if fp16 else "no"
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)
            self.num_samples_for_fid = num_samples_for_fid
            self.dataset_stats_loaded = False
        assert has_int_squareroot(
            num_samples
        ), "number of samples must have and integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        dl = get_data(
            self.batch_size,
            self.image_size,
            dataset_name,
            dataset_path,
            pin_memory,
            num_workers,
            augment_horizontal_flip,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )
            self.ema.to(self.device)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        checkpoint = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        torch.save(
            checkpoint, os.path.join(self.results_folder, f"model-{milestone}.pt")
        )

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint = torch.load(
            os.path.join(self.results_folder, f"model-{milestone}.pt"),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint["model"])

        self.step = checkpoint["step"]
        self.opt.load_state_dict(checkpoint["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(checkpoint["ema"])
        if exists(self.accelerator.scaler) and exists(checkpoint["scaler"]):
            self.accelerator.scaler.load_state_dict(checkpoint["scaler"])
    
    def load_from_path(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint["model"])

        self.step = checkpoint["step"]
        self.opt.load_state_dict(checkpoint["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(checkpoint["ema"])
        if exists(self.accelerator.scaler) and exists(checkpoint["scaler"]):
            self.accelerator.scaler.load_state_dict(checkpoint["scaler"])

    @torch.inference_mode()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)
        features = self.inception_v3(samples)[0]
        features = rearrange(features, "... 1 1 -> ...").cpu().numpy()

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):
        if self.channels == 1:
            real_samples, fake_samples = map(
                lambda t: repeat(t, "b 1 ... -> b c ...", c=3),
                (real_samples, fake_samples),
            )
        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(
            lambda t: t[:min_batch], (real_samples, fake_samples)
        )

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid = calculate_frechet_distance(m1, s1, m2, s2)
        return fid

    @torch.inference_mode()
    def load_or_precalc_dataset_stats(self):
        try:
            ckpt = torch.load(
                os.path.join(self.results_folder, f"{self.dataset_name}_stats.pkl")
            )
            self.accelerator.print("Dataset stats loaded from disk.")
        except FileNotFoundError:
            batches = num_to_groups(
                self.num_samples_for_fid, self.batch_size
            )  # TODO: Fix this mess
            stacked_real_features = []
            self.accelerator.print(
                f"Sampling {self.num_samples_for_fid} from the real dataset and stacking their features."
            )
            for i in tqdm(range(len(batches))):
                try:
                    real_samples, _ = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                if self.channels == 1:
                    real_samples = repeat(real_samples, "b 1 ... -> b c ...", c=3)
                real_features = self.inception_v3(real_samples)[0]
                real_features = rearrange(real_features, "... 1 1 -> ...")
                stacked_real_features.append(real_features)
            stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().numpy()
            ckpt = {
                "m2": np.mean(stacked_real_features, axis=0),
                "s2": np.cov(stacked_real_features, rowvar=False),
            }
            torch.save(
                ckpt,
                os.path.join(self.results_folder, f"{self.dataset_name}_stats.pkl"),
            )
            self.accelerator.print("Dataset stats saved to disk for future use.")
        self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score_with_num_samples(self):
        self.inception_v3.eval()
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        batches = num_to_groups(self.num_samples_for_fid, self.batch_size)

        stacked_fake_features = []
        self.accelerator.print(
            f"Generating {self.num_samples_for_fid} fake samples and stacking their features."
        )
        for batch in tqdm(batches):
            fake_labels = torch.randint(0, self.num_classes, (batch,)).to(self.device)
            fake_samples = self.ema.ema_model.sample(classes=fake_labels)
            if self.channels == 1:  # because Inception takes 3-channel inputs
                fake_samples = repeat(fake_samples, "b 1 ... -> b c ...", c=3)
            fake_features = self.inception_v3(fake_samples)[0]
            fake_features = rearrange(fake_features, "... 1 1 -> ...")
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        fid_scores = {}
        min_fid = 1e10
        best_mile = -1
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0
                for _ in range(self.gradient_accumulate_every):
                    data, label = next(self.dl)
                    data, label = data.to(device), label.to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data, classes=label)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if (
                    accelerator.is_main_process
                ):  # TODO: if failed, look at this part. Very sketchy.
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(
                                        classes=torch.randint(
                                            0, self.num_classes, (n,)
                                        ).to(self.device)
                                    ),
                                    batches,
                                )
                            )
                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(
                            all_images,
                            os.path.join(
                                self.results_folder,
                                f"sample-{milestone}_{self.dataset_name}.png",
                            ),
                            nrow=int(math.sqrt(self.num_samples)),
                        )
                        self.save(milestone)

                        if exists(self.inception_v3):
                            fid_score = self.fid_score(
                                real_samples=data, fake_samples=all_images
                            )
                            if fid_score < min_fid:
                                min_fid = fid_score
                                best_mile = milestone
                            fid_scores.update({milestone: fid_score})
                            accelerator.print(f"fid_score: {fid_score}")
                pbar.update(1)
        accelerator.print("Training complete!")
        fid_scores.update({"best": {best_mile: min_fid}})
        torch.save(
            fid_scores,
            os.path.join(self.results_folder, f"fid_scores_{self.dataset_name}.pkl"),
        )
