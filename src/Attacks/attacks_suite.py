import torch
import torchattacks as ta

from easydict import EasyDict
from tqdm import tqdm

DEFAULT_MOMENTS = EasyDict(
    {
        "MNIST": {"mean": [0.1307], "std": [0.3081]},
        "CIFAR10": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
        "TIMGNET": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
)

ATTACK_DEFAULT_HPS = EasyDict(
    {
        "CW": {
            "kappa": 14,
            "binary_search_steps": 10,
            "steps": 100,
            "lr": 1e-2,
        },
        "PGD": {
            "steps": 40,
        },
        "eps": {
            "CIFAR10": 0.05,
            "TIMGNET": 0.05,
            "MNIST": 0.2,
        },
    }
)

NUM_CLASSES = {
    "MNIST": 10,
    "CIFAR10": 10,
    "TIMGNET": 200,
}


def get_acc(model, images, labels):
    """
    Calculates the accuracy of a given model on a batch of images and labels.

    Args:
        model (torch.nn.Module): The model to evaluate.
        images (torch.Tensor): The input images.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        float: The accuracy of the model on the given batch.

    Raises:
        None
    """
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return accuracy


def run_attack_suite(
    model,
    dataset,
    images,
    labels,
    splits=2,
    targets=None,
    moments=DEFAULT_MOMENTS,
    hps=ATTACK_DEFAULT_HPS,
):
    """
    Runs an attack suite on a given model using a dataset of images.

    Args:
        model (object): The model to be attacked.
        dataset (str): The dataset used for the attack.
        images (list): The list of images to be attacked.
        labels (list): The labels corresponding to the images.
        splits (int, optional): The number of splits to divide the images into for batch processing. Defaults to 2.
        targets (list, optional): The targeted labels for the attack. If None, the attack is untargeted. Defaults to None.
        moments (dict, optional): The moments used for normalization. Defaults to DEFAULT_MOMENTS.
        hps (object, optional): The hyperparameters for the attack. Defaults to ATTACK_DEFAULT_HPS.

    Returns:
        dict: A dictionary containing the results of the attack, including clean accuracy, adversarial images, and robust accuracy.

    Raises:
        AssertionError: If the batch size is not divisible by the number of splits.

    """
    assert (
        len(images) % splits == 0
    ), "Batch size must be divisible by the number of splits."
    result = {}
    result["clean_accuracy"] = get_acc(
        model,
        images,
        labels,
    )
    attacks = {
        "CW": ta.CWBS(model, **hps.CW),
        "FGSM": ta.FGSM(model, eps=hps.eps[dataset]),
        "PGD": ta.PGD(model, eps=hps.eps[dataset]),
    }
    if targets is not None:
        adv_labels = (
            targets if targets != "auto" else (labels + 1) % NUM_CLASSES[dataset]
        )
    else:
        adv_labels = labels
    image_splits = torch.chunk(images, splits)
    label_splits = torch.chunk(adv_labels, splits)
    for method, atk in tqdm(attacks.items(), desc="Generating attacks"):
        atk.set_normalization_used(mean=moments[dataset].mean, std=moments[dataset].std)
        if targets is not None:
            atk.set_mode_targeted_by_label()
        adv_images = []
        for image_split, label_split in zip(image_splits, label_splits):
            adv_images.append(atk(image_split, label_split))
            torch.cuda.empty_cache()
        adv_images = torch.cat(adv_images, dim=0)
        result[method] = {
            "unnormalized_clipped_samples": atk.inverse_normalize(adv_images).clamp_(
                0, 1
            ),
            "normalized_clipped_samples": adv_images,
            "robust_accuracy": get_acc(
                model,
                adv_images,
                adv_labels,
            ),
        }
    result["clean_labels"] = labels
    return result
