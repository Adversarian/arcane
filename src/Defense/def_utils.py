import json

import torch
import torch.nn.functional as F
from einops import repeat
from pytorch_histogram_matching import Histogram_Matching
from torchmetrics.functional.regression import kl_divergence
from torchvision import transforms as T
from tqdm import tqdm
from xgboost import XGBClassifier

HM = Histogram_Matching()

DISCRIMINATOR_MOMENTS = {
    1: {"mean": [0.5], "std": [0.5]},
    3: {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
}

RECONSTRUCTION_LOSS = torch.nn.MSELoss()


class NormalizeInverse(T.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv, *args, **kwargs)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def JSD(p, q, log_prob=False):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Args:
        p (torch.Tensor): The first probability distribution.
        q (torch.Tensor): The second probability distribution.
        log_prob (bool, optional): Whether to use the logarithm of the probabilities. Defaults to False.

    Returns:
        torch.Tensor: The JSD value between the two probability distributions.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, log_prob=log_prob) + 0.5 * kl_divergence(
        q, m, log_prob=log_prob
    )


@torch.inference_mode()
def diffusion_metrics(
    conditional_diffuser,
    auxiliary_discriminator,
    inputs,
    labels,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    Calculates diffusion metrics using a conditional diffuser and an auxiliary discriminator.

    Args:
        conditional_diffuser: The conditional diffuser used for diffusion.
        auxiliary_discriminator: The auxiliary discriminator used for classification.
        inputs: The input data.
        labels: The labels for the input data.
        splits: The number of splits to use for processing the input data. Defaults to 1.
        **targeted_purify_kwargs: Additional keyword arguments to pass to the targeted purify function.

    Returns:
        recon_loss: The reconstruction loss.
        veracity: The veracity scores.
        class_posteriors: The class posteriors.
    """
    num_channels = inputs.shape[1]
    discriminator_normalizer = T.Normalize(
        mean=DISCRIMINATOR_MOMENTS[num_channels]["mean"],
        std=DISCRIMINATOR_MOMENTS[num_channels]["std"],
    )
    recon_loss = diffusion_purify_targeted(
        conditional_diffuser,
        inputs,
        labels,
        return_purified=False,
        splits=splits,
        **targeted_purify_kwargs,
    )
    dis_inputs = discriminator_normalizer(inputs)
    dis_inputs_chunks = torch.chunk(dis_inputs, splits)
    veracity, class_posteriors = [], []
    for dis_inputs_chunk in dis_inputs_chunks:
        d_result = auxiliary_discriminator.forward_emb(dis_inputs_chunk)
        veracity.append(F.sigmoid(d_result["adv_output"]))
        class_posteriors.append(F.softmax(d_result["cls_output"], dim=1))
    return recon_loss, torch.cat(veracity), torch.cat(class_posteriors)


@torch.inference_mode()
def diffusion_generate_detection_X(
    conditional_diffuser,
    auxiliary_discriminator,
    victim,
    clean_inputs,
    adv_norm_inputs,
    inverse_normalizer,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    This function performs diffusion detection on the given inputs.

    Args:
        conditional_diffuser (object): The conditional diffuser object.
        auxiliary_discriminator (object): The auxiliary discriminator object.
        victim (object): The victim object.
        xgb_clf (object): The XGBoost classifier object.
        clean_inputs (torch.Tensor): The clean inputs.
        adv_norm_inputs (torch.Tensor): The adversarial normalized inputs.
        inverse_normalizer (function): The inverse normalizer function.
        splits (int, optional): The number of splits. Defaults to 1.
        **targeted_purify_kwargs (dict): Additional keyword arguments for targeted purify.

    Returns:
        numpy.ndarray: The predictions from the XGBoost classifier.
    """
    ### CLEAN

    clean_inputs_chunks = torch.chunk(clean_inputs, splits)
    clean_victim_class_posteriors = F.softmax(
        torch.cat([victim(chunk).detach() for chunk in clean_inputs_chunks], dim=0),
        dim=1,
    )
    clean_victim_labels = torch.argmax(clean_victim_class_posteriors, dim=1)
    clean_victim_max_class_posteriors = clean_victim_class_posteriors.gather(
        dim=1, index=clean_victim_labels[:, None]
    ).squeeze()
    (
        clean_recon_loss,
        clean_veracity,
        clean_dis_class_posteriors,
    ) = diffusion_metrics(
        conditional_diffuser,
        auxiliary_discriminator,
        inverse_normalizer(clean_inputs),
        clean_victim_labels,
        splits,
        disable_tqdm=True,
        **targeted_purify_kwargs,
    )
    clean_dis_max_class_posteriors = clean_dis_class_posteriors.gather(
        dim=1, index=clean_victim_labels[:, None]
    ).squeeze()
    clean_jsd = torch.as_tensor(
        list(
            map(
                JSD,
                clean_victim_class_posteriors.unsqueeze(-2),
                clean_dis_class_posteriors.unsqueeze(-2),
            )
        )
    )
    clean_sum_of_logs = torch.log(clean_veracity) + torch.log(
        clean_dis_max_class_posteriors
    )

    ### ADVERSARIAL

    adv_norm_inputs_chunks = torch.chunk(adv_norm_inputs, splits)
    adv_victim_class_posteriors = F.softmax(
        torch.cat(
            [victim(chunk).detach() for chunk in adv_norm_inputs_chunks],
            dim=0,
        ),
        dim=1,
    )
    adv_victim_labels = torch.argmax(adv_victim_class_posteriors, dim=1)
    adv_victim_max_class_posteriors = adv_victim_class_posteriors.gather(
        dim=1, index=adv_victim_labels[:, None]
    ).squeeze()
    adv_recon_loss, adv_veracity, adv_dis_class_posteriors = diffusion_metrics(
        conditional_diffuser,
        auxiliary_discriminator,
        inverse_normalizer(adv_norm_inputs),
        adv_victim_labels,
        splits=splits,
        disable_tqdm=True,
        **targeted_purify_kwargs,
    )
    adv_dis_max_class_posteriors = adv_dis_class_posteriors.gather(
        dim=1, index=adv_victim_labels[:, None]
    ).squeeze()
    adv_jsd = torch.as_tensor(
        list(
            map(
                JSD,
                adv_victim_class_posteriors.unsqueeze(-2),
                adv_dis_class_posteriors.unsqueeze(-2),
            )
        )
    )
    adv_sum_of_logs = torch.log(adv_veracity) + torch.log(adv_dis_max_class_posteriors)

    ### AGGREGATE

    clean_X = torch.vstack(
        [
            clean_recon_loss.cpu(),
            clean_veracity.cpu(),
            clean_dis_max_class_posteriors.cpu(),
            clean_victim_max_class_posteriors.cpu(),
            clean_jsd.cpu(),
            clean_sum_of_logs.cpu(),
        ]
    )
    adv_X = torch.vstack(
        [
            adv_recon_loss.cpu(),
            adv_veracity.cpu(),
            adv_dis_max_class_posteriors.cpu(),
            adv_victim_max_class_posteriors.cpu(),
            adv_jsd.cpu(),
            adv_sum_of_logs.cpu(),
        ]
    )
    return torch.vstack([clean_X.T, adv_X.T]).numpy()


def acgan_metrics(
    conditional_generator,
    auxiliary_discriminator,
    inputs,
    labels,
    z_dim,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    Calculate the metrics for the Auxiliary Conditional Generative Adversarial Network (ACGAN).

    Args:
        conditional_generator (nn.Module): The conditional generator network.
        auxiliary_discriminator (nn.Module): The auxiliary discriminator network.
        inputs (Tensor): The input data tensor of shape (batch_size, num_channels, height, width).
        labels (Tensor): The target labels tensor of shape (batch_size, num_classes).
        z_dim (int): The dimension of the latent space.
        splits (int, optional): The number of splits to divide the input data tensor into. Defaults to 1.
        **targeted_purify_kwargs: Additional keyword arguments for the targeted purification function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing the following elements:
            - recon_loss (Tensor): The reconstruction loss tensor.
            - veracity (Tensor): The veracity scores tensor of shape (batch_size).
            - class_posteriors (Tensor): The class posterior probabilities tensor of shape (batch_size, num_classes).
    """
    num_channels = inputs.shape[1]
    discriminator_normalizer = T.Normalize(
        mean=DISCRIMINATOR_MOMENTS[num_channels]["mean"],
        std=DISCRIMINATOR_MOMENTS[num_channels]["std"],
    )
    recon_loss = acgan_purify_targeted(
        conditional_generator,
        inputs,
        labels,
        z_dim,
        return_purified=False,
        splits=splits,
        **targeted_purify_kwargs,
    )
    dis_inputs = discriminator_normalizer(inputs).detach()
    dis_inputs_chunks = torch.chunk(dis_inputs, splits)
    veracity, class_posteriors = [], []
    for dis_inputs_chunk in dis_inputs_chunks:
        d_result = auxiliary_discriminator.forward_emb(dis_inputs_chunk)
        veracity.append(F.sigmoid(d_result["adv_output"].detach()))
        class_posteriors.append(F.softmax(d_result["cls_output"].detach(), dim=1))
    return recon_loss, torch.cat(veracity), torch.cat(class_posteriors)


def acgan_generate_detection_X(
    conditional_generator,
    auxiliary_discriminator,
    victim,
    clean_inputs,
    adv_norm_inputs,
    inverse_normalizer,
    z_dim,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    Generate detection X using an ACGAN model.

    Args:
        conditional_generator (torch.nn.Module): The conditional generator model.
        auxiliary_discriminator (torch.nn.Module): The auxiliary discriminator model.
        victim (torch.nn.Module): The victim model.
        clean_inputs (torch.Tensor): The clean inputs.
        adv_norm_inputs (torch.Tensor): The adversarial normalized inputs.
        inverse_normalizer (Callable): The inverse normalizer function.
        z_dim (int): The dimension of the noise vector.
        splits (int, optional): The number of splits. Defaults to 1.
        **targeted_purify_kwargs: Additional keyword arguments for targeted purify.

    Returns:
        numpy.ndarray: The generated detection X.

    """
    ### CLEAN

    clean_inputs_chunks = torch.chunk(clean_inputs, splits)
    clean_victim_class_posteriors = F.softmax(
        torch.cat([victim(chunk).detach() for chunk in clean_inputs_chunks], dim=0),
        dim=1,
    )
    clean_victim_labels = torch.argmax(clean_victim_class_posteriors, dim=1)
    clean_victim_max_class_posteriors = clean_victim_class_posteriors.gather(
        dim=1, index=clean_victim_labels[:, None]
    ).squeeze()
    (
        clean_recon_loss,
        clean_veracity,
        clean_dis_class_posteriors,
    ) = acgan_metrics(
        conditional_generator,
        auxiliary_discriminator,
        inverse_normalizer(clean_inputs),
        clean_victim_labels,
        z_dim,
        splits,
        disable_tqdm=True,
        **targeted_purify_kwargs,
    )
    clean_dis_max_class_posteriors = clean_dis_class_posteriors.gather(
        dim=1, index=clean_victim_labels[:, None]
    ).squeeze()
    clean_jsd = torch.as_tensor(
        list(
            map(
                JSD,
                clean_victim_class_posteriors.unsqueeze(-2),
                clean_dis_class_posteriors.unsqueeze(-2),
            )
        )
    )
    clean_sum_of_logs = torch.log(clean_veracity) + torch.log(
        clean_dis_max_class_posteriors
    )

    ### ADVERSARIAL

    adv_norm_inputs_chunks = torch.chunk(adv_norm_inputs, splits)
    adv_victim_class_posteriors = F.softmax(
        torch.cat(
            [victim(chunk).detach() for chunk in adv_norm_inputs_chunks],
            dim=0,
        ),
        dim=1,
    )
    adv_victim_labels = torch.argmax(adv_victim_class_posteriors, dim=1)
    adv_victim_max_class_posteriors = adv_victim_class_posteriors.gather(
        dim=1, index=adv_victim_labels[:, None]
    ).squeeze()
    adv_recon_loss, adv_veracity, adv_dis_class_posteriors = acgan_metrics(
        conditional_generator,
        auxiliary_discriminator,
        inverse_normalizer(adv_norm_inputs),
        adv_victim_labels,
        z_dim,
        splits,
        disable_tqdm=True,
        **targeted_purify_kwargs,
    )
    adv_dis_max_class_posteriors = adv_dis_class_posteriors.gather(
        dim=1, index=adv_victim_labels[:, None]
    ).squeeze()
    adv_jsd = torch.as_tensor(
        list(
            map(
                JSD,
                adv_victim_class_posteriors.unsqueeze(-2),
                adv_dis_class_posteriors.unsqueeze(-2),
            )
        )
    )
    adv_sum_of_logs = torch.log(adv_veracity) + torch.log(adv_dis_max_class_posteriors)

    ### AGGREGATE

    clean_X = torch.vstack(
        [
            clean_recon_loss.cpu(),
            clean_veracity.cpu(),
            clean_dis_max_class_posteriors.cpu(),
            clean_victim_max_class_posteriors.cpu(),
            clean_jsd.cpu(),
            clean_sum_of_logs.cpu(),
        ]
    )
    adv_X = torch.vstack(
        [
            adv_recon_loss.cpu(),
            adv_veracity.cpu(),
            adv_dis_max_class_posteriors.cpu(),
            adv_victim_max_class_posteriors.cpu(),
            adv_jsd.cpu(),
            adv_sum_of_logs.cpu(),
        ]
    )
    return torch.vstack([clean_X.T, adv_X.T]).numpy()


def acgan_purify_targeted(
    conditional_generator,
    inputs,
    labels,
    z_dim,
    return_purified=True,
    device="cuda",
    lr=0.01,
    optimization_steps=500,
    optimizer_name="AdamW",
    splits=1,
    disable_tqdm=True,
    **optimizer_kwargs,
):
    """
    Purifies the inputs by optimizing the latent space representation using an
    ACGAN (Auxiliary Classifier Generative Adversarial Network) model in a targeted manner.

    Args:
        conditional_generator (nn.Module): The conditional generator model.
        inputs (Tensor): The input data.
        labels (Tensor): The corresponding labels for the input data.
        z_dim (int): The dimension of the latent space.
        return_purified (bool, optional): Whether to return the purified outputs. Defaults to True.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        optimization_steps (int, optional): The number of optimization steps. Defaults to 500.
        optimizer_name (str, optional): The name of the optimizer. Defaults to "AdamW".
        splits (int, optional): The number of splits to divide the inputs and labels into. Defaults to 1.
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to True.
        **optimizer_kwargs: Additional keyword arguments to be passed to the optimizer.

    Returns:
        Tuple[Tensor, Tensor] or Tensor: A tuple containing the losses and purified outputs if
            return_purified is True, otherwise only the losses.
    """
    assert (
        len(inputs) % splits == 0
    ), "Batch size must be divisible by the number of splits."
    inputs = inputs.to(device)
    labels = labels.to(device)
    input_chunks = torch.chunk(inputs, chunks=splits)
    label_chunks = torch.chunk(labels, chunks=splits)
    losses, purified = [], []
    for input_chunk, label_chunk in tqdm(
        zip(input_chunks, label_chunks), disable=disable_tqdm
    ):
        z = torch.zeros(
            [input_chunk.shape[0], z_dim], device=device, requires_grad=True
        )
        optimizer = torch.optim.__dict__[optimizer_name]([z], lr=lr, **optimizer_kwargs)
        for _ in tqdm(
            range(optimization_steps),
            desc="ACGAN Purify Targeted: Optimizing Z",
            disable=disable_tqdm,
        ):
            optimizer.zero_grad()
            outputs = conditional_generator(z, label_chunk, eval=True)
            outputs = (outputs + 1) / 2
            loss = RECONSTRUCTION_LOSS(outputs, input_chunk)
            loss.backward()
            optimizer.step()
        outputs = conditional_generator(z, label_chunk, eval=True).detach()
        outputs = ((outputs + 1) / 2).clamp(0, 1)
        loss = list(map(RECONSTRUCTION_LOSS, outputs, input_chunk))
        losses += loss
        if return_purified:
            purified.append(outputs)
    losses = torch.as_tensor(losses).detach()
    if return_purified:
        purified = torch.cat(purified, dim=0).detach()
        return losses, purified
    else:
        return losses


def acgan_purify_untargeted(
    conditional_generator,
    inputs,
    num_classes,
    z_dim,
    return_purified=True,
    device="cuda",
    lr=0.01,
    optimization_steps=500,
    optimizer_name="AdamW",
    **optimizer_kwargs,
):
    """
    Purify untargeted ACGAN.

    Args:
        conditional_generator (nn.Module): The conditional generator.
        inputs (torch.Tensor): The input tensor.
        num_classes (int): The number of classes.
        z_dim (int): The dimension of the latent space.
        return_purified (bool, optional): Whether to return the purified outputs. Defaults to True.
        device (str, optional): The device to use. Defaults to "cuda".
        lr (float, optional): The learning rate. Defaults to 0.01.
        optimization_steps (int, optional): The number of optimization steps. Defaults to 500.
        optimizer_name (str, optional): The name of the optimizer. Defaults to "AdamW".
        **optimizer_kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        tuple or list: If `return_purified` is True, returns a tuple containing the best losses and best outputs.
            Otherwise, returns a list of the best losses.
    """
    inputs = inputs.to(device)
    best_outputs, best_losses = [], []
    labels = torch.arange(0, num_classes).to(device)
    optim_class = torch.optim.__dict__[optimizer_name]
    for i in tqdm(
        range(len(inputs)), desc="ACGAN Purify Untargeted: Calculating Losses"
    ):
        z = torch.zeros([num_classes, z_dim], device=device, requires_grad=True)
        optimizer = optim_class([z], lr=lr, **optimizer_kwargs)
        rep_input = inputs[i, ...].expand(num_classes, *[-1] * (inputs.ndim - 1))
        for _ in range(optimization_steps):
            optimizer.zero_grad()
            outputs = conditional_generator(z, labels, eval=True)
            outputs = (outputs + 1) / 2
            loss = RECONSTRUCTION_LOSS(outputs, rep_input)
            loss.backward()
            optimizer.step()
        x_recon = conditional_generator(z, labels, eval=True).detach()
        x_recon = ((x_recon + 1) / 2).clamp(0, 1)
        x_recon = HM(x_recon, rep_input)
        losses = list(map(RECONSTRUCTION_LOSS, x_recon, rep_input))
        best_arg = torch.argmin(torch.as_tensor(losses))
        best_outputs.append(x_recon[best_arg].detach())
        best_losses.append(losses[best_arg].item())
        torch.cuda.empty_cache()
    if return_purified:
        return best_losses, best_outputs
    else:
        return best_losses


def acgan_detect(
    conditional_generator,
    auxiliary_discriminator,
    inputs,
    labels,
    recon_thresh,
    d_thresh,
    z_dim,
    sole_decision_maker=None,
    purified_cache=None,
    **targeted_purify_kwargs,
):
    """
    A function that performs AC-GAN detection given a conditional generator,
    an auxiliary discriminator, inputs, labels, thresholds, and other optional
    arguments.

    Args:
        conditional_generator: The conditional generator model.
        auxiliary_discriminator: The auxiliary discriminator model.
        inputs: The input data.
        labels: The labels for the input data.
        recon_thresh: The threshold for the reconstruction loss.
        d_thresh: The threshold for the discriminator output.
        z_dim: The dimension of the latent space.
        sole_decision_maker: The sole decision maker for the detection.
        purified_cache: The cached purified inputs.
        **targeted_purify_kwargs: Other keyword arguments for targeted purification.

    Returns:
        The detection result as a boolean tensor.

    Raises:
        AssertionError: If the sole_decision_maker is invalid.

    """
    assert sole_decision_maker in [None, "d_thresh", "recon_thresh"]
    num_channels = inputs.shape[1]
    discriminator_normalizer = T.Normalize(
        mean=DISCRIMINATOR_MOMENTS[num_channels]["mean"],
        std=DISCRIMINATOR_MOMENTS[num_channels]["std"],
    )
    checks = torch.zeros((len(labels), 3), dtype=torch.bool)
    if recon_thresh is not None:
        if purified_cache is None:
            recon_loss = acgan_purify_targeted(
                conditional_generator,
                inputs,
                labels,
                z_dim,
                return_purified=False,
                **targeted_purify_kwargs,
            )
        else:
            recon_loss = purified_cache
        checks[:, 1] = recon_loss > recon_thresh
    if d_thresh is not None:
        dis_inputs = discriminator_normalizer(inputs)
        if auxiliary_discriminator.d_cond_mtd == "AC":
            d_result = auxiliary_discriminator.forward_aux(dis_inputs)
        else:
            d_result = auxiliary_discriminator.forward_emb(dis_inputs)
        veracity = F.sigmoid(d_result["adv_output"])
        label_output = torch.argmax(d_result["cls_output"], dim=1)
        checks[:, 0] = veracity < d_thresh
        if auxiliary_discriminator.aux_cls_type == "ADC":
            checks[:, 2] = label_output != 2 * labels
        else:
            checks[:, 2] = label_output != labels
    match sole_decision_maker:
        case None:
            return checks.sum(axis=1) >= 2
        case "d_thresh":
            return checks[:, 0]
        case "recon_thresh":
            return checks[:, 1]


def acgan_detect_v2(
    conditional_generator,
    auxiliary_discriminator,
    victim,
    clean_inputs,
    adv_norm_inputs,
    inverse_normalizer,
    z_dim,
    xgb_clf_saved_model_path,
    xgb_clf_saved_params_path,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    Generate the predictions of a trained XGBoost classifier on the generated samples.

    Args:
        conditional_generator (object): The trained conditional generator model.
        auxiliary_discriminator (object): The trained auxiliary discriminator model.
        victim (object): The trained victim model.
        clean_inputs (ndarray): The clean input samples.
        adv_norm_inputs (ndarray): The adversarial normalized input samples.
        inverse_normalizer (object): The inverse normalizer model.
        z_dim (int): The dimension of the latent space.
        xgb_clf_saved_model_path (str): The path to the saved XGBoost classifier model.
        xgb_clf_saved_params_path (str): The path to the saved XGBoost classifier model parameters.
        splits (int, optional): The number of splits to divide the data into. Defaults to 1.
        **targeted_purify_kwargs: Additional keyword arguments for the targeted purification process.

    Returns:
        ndarray: The predicted labels for the generated samples.
    """
    with open(xgb_clf_saved_params_path, "r") as params_fp:
        params = json.load(params_fp)
    xgb_clf = XGBClassifier(objective="binary:logistic", eval_metric="auc", **params)
    xgb_clf.load_model(xgb_clf_saved_model_path)
    X = acgan_generate_detection_X(
        conditional_generator,
        auxiliary_discriminator,
        victim,
        clean_inputs,
        adv_norm_inputs,
        inverse_normalizer,
        z_dim,
        splits,
        **targeted_purify_kwargs,
    )
    return xgb_clf.predict_proba(X)[:, 1]


@torch.inference_mode()
def diffusion_purify_targeted(
    conditional_diffuser,
    inputs,
    labels,
    noise_steps=200,
    cond_scale=1.5,
    return_purified=True,
    device="cuda",
    splits=1,
    disable_tqdm=True,
):
    """
    Applies diffusion purification to a targeted set of inputs.

    Args:
        conditional_diffuser (torch.nn.Module): The conditional diffuser model.
        inputs (torch.Tensor): The input data.
        labels (torch.Tensor): The target labels.
        noise_steps (int, optional): The number of noise steps. Default is 300.
        cond_scale (int, optional): The conditional scale. Default is 2.
        return_purified (bool, optional): Whether to return the purified data. Default is True.
        device (str, optional): The device to use. Default is "cuda".
        splits (int, optional): The number of splits for parallel processing. Default is 1.
        disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Default is True.

    Returns:
        tuple or torch.Tensor: If return_purified is True, returns a tuple of two torch.Tensor objects.
            The first tensor contains the losses for each input, and the second tensor contains the purified data.
            If return_purified is False, returns a single torch.Tensor object containing the losses.
    """
    assert (
        len(inputs) % splits == 0
    ), "Batch size must be divisible by the number of splits."
    noise_steps = torch.tensor([noise_steps], device=device).long()
    inputs = inputs.to(device)
    labels = labels.to(device)
    input_chunks = torch.chunk(inputs, chunks=splits)
    label_chunks = torch.chunk(labels, chunks=splits)
    losses, purified = [], []
    for input_chunk, label_chunk in tqdm(
        zip(input_chunks, label_chunks), disable=disable_tqdm
    ):
        x_t = conditional_diffuser.q_sample(input_chunk, noise_steps)
        x_recon = conditional_diffuser.p_sample_loop_from_x(
            x_t, label_chunk, noise_steps, cond_scale
        )
        loss = list(map(RECONSTRUCTION_LOSS, x_recon, inputs))
        losses += loss
        if return_purified:
            purified.append(x_recon)
    losses = torch.as_tensor(losses)
    if return_purified:
        purified = torch.cat(purified, dim=0)
        return losses, x_recon
    else:
        return losses


@torch.inference_mode()
def diffusion_purify_untargeted(
    conditional_diffuser,
    inputs,
    num_classes,
    noise_steps=75,
    cond_scale=0.5,
    return_purified=True,
    device="cuda",
):
    """
    diffusion_purify_untargeted is a function that performs diffusion purification on a batch of inputs in an untargeted manner.

    Parameters:
    - conditional_diffuser: The conditional diffuser object used for the purification process.
    - inputs: The batch of inputs to be purified.
    - num_classes: The number of target classes.
    - noise_steps: The number of noise steps for the purification process. Default is 200.
    - cond_scale: The scale factor for the conditional noise. Default is 1.0.
    - return_purified: A boolean value indicating whether to return the purified outputs. Default is True.
    - device: The device on which the purification process will be performed. Default is "cuda".

    Returns:
    - If return_purified is True, the function returns a tuple containing the best losses and the best purified outputs.
    - If return_purified is False, the function returns a list of the best losses.
    """
    noise_steps = torch.tensor([noise_steps], device=device).long()
    inputs = inputs.to(device)
    best_outputs, best_losses = [], []
    labels = torch.arange(0, num_classes).to(device)
    for i in tqdm(
        range(len(inputs)), desc="Diffusion Purify Untargeted: Calculating Losses"
    ):
        rep_input = inputs[i, ...].expand(num_classes, *[-1] * (inputs.ndim - 1))
        single_x_t = conditional_diffuser.q_sample(inputs[i], noise_steps)
        x_t = repeat(single_x_t, "... -> n ...", n=num_classes)
        x_recon = conditional_diffuser.p_sample_loop_from_x(
            x_t, labels, noise_steps, cond_scale
        )
        x_recon = HM(x_recon, rep_input)
        losses = list(map(RECONSTRUCTION_LOSS, x_recon, rep_input))
        best_arg = torch.argmin(torch.as_tensor(losses))
        best_losses.append(losses[best_arg].item())
        best_outputs.append(x_recon[best_arg])
        torch.cuda.empty_cache()
    if return_purified:
        return best_losses, best_outputs
    else:
        return best_losses


@torch.inference_mode()
def diffusion_detect(
    conditional_diffuser,
    auxiliary_discriminator,
    inputs,
    labels,
    recon_thresh,
    d_thresh,
    sole_decision_maker=None,
    purified_cache=None,
    **targeted_purify_kwargs,
):
    """
    Perform diffusion detection based on conditional diffuser and auxiliary discriminator.

    Args:
        conditional_diffuser: The conditional diffuser model.
        auxiliary_discriminator: The auxiliary discriminator model.
        inputs: The input data.
        labels: The labels for the input data.
        recon_thresh: The reconstruction threshold.
        d_thresh: The discrimination threshold.
        sole_decision_maker: The sole decision maker.
        purified_cache: The purified cache.
        **targeted_purify_kwargs: Additional keyword arguments for targeted purification.

    Returns:
        A boolean tensor indicating the detection result.
    """
    assert sole_decision_maker in [None, "d_thresh", "recon_thresh"]
    num_channels = inputs.shape[1]  # bchw
    discriminator_normalizer = T.Normalize(
        mean=DISCRIMINATOR_MOMENTS[num_channels]["mean"],
        std=DISCRIMINATOR_MOMENTS[num_channels]["std"],
    )
    checks = torch.zeros((len(labels), 3), dtype=torch.bool)
    if recon_thresh is not None:
        if purified_cache is None:
            recon_loss = diffusion_purify_targeted(
                conditional_diffuser,
                inputs,
                labels,
                return_purified=False,
                **targeted_purify_kwargs,
            )
        else:
            recon_loss = purified_cache
        checks[:, 1] = recon_loss > recon_thresh
    if d_thresh is not None:
        dis_inputs = discriminator_normalizer(inputs)
        if auxiliary_discriminator.d_cond_mtd == "AC":
            d_result = auxiliary_discriminator.forward_aux(dis_inputs)
        else:
            d_result = auxiliary_discriminator.forward_emb(dis_inputs)
        veracity = F.sigmoid(d_result["adv_output"])
        label_output = torch.argmax(d_result["cls_output"], dim=1)
        checks[:, 0] = veracity < d_thresh
        if auxiliary_discriminator.aux_cls_type == "ADC":
            checks[:, 2] = label_output != 2 * labels
        else:
            checks[:, 2] = label_output != labels
    match sole_decision_maker:
        case None:
            return checks.sum(axis=1) >= 2
        case "d_thresh":
            return checks[:, 0]
        case "recon_thresh":
            return checks[:, 1]


@torch.inference_mode()
def diffusion_detect_v2(
    conditional_diffuser,
    auxiliary_discriminator,
    victim,
    clean_inputs,
    adv_norm_inputs,
    inverse_normalizer,
    xgb_clf_saved_model_path,
    xgb_clf_saved_params_path,
    splits=1,
    **targeted_purify_kwargs,
):
    """
    Perform diffusion detection using a trained XGBoost classifier.

    Args:
        conditional_diffuser (ConditionalDiffuser): The conditional diffuser.
        auxiliary_discriminator (AuxiliaryDiscriminator): The auxiliary discriminator.
        victim (Victim): The victim model.
        clean_inputs (torch.Tensor): The clean inputs.
        adv_norm_inputs (torch.Tensor): The adversarial normalized inputs.
        inverse_normalizer (InverseNormalizer): The inverse normalizer.
        xgb_clf_saved_model_path (str): The path to the saved XGBoost classifier model.
        xgb_clf_saved_params_path (str): The path to the saved XGBoost classifier parameters.
        splits (int, optional): The number of splits for generating detection features. Defaults to 1.
        **targeted_purify_kwargs: Additional keyword arguments for targeted purification.

    Returns:
        torch.Tensor: The predicted labels from the XGBoost classifier.
    """
    with open(xgb_clf_saved_params_path, "r") as params_fp:
        params = json.load(params_fp)
    xgb_clf = XGBClassifier(objective="binary:logistic", eval_metric="auc", **params)
    xgb_clf.load_model(xgb_clf_saved_model_path)
    X = diffusion_generate_detection_X(
        conditional_diffuser,
        auxiliary_discriminator,
        victim,
        clean_inputs,
        adv_norm_inputs,
        inverse_normalizer,
        splits,
        **targeted_purify_kwargs,
    )
    return xgb_clf.predict_proba(X)[:, 1]
