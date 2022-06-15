from functools import partial
from typing import Callable, Union

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F


def fgsm(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         ε: Union[float, Tensor],
         masks: Tensor = None,
         targeted: bool = False,) -> Tensor:
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    if isinstance(ε, (int, float)):
        ε = torch.full((batch_size,), ε, dtype=torch.float, device=inputs.device)

    inputs.requires_grad_(True)
    logits = model(inputs)

    if masks is None:
        num_classes = logits.size(1)
        masks = labels < num_classes
    cross_entropy = multiplier * F.cross_entropy(logits, labels * masks, reduction='none').masked_select(masks)
    δ = grad(cross_entropy.sum(), inputs=inputs, only_inputs=True)[0].sign()

    inputs.detach_()
    adv_inputs = inputs + batch_view(ε) * δ
    adv_inputs.clamp_(0, 1)

    return adv_inputs


def mifgsm(model: nn.Module,
           inputs: Tensor,
           labels: Tensor,
           ε: Union[float, Tensor],
           masks: Tensor = None,
           targeted: bool = False,
           num_steps: int = 20,
           μ: float = 1,
           q: float = 1) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    neg_inputs = -inputs
    one_minus_inputs = 1 - inputs
    if isinstance(ε, (int, float)):
        ε = torch.full((batch_size,), ε, dtype=torch.float, device=device)

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    g = torch.zeros_like(inputs)

    # Init trackers
    best_percent = torch.zeros(batch_size, device=device)
    best_adv = inputs.clone()

    α = ε / num_steps
    for i in range(num_steps):
        adv_inputs = inputs + δ
        logits = model(adv_inputs)

        if i == 0:
            if masks is None:
                num_classes = logits.size(1)
                masks = labels < num_classes
            labels_ = labels * masks

        pred_labels = logits.argmax(dim=1)
        loss = multiplier * F.cross_entropy(logits, labels_, reduction='none').masked_select(masks)
        δ_grad = grad(loss.sum(), inputs=δ, only_inputs=True)[0]

        if μ == 0:
            g = δ_grad
        else:
            grad_norm = δ_grad.flatten(1).norm(p=q, dim=1)
            g.mul_(μ).add_(δ_grad / batch_view(grad_norm))

        pixel_is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        adv_percent = (pixel_is_adv.float() * masks).flatten(1).sum(dim=1) / masks.flatten(1).sum(dim=1)
        is_better = adv_percent >= best_percent
        best_percent = torch.where(is_better, adv_percent, best_percent)
        best_adv = torch.where(batch_view(is_better), adv_inputs.detach(), best_adv)

        δ.data.add_(batch_view(α) * g.sign())
        δ.data.clamp_(min=-batch_view(ε), max=batch_view(ε)).clamp_(min=neg_inputs, max=one_minus_inputs)

    return best_adv


def fgm(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        ε: Union[float, Tensor],
        masks: Tensor = None,
        targeted: bool = False) -> Tensor:
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    if isinstance(ε, (int, float)):
        ε = torch.full((batch_size,), ε, dtype=torch.float, device=inputs.device)

    inputs.requires_grad_(True)
    logits = model(inputs)

    if masks is None:
        num_classes = logits.size(1)
        masks = labels < num_classes
    cross_entropy = multiplier * F.cross_entropy(logits, labels * masks, reduction='none').masked_select(masks)
    δ = grad(cross_entropy.sum(), inputs=inputs, only_inputs=True)[0]
    δ.div_(batch_view(δ.flatten(1).norm(p=2, dim=1).clamp_min(1e-6)))

    inputs.detach_()
    adv_inputs = inputs + batch_view(ε) * δ
    adv_inputs.clamp_(0, 1)

    return adv_inputs


def minimal_fast_gradient(model: nn.Module,
                          inputs: Tensor,
                          labels: Tensor,
                          max_ε: float,
                          attack: Callable,
                          masks: Tensor = None,
                          targeted: bool = False,
                          adv_threshold: float = 0.99,
                          binary_search_steps: int = 20) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    best_ε = torch.full((batch_size,), 2 * max_ε, dtype=torch.float, device=device)
    ε_low = torch.zeros_like(best_ε)

    attack = partial(attack, inputs=inputs, labels=labels, model=model, targeted=targeted, masks=masks)

    for i in range(binary_search_steps):
        ε = (ε_low + best_ε) / 2

        adv_inputs_run = attack(ε=ε)
        logits = model(adv_inputs_run)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)

        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum

        better_adv = (adv_percent >= adv_threshold) & (ε < best_ε)
        adv_inputs[better_adv] = adv_inputs_run[better_adv]

        ε_low = torch.where(better_adv, ε_low, ε)
        best_ε = torch.where(better_adv, ε, best_ε)

    return adv_inputs
