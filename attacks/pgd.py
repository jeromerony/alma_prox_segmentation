from functools import partial
from typing import Optional, Tuple, Union

import torch
from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F


def pgd(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        ε: Union[float, Tensor],
        masks: Tensor = None,
        targeted: bool = False,
        norm: float = float('inf'),
        num_steps: int = 40,
        random_init: bool = False,
        restarts: int = 1,
        loss_function: str = 'ce',
        relative_step_size: float = 0.1 / 3,
        absolute_step_size: Optional[float] = None) -> Tensor:
    if isinstance(ε, (int, float)):
        ε = torch.full((len(inputs),), ε, dtype=torch.float, device=inputs.device)

    adv_inputs = inputs.clone()
    adv_percent = torch.zeros_like(ε)
    pgd_attack = partial(_pgd, model=model, ε=ε, targeted=targeted, masks=masks, norm=norm, num_steps=num_steps,
                         loss_function=loss_function, relative_step_size=relative_step_size,
                         absolute_step_size=absolute_step_size)

    for i in range(restarts):
        adv_percent_run, adv_inputs_run = pgd_attack(inputs=inputs, labels=labels, random_init=random_init or (i != 0))
        better_adv = adv_percent_run >= adv_percent
        adv_inputs[better_adv] = adv_inputs_run
        adv_percent[better_adv] = adv_percent_run

    return adv_inputs


def _pgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         ε: Tensor,
         masks: Tensor = None,
         targeted: bool = False,
         norm: float = float('inf'),
         num_steps: int = 40,
         random_init: bool = False,
         loss_function: str = 'ce',
         relative_step_size: float = 0.1 / 3,
         absolute_step_size: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    _loss_functions = {
        'ce': (partial(F.cross_entropy, reduction='none'), 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
    }
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    neg_inputs = -inputs
    one_minus_inputs = 1 - inputs

    loss_func, multiplier = _loss_functions[loss_function.lower()]
    if targeted:
        multiplier *= -1

    if absolute_step_size is not None:
        step_size = torch.full((len(inputs),), absolute_step_size, dtype=torch.float, device=inputs.device)
    else:
        step_size = ε * relative_step_size

    δ = torch.zeros_like(inputs, requires_grad=True)
    best_percent = torch.zeros(batch_size, device=device)
    best_adv = inputs.clone()

    if random_init:
        if norm == float('inf'):
            δ.data.uniform_().sub_(0.5).mul_(2 * batch_view(ε))
        elif norm == 2:
            δ.data.normal_()
            δ.data.mul_(batch_view(ε / δ.data.flatten(1).norm(p=2, dim=1)))
        δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

    for i in range(num_steps):
        adv_inputs = inputs + δ
        logits = model(inputs + δ)

        if i == 0:
            if masks is None:
                num_classes = logits.size(1)
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks

            if loss_function.lower() in ['dl', 'dlr']:
                labels_infhot = torch.zeros_like(logits).scatter(1, labels_.unsqueeze(1), float('inf'))
                loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, labels_).masked_select(masks)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        adv_percent = (pixel_is_adv.float() * masks).flatten(1).sum(dim=1) / masks_sum
        is_better = adv_percent >= best_percent
        best_percent = torch.where(is_better, adv_percent, best_percent)
        best_adv = torch.where(batch_view(is_better), adv_inputs.detach(), best_adv)

        if norm == float('inf'):
            δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
        elif norm == 2:
            δ_grad.div_(δ_grad.flatten(1).norm(p=2, dim=1).clamp_min_(1e-6))
            δ.data.add_(batch_view(step_size) * δ_grad)
            δ_norm = δ.data.flatten(1).norm(p=2, dim=1)
            δ.data.mul_(batch_view(ε / torch.where(δ_norm > ε, δ_norm, ε)))
        δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

    return best_percent, best_adv


def minimal_pgd(model: nn.Module,
                inputs: Tensor,
                labels: Tensor,
                max_ε: float,
                masks: Tensor = None,
                targeted: bool = False,
                adv_threshold: float = 0.99,
                binary_search_steps: int = 20, **kwargs) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    best_ε = torch.full((batch_size,), 2 * max_ε, dtype=torch.float, device=device)
    ε_low = torch.zeros_like(best_ε)

    attack = partial(pgd, inputs=inputs, labels=labels, model=model, targeted=targeted, masks=masks, **kwargs)

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
