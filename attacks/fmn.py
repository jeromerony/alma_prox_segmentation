# Adapted from https://github.com/maurapintor/Fast-Minimum-Norm-FMN-Attack

import math
from functools import partial
from typing import Optional

import torch
from adv_lib.utils.visdom_logger import VisdomLogger
from torch import nn, Tensor
from torch.autograd import grad

from adv_lib.utils.losses import difference_of_logits
from adv_lib.attacks.fast_minimum_norm import (
    l0_projection_, l0_mid_points,
    l1_projection_, l1_mid_points,
    l2_projection_, l2_mid_points,
    linf_projection_, linf_mid_points
)


def fmn(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        norm: float,
        targeted: bool = False,
        masks: Tensor = None,
        adv_threshold: float = 0.99,
        steps: int = 10,
        α_init: float = 1.0,
        α_final: Optional[float] = None,
        γ_init: float = 0.05,
        γ_final: float = 0.001,
        callback: Optional[VisdomLogger] = None) -> Tensor:
    """Fast Minimum-Norm attack from https://arxiv.org/abs/2102.12827."""
    _dual_projection_mid_points = {
        0: (None, l0_projection_, l0_mid_points),
        1: (float('inf'), l1_projection_, l1_mid_points),
        2: (2, l2_projection_, l2_mid_points),
        float('inf'): (1, linf_projection_, linf_mid_points),
    }
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    dual, projection, mid_point = _dual_projection_mid_points[norm]
    α_final = α_final or α_init / 100
    multiplier = 1 if targeted else -1

    # If starting_points is provided, search for the boundary
    δ = torch.zeros_like(inputs, requires_grad=True)

    if norm == 0:
        ε = torch.ones(batch_size, device=device)
    else:
        ε = torch.full((batch_size,), float('inf'), device=device)

    # Init trackers
    worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=norm, dim=1)
    best_norm = worst_norm.clone()
    best_adv = inputs.clone()
    best_adv_percent = torch.zeros_like(ε)
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(steps):
        cosine = (1 + math.cos(math.pi * i / steps)) / 2
        α = α_final + (α_init - α_final) * cosine
        γ = γ_final + (γ_init - γ_final) * cosine

        δ_norm = δ.data.flatten(1).norm(p=norm, dim=1)
        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        pred_labels = logits.argmax(dim=1)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks

            labels_infhot = torch.zeros_like(logits).scatter_(1, labels_.unsqueeze(1), float('inf'))
            logit_diff_func = partial(difference_of_logits, labels=labels_, labels_infhot=labels_infhot)

        logit_diffs = logit_diff_func(logits=logits).mul(masks).flatten(1).sum(dim=1) / masks_sum
        loss = multiplier * logit_diffs
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        pixel_is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = δ_norm <= best_norm
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        adv_found.logical_or_(is_adv)
        best_norm = torch.where(is_better_adv, δ_norm, best_norm)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        if norm == 0:
            ε = torch.where(is_adv,
                            torch.minimum(torch.minimum(ε - 1, (ε * (1 - γ)).floor_()), best_norm),
                            torch.maximum(ε + 1, (ε * (1 + γ)).floor_()))
            ε.clamp_min_(0)
        else:
            distance_to_boundary = loss.detach().abs() / δ_grad.flatten(1).norm(p=dual, dim=1).clamp_min_(1e-12)
            ε = torch.where(is_adv,
                            torch.minimum(ε * (1 - γ), best_norm),
                            torch.where(adv_found, ε * (1 + γ), δ_norm + distance_to_boundary))

        # clip ε
        ε = torch.minimum(ε, worst_norm)

        # normalize gradient
        grad_l2_norms = δ_grad.flatten(1).norm(p=2, dim=1).clamp_min_(1e-12)
        δ_grad.div_(batch_view(grad_l2_norms))

        # gradient ascent step
        δ.data.add_(δ_grad, alpha=α)

        # project in place
        projection(δ=δ.data, ε=ε)

        # clamp
        δ.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        if callback is not None:
            attack_name = f'FMN L{norm}'
            callback.accumulate_line('loss', i, loss.mean(), title=attack_name + ' - Loss')
            callback.accumulate_line(['ε', f'l{norm}', f'best_l{norm}'], i, [ε.mean(), δ_norm.mean(), best_norm.mean()],
                                     title=attack_name + ' - Norms')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=attack_name + ' - APSR')

            if (i + 1) % (steps // 20) == 0 or (i + 1) == steps:
                callback.update_lines()

    return best_adv
