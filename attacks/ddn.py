import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.visdom_logger import VisdomLogger


def ddn(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        masks: Tensor = None,
        targeted: bool = False,
        adv_threshold: float = 0.99,
        steps: int = 100,
        γ: float = 0.05,
        init_norm: float = 1.,
        levels: Optional[int] = 256,
        callback: Optional[VisdomLogger] = None) -> Tensor:
    """Decoupled Direction and Norm attack from https://arxiv.org/abs/1811.09600."""
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

    # Init variables
    multiplier = -1 if targeted else 1
    δ = torch.zeros_like(inputs, requires_grad=True)
    ε = torch.full((batch_size,), init_norm, device=device, dtype=torch.float)
    worst_norm = torch.max(inputs, 1 - inputs).flatten(1).norm(p=2, dim=1)

    # Init trackers
    best_l2 = worst_norm.clone()
    best_adv = inputs.clone()
    best_adv_percent = torch.zeros_like(ε)
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(steps):
        α = torch.tensor(0.01 + (1 - 0.01) * (1 + math.cos(math.pi * i / steps)) / 2, device=device)

        l2 = δ.data.flatten(1).norm(p=2, dim=1)
        adv_inputs = inputs + δ
        logits = model(adv_inputs)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks

        pred_labels = logits.argmax(1)
        ce_loss = multiplier * F.cross_entropy(logits, labels_, reduction='none').masked_select(masks)
        δ_grad = grad(ce_loss.sum(), inputs=δ, only_inputs=True)[0]

        pixel_is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = l2 <= best_l2
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_better_adv, l2.detach(), best_l2)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        # renorming gradient
        grad_norms = δ_grad.flatten(1).norm(p=2, dim=1)
        δ_grad.div_(batch_view(grad_norms))
        # avoid nan or inf if gradient is 0
        if (zero_grad := (grad_norms < 1e-12)).any():
            δ_grad[zero_grad] = torch.randn_like(δ_grad[zero_grad])

        if callback is not None:
            attack_name = 'DDN'
            cosine = F.cosine_similarity(δ_grad.flatten(1), δ.data.flatten(1), dim=1).mean()
            callback.accumulate_line('ce', i, ce_loss.mean(), title=attack_name + ' - Loss')
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['ε', 'l2', 'best_l2'], i, [ε.mean(), l2.mean(), callback_best],
                                     title=attack_name + ' - Norms')
            callback.accumulate_line(['adv%', 'best_adv%', 'cosine', 'α'], i,
                                     [adv_percent.mean(), best_adv_percent.mean(), cosine, α],
                                     title=attack_name + 'APSR')

            if (i + 1) % (steps // 20) == 0 or (i + 1) == steps:
                callback.update_lines()

        δ.data.add_(δ_grad, alpha=α)

        ε = torch.where(is_adv, (1 - γ) * ε, (1 + γ) * ε)
        ε = torch.minimum(ε, worst_norm)

        δ.data.mul_(batch_view(ε / δ.data.flatten(1).norm(p=2, dim=1)))
        δ.data.add_(inputs).clamp_(0, 1)
        if levels is not None:
            δ.data.mul_(levels - 1).round_().div_(levels - 1)
        δ.data.sub_(inputs)

    return best_adv
