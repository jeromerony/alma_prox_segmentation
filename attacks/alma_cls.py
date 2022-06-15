from functools import partial
from typing import Callable, Optional

import torch
from adv_lib.attacks.augmented_lagrangian import _distances, all_penalties, difference_of_logits_ratio, init_lr_finder
from adv_lib.utils.visdom_logger import VisdomLogger
from torch import Tensor, nn
from torch.autograd import grad


def alma(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         masks: Tensor = None,
         targeted: bool = False,
         adv_threshold: float = 0.99,
         penalty: Callable = all_penalties['P2'],
         num_steps: int = 1000,
         lr_init: float = 0.1,
         lr_reduction: float = 0.01,
         distance: str = 'l2',
         init_lr_distance: Optional[float] = None,
         μ_init: float = 1e-4,
         ρ_init: float = 1,
         check_steps: int = 10,
         τ: float = 0.95,
         γ: float = 1.2,
         α: float = 0.9,
         α_rms: Optional[float] = None,
         momentum: Optional[float] = None,
         logit_tolerance: float = 1e-4,
         constraint_masking: bool = False,
         mask_decay: bool = False,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    attack_name = f'ALMA cls {distance}'
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    α_rms = α if α_rms is None else α_rms

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    square_avg = torch.ones_like(inputs)
    momentum_buffer = torch.zeros_like(inputs)
    lr = torch.full((batch_size,), lr_init, device=device, dtype=torch.float)
    α_rms, momentum = α_rms or α, momentum or α
    lower, upper = -inputs, 1 - inputs

    # Init rho and mu
    μ = torch.full_like(labels, μ_init, device=device, dtype=torch.float)
    ρ = torch.full_like(labels, ρ_init, device=device, dtype=torch.float)

    # Init similarity metric
    if distance in ['lpips']:
        dist_func = _distances[distance](target=inputs)
    else:
        dist_func = partial(_distances[distance], inputs)

    # Init trackers
    best_dist = torch.full_like(lr, float('inf'))
    best_adv_percent = torch.zeros_like(lr)
    adv_found = torch.zeros_like(lr, dtype=torch.bool)
    best_adv = inputs.clone()
    pixel_adv_found = torch.zeros_like(labels, dtype=torch.bool)
    step_found = torch.full_like(lr, num_steps // 2)

    for i in range(num_steps):

        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        dist = dist_func(adv_inputs)

        if i == 0:
            # initialize variables based on model's output
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks
            masks_inf = torch.zeros_like(masks, dtype=torch.float).masked_fill_(~masks, float('inf'))
            labels_infhot = torch.zeros_like(logits.detach()).scatter_(1, labels_.unsqueeze(1), float('inf'))
            diff_func = partial(difference_of_logits_ratio, labels=labels_, labels_infhot=labels_infhot,
                                targeted=targeted, ε=logit_tolerance)
            k = ((1 - adv_threshold) * masks_sum).long()  # number of constraints that can be violated
            constraint_mask = masks

        # track progress
        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        pixel_adv_found.logical_or_(pixel_is_adv)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = dist <= best_dist
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        step_found.masked_fill_((~adv_found) & is_adv, i)
        adv_found.logical_or_(is_adv)
        best_dist = torch.where(is_better_adv, dist.detach(), best_dist)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        dlr = multiplier * diff_func(logits)

        if constraint_masking:
            if mask_decay:
                k = ((1 - adv_threshold) * masks_sum).mul_(i / (num_steps - 1)).long()
            if k.any():
                top_constraints = dlr.detach().sub(masks_inf).flatten(1).topk(k=k.max()).values
                ξ = top_constraints.gather(1, k.unsqueeze(1) - 1).squeeze(1)
                constraint_mask = masks & (dlr <= ξ.view(-1, 1, 1))

        # adjust constraint parameters
        if i == 0:
            prev_dlr = dlr.detach()
        elif (i + 1) % check_steps == 0:
            improved_constraint = (dlr.detach() * constraint_mask <= τ * prev_dlr)
            ρ = torch.where(~(pixel_adv_found | improved_constraint), γ * ρ, ρ)
            prev_dlr = dlr.detach()
            pixel_adv_found.fill_(False)

        if i:
            new_μ = grad(penalty(dlr, ρ, μ)[constraint_mask].sum(), dlr, only_inputs=True)[0]
            μ.mul_(α).add_(new_μ, alpha=1 - α).clamp_(min=1e-12, max=1)

        loss = dist + (penalty(dlr, ρ, μ) * constraint_mask).flatten(1).sum(dim=1)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        grad_norm = δ_grad.flatten(1).norm(p=2, dim=1)
        if init_lr_distance is not None and i == 0:
            randn_grad = torch.randn_like(δ_grad).renorm(dim=0, p=2, maxnorm=1)
            δ_grad = torch.where(batch_view(grad_norm <= 1e-6), randn_grad, δ_grad)
            lr = init_lr_finder(inputs, δ_grad, dist_func, target_distance=init_lr_distance)

        exp_decay = lr_reduction ** ((i - step_found).clamp_(min=0) / (num_steps - step_found))
        step_lr = lr * exp_decay
        square_avg.mul_(α_rms).addcmul_(δ_grad, δ_grad, value=1 - α_rms)
        momentum_buffer.mul_(momentum).addcdiv_(δ_grad, square_avg.sqrt().add_(1e-8))
        δ.data.addcmul_(momentum_buffer, batch_view(step_lr), value=-1)

        δ.data.clamp_(min=lower, max=upper)

        if callback:
            callback.accumulate_line('dlr', i, dlr.mean(), title=attack_name + ' - Constraints')
            callback.accumulate_line(['μ_c', 'ρ_c'], i, [μ.mean(), ρ.mean()],
                                     title=attack_name + ' - Penalty parameters', ytype='log')
            callback.accumulate_line('||g||₂', i, grad_norm.mean(),
                                     title=attack_name + ' - Grad norm', ytype='log')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=attack_name + ' - APSR')
            callback.accumulate_line([distance, f'best {distance}'], i,
                                     [dist.mean(), best_dist.mean()], title=attack_name + ' - Distances')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
