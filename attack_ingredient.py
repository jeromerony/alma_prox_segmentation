from functools import partial
from typing import Optional, Tuple

from adv_lib.attacks.segmentation import (
    alma_prox as alma_prox_seg,
    asma as asma_seg,
    dag as dag_seg,
    pdgd as pdgd_seg,
    pdpgd as pdpgd_seg
)
from sacred import Ingredient

from attacks.alma_cls import alma as alma_cls_seg
from attacks.ddn import ddn as ddn_seg
from attacks.fast_gradient import (
    fgm as fgm_seg,
    fgsm as fgsm_seg,
    mifgsm as mifgsm_seg,
    minimal_fast_gradient as minimal_fast_gradient_seg
)
from attacks.fmn import fmn as fmn_seg
from attacks.pgd import minimal_pgd, pgd

attack_ingredient = Ingredient('attack')


@attack_ingredient.config
def config():
    pass


@attack_ingredient.named_config
def alma_cls():
    name = 'alma_cls'
    distance = 'l2'
    num_steps = 500
    lr_reduction = 0.1
    init_lr_distance = 1
    alpha = 0.8


@attack_ingredient.capture
def get_alma_cls(distance: str, num_steps: int, alpha: float, lr_reduction: float, init_lr_distance: float,
                 _log, rho_init: float = 1, constraint_masking: bool = True, mask_decay: bool = True):
    attack = partial(alma_cls_seg, distance=distance, num_steps=num_steps, α=alpha, lr_reduction=lr_reduction,
                     constraint_masking=constraint_masking, mask_decay=mask_decay,
                     init_lr_distance=init_lr_distance, ρ_init=rho_init)
    name = f'ALMA_cls_{distance.upper()}'
    return attack, name


@attack_ingredient.named_config
def alma_prox_l1():
    name = 'alma_prox'
    norm = 1
    num_steps = 500
    lr_reduction = 0.1
    init_lr_distance = 1000
    alpha = 0.8
    rho_init = 1
    scale_min = 0.05
    scale_max = 1
    scale_init = None


@attack_ingredient.named_config
def alma_prox_l2():
    name = 'alma_prox'
    norm = 2
    num_steps = 500
    lr_reduction = 0.1
    init_lr_distance = 1
    alpha = 0.8
    rho_init = 1
    scale_min = 0.05
    scale_max = 1
    scale_init = None


@attack_ingredient.named_config
def alma_prox_linf():
    name = 'alma_prox'
    norm = float('inf')
    num_steps = 500
    lr_reduction = 0.1
    init_lr_distance = 16  # this will be divided by 255
    alpha = 0.8
    rho_init = 0.01
    scale_min = 0.05
    scale_max = 1
    scale_init = None


@attack_ingredient.capture
def get_alma_prox(norm: float, num_steps: int, alpha: float, lr_reduction: float, init_lr_distance: float,
                  scale_min: float, scale_max: float, _log, rho_init: float = 1, mu_init: float = 1e-4,
                  scale_init: Optional[float] = None, constraint_masking: bool = True, mask_decay: bool = True):
    if norm == float('inf'):
        _log.warning('Divided init_lr_distance by 255')
        init_lr_distance = init_lr_distance / 255
    attack = partial(alma_prox_seg, norm=norm, num_steps=num_steps, α=alpha, lr_reduction=lr_reduction, ρ_init=rho_init,
                     μ_init=mu_init, init_lr_distance=init_lr_distance, scale_min=scale_min, scale_max=scale_max,
                     scale_init=scale_init, constraint_masking=constraint_masking, mask_decay=mask_decay)
    name = f'ALMA_prox_L{norm}_{num_steps}'
    return attack, name


@attack_ingredient.named_config
def asma():
    name = 'asma'
    num_steps = 500
    tau = 1e-7
    beta = 1e-6


@attack_ingredient.capture
def get_asma(num_steps: int, tau: float, beta: float):
    attack = partial(asma_seg, num_steps=num_steps, τ=tau, β=beta)
    name = 'ASMA'
    return attack, name


@attack_ingredient.named_config
def dag():
    name = 'dag'
    max_iter = 500
    gamma = 0.5


@attack_ingredient.capture
def get_dag(max_iter: int, gamma: float, p: float = float('inf')):
    attack = partial(dag_seg, max_iter=max_iter, γ=gamma, p=p)
    name = 'DAG'
    return attack, name


@attack_ingredient.named_config
def ddn():
    name = 'ddn'
    steps = 500
    gamma = 0.05


@attack_ingredient.capture
def get_ddn(steps: int, gamma: float):
    attack = partial(ddn_seg, steps=steps, γ=gamma)
    name = 'DDN'
    return attack, name


@attack_ingredient.named_config
def fmn_l1():
    name = 'fmn'
    norm = 1
    steps = 500
    alpha_init = 1


@attack_ingredient.named_config
def fmn_l2():
    name = 'fmn'
    norm = 2
    steps = 500
    alpha_init = 1


@attack_ingredient.named_config
def fmn_linf():
    name = 'fmn'
    norm = float('inf')
    steps = 500
    alpha_init = 10


@attack_ingredient.capture
def get_fmn(norm: float, steps: int, alpha_init: float):
    attack = partial(fmn_seg, steps=steps, norm=norm, α_init=alpha_init)
    name = f'FMN_L{norm}'
    return attack, name


@attack_ingredient.named_config
def fgm():
    name = 'fast_gradient'
    norm = 2
    epsilon = 0.5


@attack_ingredient.named_config
def fgsm():
    name = 'fast_gradient'
    norm = float('inf')
    epsilon = 1  # this will be divided by 255


_fast_gradient_attacks = {
    2: fgm_seg,
    float('inf'): fgsm_seg
}


@attack_ingredient.capture
def get_fast_gradient(norm: float, epsilon: float, _log):
    if norm == float('inf'):
        epsilon = epsilon / 255
        _log.warning('Divided epsilon by 255')
    attack = partial(_fast_gradient_attacks[norm], ε=epsilon)
    name = f'FGSM_eps={epsilon * 255:.2f}' if norm == float('inf') else f'FGM_eps={epsilon:.2f}'
    return attack, name


@attack_ingredient.capture
def get_minimal_mifgsm(max_eps: float, binary_search_steps: int, num_steps: int, mu: float):
    attack = partial(minimal_fast_gradient_seg, max_ε=max_eps, attack=partial(mifgsm_seg, num_steps=num_steps, μ=mu),
                     binary_search_steps=binary_search_steps)
    name = 'minimal_FGSM'
    return attack, name


@attack_ingredient.named_config
def mifgsm():
    name = 'mifgsm'
    epsilon = 1  # this will be divided by 255
    num_steps = 20
    mu = 1


@attack_ingredient.capture
def get_mifgsm(epsilon: float, num_steps: int, mu: float, _log):
    epsilon = epsilon / 255
    _log.warning('Divided epsilon by 255')
    attack = partial(mifgsm_seg, ε=epsilon, num_steps=num_steps, μ=mu)
    name = f'MI-FGSM_eps={epsilon * 255:.2f}'
    return attack, name


@attack_ingredient.named_config
def minimal_fgm():
    name = 'minimal_fast_gradient'
    norm = 2
    max_eps = 80
    binary_search_steps = 13


@attack_ingredient.named_config
def minimal_fgsm():
    name = 'minimal_fast_gradient'
    norm = float('inf')
    max_eps = 1
    binary_search_steps = 13


@attack_ingredient.named_config
def minimal_mifgsm():
    name = 'minimal_mifgsm'
    max_eps = 1
    binary_search_steps = 13
    num_steps = 20
    mu = 1


@attack_ingredient.capture
def get_minimal_fast_gradient(norm: float, max_eps: float, binary_search_steps: int):
    attack = partial(minimal_fast_gradient_seg, max_ε=max_eps, attack=_fast_gradient_attacks[norm],
                     binary_search_steps=binary_search_steps)
    name = 'minimal_' + ('FGSM' if norm == float('inf') else 'FGM')
    return attack, name


@attack_ingredient.named_config
def pgd_l2():
    name = 'pgd'
    norm = 2
    epsilon = 1
    num_steps = 40
    random_init = False
    restarts = 1
    loss = 'ce'
    relative_step_size = 0.1 / 3
    absolute_step_size = None


@attack_ingredient.named_config
def pgd_linf():
    name = 'pgd'
    norm = float('inf')
    epsilon = 1  # this will be divided by 255
    num_steps = 40
    random_init = False
    restarts = 1
    loss = 'ce'
    relative_step_size = 0.1 / 3
    absolute_step_size = None


@attack_ingredient.capture
def get_pgd(norm: float, epsilon: float, num_steps: int, random_init: bool, restarts: int, loss: str,
            relative_step_size: float, absolute_step_size: Optional[float], _log):
    if norm == float('inf'):
        epsilon = epsilon / 255
        _log.warning('Divided epsilon by 255')
    attack = partial(pgd, norm=norm, ε=epsilon, num_steps=num_steps, random_init=random_init, restarts=restarts,
                     loss_function=loss, relative_step_size=relative_step_size, absolute_step_size=absolute_step_size)
    name = f'PGD_L{norm}'
    return attack, name


@attack_ingredient.named_config
def minimal_pgd_l2():
    name = 'minimal_pgd'
    norm = 2
    max_eps = 80
    binary_search_steps = 13
    num_steps = 40
    random_init = False
    restarts = 1
    loss = 'ce'
    relative_step_size = 0.1 / 3
    absolute_step_size = None


@attack_ingredient.named_config
def minimal_pgd_linf():
    name = 'minimal_pgd'
    norm = float('inf')
    max_eps = 1
    binary_search_steps = 13
    num_steps = 40
    random_init = False
    restarts = 1
    loss = 'ce'
    relative_step_size = 0.1 / 3
    absolute_step_size = None


@attack_ingredient.capture
def get_minimal_pgd(max_eps: float, binary_search_steps: int, norm: float, num_steps: int, random_init: bool,
                    restarts: int, loss: str, relative_step_size: float, absolute_step_size: Optional[float]):
    attack = partial(minimal_pgd, max_ε=max_eps, binary_search_steps=binary_search_steps, norm=norm,
                     num_steps=num_steps, random_init=random_init, restarts=restarts, loss_function=loss,
                     relative_step_size=relative_step_size, absolute_step_size=absolute_step_size)
    name = f'minimal_PGD_L{norm}'
    return attack, name


@attack_ingredient.named_config
def pdgd():
    name = 'pdgd'
    steps = 500
    primal_lr = 0.01
    dual_ratio_init = 1


@attack_ingredient.capture
def get_pdgd(steps: int, primal_lr: float, dual_ratio_init: float,
             constraint_masking: bool = False, mask_decay: bool = False):
    attack = partial(pdgd_seg, num_steps=steps, primal_lr=primal_lr, dual_ratio_init=dual_ratio_init,
                     constraint_masking=constraint_masking, mask_decay=mask_decay)
    name = 'PDGD'
    return attack, name


@attack_ingredient.named_config
def pdpgd():
    name = 'pdpgd'
    norm = float('inf')
    steps = 500
    primal_lr = 0.01
    dual_ratio_init = 1


@attack_ingredient.capture
def get_pdpgd(norm: float, steps: int, primal_lr: float, dual_ratio_init: float,
              constraint_masking: bool = False, mask_decay: bool = False):
    attack = partial(pdpgd_seg, norm=norm, num_steps=steps, primal_lr=primal_lr, dual_ratio_init=dual_ratio_init,
                     constraint_masking=constraint_masking, mask_decay=mask_decay)
    name = f'PDPGD_L{norm}'
    return attack, name


_attacks = {
    'alma_cls': get_alma_cls,
    'alma_prox': get_alma_prox,
    'asma': get_asma,
    'dag': get_dag,
    'ddn': get_ddn,
    'fast_gradient': get_fast_gradient,
    'fmn': get_fmn,
    'mifgsm': get_mifgsm,
    'minimal_fast_gradient': get_minimal_fast_gradient,
    'minimal_mifgsm': get_minimal_mifgsm,
    'minimal_pgd': get_minimal_pgd,
    'pdgd': get_pdgd,
    'pdpgd': get_pdpgd,
    'pgd': get_pgd,
}


@attack_ingredient.capture
def get_attack(name: str) -> Tuple[partial, str]:
    return _attacks[name]()
