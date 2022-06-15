import os
from collections import OrderedDict
from pprint import pformat
from typing import Optional, Union

import torch
from PIL import Image
from adv_lib.distances.lp_norms import l1_distances, l2_distances, linf_distances
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torchvision.transforms.functional import pil_to_tensor

from attack_ingredient import attack_ingredient, get_attack
from attack_utils import run_attack
from dataset_ingredient import dataset_ingredient, get_dataset
from model_ingredient import get_model, model_ingredient

ex = Experiment('segmentation_attack', ingredients=[dataset_ingredient, model_ingredient, attack_ingredient])


@ex.config
def config():
    cpu = False  # force experiment to run on CPU
    save_adv = False  # save the adversarial images produced by the attack
    target = None  # specify a target as a png image or int
    cudnn_flag = 'benchmark'


@ex.named_config
def save_adv():
    save_adv = True


metrics = OrderedDict([
    ('linf', linf_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


@ex.automain
def main(cpu: bool,
         cudnn_flag: str,
         save_adv: bool,
         target: Optional[Union[int, str]],
         _config, _run, _log):
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    setattr(torch.backends.cudnn, cudnn_flag, True)

    loader, label_func = get_dataset()
    model = get_model(dataset=_config['dataset']['name'], device=device)
    attack, attack_name = get_attack()

    file_observers = [obs for obs in _run.observers if isinstance(obs, FileStorageObserver)]
    save_dir = file_observers[0].dir if len(file_observers) else None

    if isinstance(target, str):
        target_size = _config['dataset']['size']
        if not isinstance(target_size, (list, tuple)):
            target_size = (target_size, target_size)
        img_target = Image.open(target).resize(size=target_size[::-1], resample=Image.NEAREST)
        target = pil_to_tensor(label_func(img_target)).long().to(device)

    attack_data = run_attack(model=model, loader=loader, attack=attack, target=target, metrics=metrics,
                             return_adv=save_adv and save_dir is not None)

    if save_adv and save_dir is not None:
        dataset_name = _config['dataset']['name']
        model_name = _config['model']['name']
        torch.save(attack_data, os.path.join(save_dir, f'attack_data_{dataset_name}_{model_name}_{attack_name}.pt'))

    if 'images' in attack_data.keys():
        del attack_data['images'], attack_data['adv_images']
    _run.info = attack_data

    _log.info(pformat(attack_data))
