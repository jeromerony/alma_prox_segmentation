from typing import Tuple

import torch
from adv_lib.utils import requires_grad_
from mmseg.apis import init_segmentor
from sacred import Ingredient
from torch import nn

from utils import MMSegNormalizer

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


@model_ingredient.named_config
def fcn_hrnetv2_w48():
    name = 'fcn_hrnetv2_w48'
    origin = 'mmseg'


@model_ingredient.named_config
def deeplabv3plus_resnet50():
    name = 'deeplabv3plus_resnet50'
    origin = 'mmseg'


@model_ingredient.named_config
def deeplabv3plus_resnet101():
    name = 'deeplabv3plus_resnet101'
    origin = 'mmseg'


@model_ingredient.named_config
def segformer_mitb0():
    name = 'segformer_mitb0'
    origin = 'mmseg'


@model_ingredient.named_config
def segformer_mitb3():
    name = 'segformer_mitb3'
    origin = 'mmseg'


_mmseg_configs_checkpoints = {
    'fcn_hrnetv2_w48': {
        'pascal_voc_2012': {
            'config': 'mmsegmentation/configs/hrnet/fcn_hr48_512x512_40k_voc12aug.py',
            'checkpoint': 'checkpoints/fcn_hr48_512x512_40k_voc12aug_20200613_222111-1b0f18bc.pth',
        },
        'cityscapes': {
            'config': 'mmsegmentation/configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py',
            'checkpoint': 'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth',
        }
    },
    'deeplabv3plus_resnet50': {
        'pascal_voc_2012': {
            'config': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py',
            'checkpoint': 'checkpoints/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth',
        },
        'cityscapes': {
            'config': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes.py',
            'checkpoint': 'checkpoints/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes_20201225_213645-a97e4e43.pth',
        },
    },
    'deeplabv3plus_resnet101': {
        'pascal_voc_2012': {
            'config': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py',
            'checkpoint': 'checkpoints/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth',
        },
    },
    'segformer_mitb0': {
        'cityscapes': {
            'config': 'mmsegmentation/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py',
            'checkpoint': 'checkpoints/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'
        }
    },
    'segformer_mitb3': {
        'cityscapes': {
            'config': 'mmsegmentation/configs/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes.py',
            'checkpoint': 'checkpoints/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth'
        }
    },
}


@model_ingredient.capture
def get_mmseg_model(dataset: str, name: str, device: torch.device,
                    mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    config_checkpoint = _mmseg_configs_checkpoints[name][dataset]
    segmentor = init_segmentor(**config_checkpoint, device=device)
    model = MMSegNormalizer(model=segmentor, mean=mean, std=std)
    return model


_model_getters = {
    'mmseg': get_mmseg_model,
}


@model_ingredient.capture
def get_model(dataset: str, device: torch.device, origin: str) -> nn.Module:
    model = _model_getters[origin](dataset=dataset, device=device)
    model.eval()
    model.to(device)
    requires_grad_(model, False)
    return model
