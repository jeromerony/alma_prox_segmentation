from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from mmseg.models import EncoderDecoder
from torch import Tensor, nn
from torchvision.transforms import functional as F

cityscapes_label_ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image.float() / 255, target


class ImageNormalizer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float],
                 key: str = None) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))
        self.model = model
        self.key = key

    def forward(self, inputs: Tensor) -> Tensor:
        normalized_inputs = (inputs - self.mean) / self.std
        out = self.model(normalized_inputs)
        if self.key is not None:
            out = out[self.key]
        return out


class MMSegNormalizer(nn.Module):
    def __init__(self,
                 model: EncoderDecoder,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(MMSegNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))
        self.model = model
        self.logit_func = model.slide_inference if model.test_cfg['mode'] == 'slide' else model.whole_inference

    def forward(self, inputs: Tensor) -> Tensor:
        normalized_inputs = (inputs - self.mean) / self.std
        ori_shape = inputs.shape[-2:] + (inputs.shape[1],)
        out = self.logit_func(img=normalized_inputs, img_meta=[{'ori_shape': ori_shape, 'flip': False}], rescale=True)
        return out


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, accs, ious = self.compute()
        return ('Row correct: ' + '|'.join(f'{acc:>5.2%}' for acc in accs.tolist()) + '\n'
                                                                                      f'IoUs       : ' + '|'.join(
            f'{iou:>5.2%}' for iou in ious.tolist()) + '\n'
                                                       f'Pixel Acc. : {acc_global.item():.2%}\n'
                                                       f'mIoU       : {ious.nanmean().item():.2%}')


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def pred_to_image(prediction: Tensor, cmap: Optional[Tensor] = None) -> Tensor:
    if cmap is None:
        cmap = torch.as_tensor(color_map(normalized=True), dtype=torch.float, device=prediction.device)
    pred_image = cmap[prediction].movedim(-1, -3)
    return pred_image


def cityscapes_lut(ignore_index: int = 255):
    label_lut = [ignore_index for _ in range(256)]
    for i, label in enumerate(cityscapes_label_ids):
        label_lut[label] = i
    return label_lut


def label_map_cityscapes(img: Image.Image, ignore_index: int = 255) -> Image.Image:
    lut = cityscapes_lut(ignore_index=ignore_index)
    return img.point(lut)
