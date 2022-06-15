import warnings
from distutils.version import LooseVersion
from typing import Callable, Dict, Optional, Union

import torch
from adv_lib.utils import BackwardCounter, ForwardCounter
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import ConfusionMatrix


def run_attack(model: nn.Module,
               loader: DataLoader,
               attack: Callable,
               target: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               return_adv: bool = False) -> dict:
    device = next(model.parameters()).device
    targeted = True if target is not None else False
    loader_length = len(loader)
    image_list = getattr(loader.sampler.data_source, 'dataset', loader.sampler.data_source).images

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, apsrs, apsrs_orig = [], [], [], []
    distances = {k: [] for k in metrics.keys()}

    if return_adv:
        images, adv_images = [], []

    for i, (image, label) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        if return_adv:
            images.append(image.clone())

        image, label = image.to(device), label.to(device).squeeze(1).long()
        if targeted:
            if isinstance(target, Tensor):
                attack_label = target.to(device).expand(image.shape[0], -1, -1)
            elif isinstance(target, int):
                attack_label = torch.full_like(label, fill_value=target)
        else:
            attack_label = label

        logits = model(image)
        if i == 0:
            num_classes = logits.size(1)
            confmat_orig = ConfusionMatrix(num_classes=num_classes)
            confmat_adv = ConfusionMatrix(num_classes=num_classes)

        mask = label < num_classes
        mask_sum = mask.flatten(1).sum(dim=1)
        pred = logits.argmax(dim=1)
        accuracies.extend(((pred == label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())
        confmat_orig.update(label, pred)

        if targeted:
            target_mask = attack_label < logits.size(1)
            target_sum = target_mask.flatten(1).sum(dim=1)
            apsrs_orig.extend(((pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs_orig.extend(((pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())

        forward_counter.reset(), backward_counter.reset()
        start.record()
        adv_image = attack(model=model, inputs=image, labels=attack_label, targeted=targeted)
        # performance monitoring
        end.record()
        torch.cuda.synchronize()
        times.append((start.elapsed_time(end)) / 1000)  # times for cuda Events are in milliseconds
        forwards.append(forward_counter.num_samples_called)
        backwards.append(backward_counter.num_samples_called)
        forward_counter.reset(), backward_counter.reset()

        if adv_image.min() < 0 or adv_image.max() > 1:
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_image.clamp_(min=0, max=1)

        if return_adv:
            adv_images.append(adv_image.cpu().clone())

        adv_logits = model(adv_image)
        adv_pred = adv_logits.argmax(dim=1)
        confmat_adv.update(label, adv_pred)
        if targeted:
            apsrs.extend(((adv_pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs.extend(((adv_pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_image, image).detach().cpu().tolist())

    acc_global, accs, ious = confmat_orig.compute()
    adv_acc_global, adv_accs, adv_ious = confmat_adv.compute()

    data = {
        'image_names': image_list[:len(apsrs)],
        'targeted': targeted,
        'accuracy': accuracies,
        'acc_global': acc_global.item(),
        'adv_acc_global': adv_acc_global.item(),
        'ious': ious.cpu().tolist(),
        'adv_ious': adv_ious.cpu().tolist(),
        'apsr_orig': apsrs_orig,
        'apsr': apsrs,
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': distances,
    }

    if return_adv:
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            images = torch.cat(images, dim=0)
            adv_images = torch.cat(adv_images, dim=0)
        data['images'] = images
        data['adv_images'] = adv_images

    return data
