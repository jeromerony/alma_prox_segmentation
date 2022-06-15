from typing import Callable, Optional, Tuple

from sacred import Ingredient
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Cityscapes, VOCSegmentation
from torchvision.transforms import InterpolationMode

from utils import label_map_cityscapes

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.named_config
def pascal_voc_2012():
    name = 'pascal_voc_2012'
    root = 'data'
    split = 'val'
    size = 512
    num_images = None


@dataset_ingredient.named_config
def cityscapes():
    name = 'cityscapes'
    root = 'data/cityscapes'
    split = 'val'
    size = (1024, 2048)
    num_images = None


@dataset_ingredient.capture
def get_pascal_voc_2012(root: str, size: int, split: str,
                        num_images: Optional[int] = None) -> Tuple[DataLoader, Optional[Callable]]:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    ])
    dataset = VOCSegmentation(root=root, year='2012', image_set=split,
                              transform=transform, target_transform=target_transform)
    if num_images is not None:
        assert num_images <= len(dataset)
        dataset = Subset(dataset, indices=list(range(num_images)))
    loader = DataLoader(dataset=dataset, shuffle=False)
    return loader, None


@dataset_ingredient.capture
def get_cityscapes(root: str, size: int, split: str,
                   num_images: Optional[int] = None, batch_size: int = 1) -> Tuple[DataLoader, Optional[Callable]]:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        label_map_cityscapes,
        transforms.PILToTensor(),
        transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
    ])
    dataset = Cityscapes(root=root, split=split, target_type='semantic',
                         transform=transform, target_transform=target_transform)
    if num_images is not None:
        assert num_images <= len(dataset)
        dataset = Subset(dataset, indices=list(range(num_images)))
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    return loader, label_map_cityscapes


_loaders = {
    'pascal_voc_2012': get_pascal_voc_2012,
    'cityscapes': get_cityscapes
}


@dataset_ingredient.capture
def get_dataset(name: str) -> DataLoader:
    return _loaders[name]()
