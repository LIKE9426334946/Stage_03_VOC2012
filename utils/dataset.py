import torch

from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors
from torchvision.transforms import functional as F


class VOCTransform:
    def __init__(self):
        self.resize = T.Resize(
            size=(256, 256), interpolation=InterpolationMode.BILINEAR
        )

    def __call__(self, image, mask):

        image = tv_tensors.Image(F.pil_to_tensor(image))
        mask = tv_tensors.Mask(F.pil_to_tensor(mask))
        image, mask = self.resize(image, mask)
        image = F.convert_image_dtype(image, torch.float32)
        mask = mask.squeeze(0).long()
        return image, mask


def build_loaders(config):
    loader_cfg = config["training"]
    transform = VOCTransform()
    train_dataset = VOCSegmentation(
        root=r"D:\04_Datasets\VOCtrainval_11-May-2012",
        year="2012",
        image_set="train",
        transforms=transform,
        download=False,
    )
    val_dataset = VOCSegmentation(
        root=r"D:\04_Datasets\VOCtrainval_11-May-2012",
        year="2012",
        image_set="val",
        transform=transform,
        download=False,
    )
    train_subset = Subset(train_dataset, range(20))
    val_subset = Subset(val_dataset, range(10))

    train_loader = DataLoader(
        train_subset, loader_cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, loader_cfg["batch_size"], shuffle=False, num_workers=0
    )
    return train_loader, val_loader
