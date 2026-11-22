import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F


VOC_NUM_CLASSES = 21
VOC_IGNORE_INDEX = 255
DEFAULT_VOC_ROOT = r"D:\04_Datasets\VOCtrainval_11-May-2012"


class VOCTransform:
    def __init__(self, image_size=256, augmentation=None):
        self.image_size = image_size
        self.resize = T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        augmentation = augmentation or {}
        self.augment = augmentation.get("enabled", False)

        if self.augment:
            self.scale_range = augmentation.get("scale_range", (0.8, 1.25))
            min_scale, max_scale = self.scale_range
            if not 0 < min_scale <= max_scale:
                raise ValueError("dataset.augmentation.scale_range 必须满足 0 < 最小值 <= 最大值")

            self.crop = T.RandomCrop((image_size, image_size))
            fill = {tv_tensors.Image: 0, tv_tensors.Mask: VOC_IGNORE_INDEX}
            rotation = T.RandomRotation(
                degrees=augmentation.get("rotation_degrees", 10),
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            )
            color_jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
            )
            blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
            self.geometry = T.Compose(
                [
                    T.RandomHorizontalFlip(p=augmentation.get("horizontal_flip_p", 0.5)),
                    T.RandomApply([rotation], p=augmentation.get("rotation_p", 0.2)),
                ]
            )
            self.photometric = T.Compose(
                [
                    T.RandomApply([color_jitter], p=augmentation.get("color_jitter_p", 0.8)),
                    T.RandomApply([blur], p=augmentation.get("blur_p", 0.1)),
                ]
            )

    def _scale_and_crop(self, image, mask):
        height, width = image.shape[-2:]
        short_edge = self.image_size * torch.empty(()).uniform_(*self.scale_range).item()
        # 限制极长图像的中间尺寸，缩放时保持原始宽高比。
        scale = min(short_edge / min(height, width), 2 * self.image_size / max(height, width))
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
        size = [new_height, new_width]
        image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, size)  # tv_tensors.Mask 使用最近邻插值。

        pad_h = max(self.image_size - new_height, 0)
        pad_w = max(self.image_size - new_width, 0)
        if pad_h or pad_w:
            left, top = pad_w // 2, pad_h // 2
            padding = [left, top, pad_w - left, pad_h - top]
            image = F.pad(image, padding, fill=0)
            mask = F.pad(mask, padding, fill=VOC_IGNORE_INDEX)
        return self.crop(image, mask)

    def __call__(self, image, mask):
        image = tv_tensors.Image(F.pil_to_tensor(image))
        mask = tv_tensors.Mask(F.pil_to_tensor(mask))
        if self.augment:
            image, mask = self._scale_and_crop(image, mask)
            image, mask = self.geometry(image, mask)
            image = self.photometric(image)
        else:
            image, mask = self.resize(image, mask)

        image = F.to_dtype(image, torch.float32, scale=True)
        mask = mask.squeeze(0).long()
        return image, mask


def build_loaders(config):
    loader_cfg = config["training"]
    dataset_cfg = config.get("dataset", {})
    root = dataset_cfg.get("root", DEFAULT_VOC_ROOT)
    image_size = dataset_cfg.get("image_size", 256)
    train_dataset = VOCSegmentation(
        root=root,
        year="2012",
        image_set="train",
        transforms=VOCTransform(image_size, dataset_cfg.get("augmentation")),
        download=dataset_cfg.get("download", True),
    )
    val_dataset = VOCSegmentation(
        root=root,
        year="2012",
        image_set="val",
        transforms=VOCTransform(image_size),
    )

    train_loader = DataLoader(train_dataset, loader_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, loader_cfg["batch_size"], shuffle=False, num_workers=0)
    return train_loader, val_loader
