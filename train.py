import csv
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from tqdm import tqdm


from utils.config import load_config
from utils.dataset import VOC_IGNORE_INDEX, VOC_NUM_CLASSES, build_loaders
from models import build_model
from utils.metrics import SegmentationMetrics


def run_epoch(
    model,
    loader,
    criterion,
    device,
    ignore_index,
    optimizer=None,
    description="Train",
    *,
    num_classes=VOC_NUM_CLASSES,
    metric_reduction="macro",
):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes, ignore_index, device, reduction=metric_reduction)
    total_loss = 0.0
    total_pixels = 0
    progress = tqdm(loader, desc=description)

    for images, masks in progress:
        valid_pixels = (masks != ignore_index).sum().item()
        if not valid_pixels:
            continue
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss_sum = criterion(logits, masks)
            loss = loss_sum / valid_pixels

            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.detach().argmax(dim=1)
        metrics.update(predictions, masks)
        total_loss += loss_sum.detach().item()
        total_pixels += valid_pixels
        progress.set_postfix(loss=f"{total_loss / total_pixels:.4f}")
    if not total_pixels:
        raise RuntimeError(f"{description} 没有有效标注像素，请检查掩码和数据增强设置")
    return {"loss": total_loss / total_pixels, **metrics.compute()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备", device)

    config = load_config()
    seed = config["training"]["seed"]
    torch.manual_seed(seed)

    train_loader, val_loader = build_loaders(config)

    print("训练集数据数量：", len(train_loader.dataset))
    print("验证集数据数量：", len(val_loader.dataset))

    model = build_model(config).to(device)
    print(f"使用的模型为：{config['model']['name']}")

    num_classes = config["dataset"]["out_channels"]
    metric_reduction = config.get("metrics", {}).get("reduction", "macro")
    criterion = nn.CrossEntropyLoss(ignore_index=VOC_IGNORE_INDEX, reduction="sum")
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = Path(config["training"]["output_dir"]) / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["loss", "miou", "f1", "precision", "recall"]
    fieldnames = ["epoch"] + [
        f"{split}_{name}" for split in ("train", "val") for name in metric_names
    ]
    best_miou = -1.0
    epochs = config["training"]["epochs"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_result = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                VOC_IGNORE_INDEX,
                optimizer,
                description=f"Epoch {epoch} Train",
                num_classes=num_classes,
                metric_reduction=metric_reduction,
            )
            val_result = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                VOC_IGNORE_INDEX,
                description=f"Epoch {epoch} Val",
                num_classes=num_classes,
                metric_reduction=metric_reduction,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_result.items()},
                    **{f"val_{key}": value for key, value in val_result.items()},
                }
            )
            file.flush()
            if_best = val_result["miou"] > best_miou
            best_miou = max(best_miou, val_result["miou"])
            if if_best:
                torch.save(model.state_dict(), output_dir / "best.pth")

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train loss {train_result['loss']:.4f}, mIoU {train_result['miou']:.4f} | "
                f"val loss {val_result['loss']:.4f}, mIoU {val_result['miou']:.4f}, "
                f"F1 {val_result['f1']:.4f} | best mIoU {best_miou:.4f}"
            )


if __name__ == "__main__":
    main()
