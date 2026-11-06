import csv
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from tqdm import tqdm


from utils.config import load_config
from utils.dataset import build_loaders
from models import build_model
from utils.metrics import SegmentationMetrics


def run_epoch(
    model, loader, criterion, device, ignore_index, optimizer=None, description="Train"
):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(21, ignore_index=255, device=device)
    total_loss = 0.0
    total_pixels = 0
    progress = tqdm(loader, desc=description)

    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)
        valid_pixels = (masks != ignore_index).sum()

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss_sum = criterion(logits, masks)
            loss = loss_sum / valid_pixels.clamp_min(1)

            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.detach().argmax(dim=1)
        metrics.update(predictions, masks)
        total_loss += loss_sum.detach().item()
        total_pixels += valid_pixels.item()
        progress.set_postfix(loss=f"{total_loss / max(total_pixels,1):.4f}")
    return {"loss": total_loss / max(total_pixels, 1), **metrics.compute()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备", device)

    config = load_config()
    seed = config["training"]["seed"]
    torch.manual_seed(seed)

    train_loader, val_loader = build_loaders(config)

    print("训练集数据数量：", len(train_loader.dataset))
    print("验证集数据数量：", len(val_loader.dataset))

    model = build_model(config["model"]).to(device)
    print(f"使用的模型为：{config['model']['name']}")

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="sum")
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = Path(config["training"]["output_dir"]) / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["loss", "miou", "f1", "precision", "recall"]
    fieldnames = ["epoch"] + [
        f"{split}_{name}" for split in ("train", "val") for name in metric_names
    ]
    best_miou = -1.0
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, config["training"]["epochs"] + 1):
            train_result = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                255,
                optimizer,
                description=f"Epoch {epoch} Train",
            )
            val_result = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                255,
                description=f"Epoch {epoch} Val",
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
                f"Epoch {epoch}/{config["training"]['epochs']} | "
                f"train loss {train_result['loss']:.4f}, mIoU {train_result['miou']:.4f} | "
                f"val loss {val_result['loss']:.4f}, mIoU {val_result['miou']:.4f}, "
                f"F1 {val_result['f1']:.4f} | best mIoU {best_miou:.4f}"
            )


if __name__ == "__main__":
    main()
