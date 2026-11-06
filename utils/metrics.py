import torch
import segmentation_models_pytorch as smp


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index, device):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.stats = [
            torch.zeros((1, num_classes), dtype=torch.long, device=device)
            for _ in range(4)
        ]

    @torch.no_grad()
    def update(self, predictions, targets):
        batch_stats = smp.metrics.get_stats(
            predictions,
            targets,
            mode="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        for total, batch in zip(self.stats, batch_stats):
            total += batch.sum(dim=0, keepdim=True).to(total.device)

    def compute(self):
        functions = {
            "miou": smp.metrics.iou_score,
            "f1": smp.metrics.f1_score,
            "precision": smp.metrics.precision,
            "recall": smp.metrics.recall,
        }
        return {
            name: function(*self.stats, reduction="macro", zero_division=0.0).item()
            for name, function in functions.items()
        }
