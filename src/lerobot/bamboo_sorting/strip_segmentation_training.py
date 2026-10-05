"""Mask2Former adapter for the portable, reviewed strip dataset (no robot imports)."""

from pathlib import Path

import numpy as np
from PIL import Image

from .strip_segmentation_data import CLASSES, child, read_json


class StripDataset:
    def __init__(self, root, split):
        self.root = Path(root)
        self.samples = [s for s in read_json(self.root / "dataset.json")["samples"] if s["split"] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        sample = self.samples[index]
        with Image.open(child(self.root, sample["image"])) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(child(self.root, sample["instance_map"])) as image:
            instance_map = np.array(image)
        masks = [instance_map == s["mask_id"] for s in sample["segments"]]
        masks = np.stack(masks) if masks else np.zeros((0, *instance_map.shape), dtype=bool)
        return {"image": rgb, "mask_labels": torch.from_numpy(masks.astype(np.float32)),
                "class_labels": torch.tensor([s["class_id"] for s in sample["segments"]], dtype=torch.long),
                "sample": sample}


def collate_samples(items, processor):
    # All current frames are 640x480, both divisible by 32. No stretching, cropping,
    # resizing or label reduction. Background is absent from the instance targets.
    sizes = {item["image"].shape[:2] for item in items}
    if len(sizes) != 1 or any(d % 32 for d in next(iter(sizes))):
        raise ValueError("this baseline requires equal image sizes divisible by 32")
    batch = dict(processor(images=[x["image"] for x in items], do_resize=False, return_tensors="pt"))
    batch["mask_labels"] = [x["mask_labels"] for x in items]
    batch["class_labels"] = [x["class_labels"] for x in items]
    batch["samples"] = [x["sample"] for x in items]
    return batch


def move_batch(batch, device):
    return {k: [x.to(device) for x in v] if isinstance(v, list) else v.to(device)
            for k, v in batch.items() if k != "samples"}


def forward_loss(model, batch):
    """Explicit no-object supervision for all-empty batches; no fake background object."""
    import torch
    import torch.nn.functional as functional

    if any(len(x) for x in batch["class_labels"]):
        output = model(**batch)
        return output, output.loss
    output = model(pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask"),
                   output_auxiliary_logits=True)
    logits = [output.class_queries_logits]
    if model.config.use_auxiliary_loss:
        logits += [x["class_queries_logits"] for x in (output.auxiliary_logits or [])]
    loss = sum(functional.cross_entropy(x.transpose(1, 2),
               torch.full(x.shape[:2], model.config.num_labels, dtype=torch.long, device=x.device))
               for x in logits) * model.config.class_weight
    return output, loss


def match_instances(predicted, target, iou_threshold=0.5):
    """Class-aware one-to-one matching; these are fixed-threshold metrics, not COCO AP."""
    from scipy.optimize import linear_sum_assignment

    quality = np.zeros((len(predicted), len(target)), dtype=float)
    for i, (pclass, pmask) in enumerate(predicted):
        for j, (tclass, tmask) in enumerate(target):
            if pclass == tclass:
                union = np.count_nonzero(pmask | tmask)
                quality[i, j] = np.count_nonzero(pmask & tmask) / union if union else 0
    # Favor number of valid matches first, then their IoU.
    weight = (quality >= iou_threshold) * (min(quality.shape, default=0) + 1) + quality
    rows, cols = linear_sum_assignment(-weight)
    pairs = [(i, j) for i, j in zip(rows, cols, strict=True) if quality[i, j] >= iou_threshold]
    counts = []
    for category in range(len(CLASSES)):
        tp = sum(target[j][0] == category for _, j in pairs)
        counts.append({"tp": tp, "fp": sum(x[0] == category for x in predicted) - tp,
                       "fn": sum(x[0] == category for x in target) - tp})
    return counts


def evaluate(model, processor, loader, device, score_threshold=0.5, prediction_dir=None):
    import cv2
    import torch

    model.eval()
    totals = [{"tp": 0, "fp": 0, "fn": 0} for _ in CLASSES]
    loss_sum, count, empty, false_empty = 0.0, 0, 0, 0
    details = []
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            output, loss = forward_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            samples = cpu_batch["samples"]
            results = processor.post_process_instance_segmentation(
                output, threshold=score_threshold, mask_threshold=0.5,
                target_sizes=[(s["height"], s["width"]) for s in samples])
            loss_sum += loss.item() * len(samples)
            count += len(samples)
            for k, (sample, result) in enumerate(zip(samples, results, strict=True)):
                seg = result["segmentation"].cpu().numpy()
                info = result["segments_info"]
                predictions = [(s["label_id"], seg == s["id"]) for s in info if np.any(seg == s["id"])]
                truth = [(int(c), m.numpy() > 0) for c, m in
                         zip(cpu_batch["class_labels"][k], cpu_batch["mask_labels"][k], strict=True)]
                scores = match_instances(predictions, truth)
                for target, source in zip(totals, scores, strict=True):
                    for metric in target:
                        target[metric] += source[metric]
                empty += int(not truth)
                false_empty += int(not truth and bool(predictions))
                detail = {"id": sample["id"], "placement_id": sample["placement_id"],
                          "counts": scores, "predicted_instances": len(predictions), "true_instances": len(truth)}
                if prediction_dir is not None:
                    # Preserve model predictions, with 0 reserved for background.
                    png = np.zeros(seg.shape, dtype=np.uint16)
                    segments = []
                    for mask_id, s in enumerate(info, 1):
                        png[seg == s["id"]] = mask_id
                        segments.append({"mask_id": mask_id, "class_id": s["label_id"], "score": s["score"]})
                    path = prediction_dir / f"{sample['id']}.png"
                    if not cv2.imwrite(str(path), png):
                        raise OSError(path)
                    detail.update(instance_map=path.name, segments=segments)
                details.append(detail)
    metrics = {}
    for category, values in zip(CLASSES, totals, strict=True):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        metrics[category["name"]] = {**values, "precision": tp / max(tp + fp, 1),
                                     "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    return {"loss": loss_sum / max(count, 1), "images": count, "classes": metrics,
            "macro_f1": sum(x["f1"] for x in metrics.values()) / len(metrics),
            "empty_images": empty, "empty_false_positive_images": false_empty,
            "empty_false_positive_rate": false_empty / max(empty, 1),
            "score_threshold": score_threshold, "matching_mask_iou": 0.5, "samples": details}
