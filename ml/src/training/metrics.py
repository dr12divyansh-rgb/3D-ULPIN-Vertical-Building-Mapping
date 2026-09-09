"""Segmentation metrics: confusion matrix, per-class P/R/F1/IoU, mean IoU."""

from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a (num_classes, num_classes) confusion matrix (rows=true, cols=pred)."""
    mask = (target >= 0) & (target < num_classes)
    target = target[mask]
    pred = pred[mask]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


def segmentation_metrics(pred: np.ndarray, target: np.ndarray,
                         num_classes: int, class_names=None) -> dict:
    """Compute overall accuracy, per-class precision/recall/F1/IoU, mean IoU."""
    cm = confusion_matrix(pred, target, num_classes)
    names = class_names or [str(i) for i in range(num_classes)]

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp          # predicted as class, actually not
    fn = cm.sum(axis=1) - tp          # actually class, predicted not
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        iou = tp / (tp + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    precision = np.where(np.isfinite(precision), precision, 0.0)
    recall = np.where(np.isfinite(recall), recall, 0.0)
    iou = np.where(np.isfinite(iou), iou, 0.0)
    f1 = np.where(np.isfinite(f1), f1, 0.0)

    per_class = {
        names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
        }
        for i in range(num_classes)
    }

    present = (cm.sum(axis=1) + cm.sum(axis=0) - tp) > 0
    mean_iou = float(iou[present].mean()) if present.any() else 0.0

    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "mean_iou": mean_iou,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def class_names_from_repo():
    from synthetic_city.core.labels import CLASS_NAMES
    return list(CLASS_NAMES)
