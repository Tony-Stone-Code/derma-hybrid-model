from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_classification_metrics(
    targets: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    targets_array = np.asarray(targets)
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    predictions = probabilities_array.argmax(axis=1)
    metrics: dict[str, object] = {
        "macro_f1": float(f1_score(targets_array, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(targets_array, predictions)),
        "confusion_matrix": confusion_matrix(
            targets_array, predictions, labels=np.arange(len(class_names))
        ),
        "per_class": {},
    }

    for class_index, class_name in enumerate(class_names):
        y_true = (targets_array == class_index).astype(np.int32)
        y_score = probabilities_array[:, class_index]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        class_metrics: dict[str, object] = {
            "precision": precision,
            "recall": recall,
            "average_precision": float(average_precision_score(y_true, y_score)),
        }
        if y_true.min() != y_true.max():
            class_metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        metrics["per_class"][class_name] = class_metrics

    return metrics
