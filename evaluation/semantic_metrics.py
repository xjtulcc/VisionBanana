import numpy as np


class SemanticSegmentationMetrics:
    """
    Semantic segmentation evaluation based on confusion matrix.

    Cityscapes:
        Valid classes: 0 ~ 18
        Ignore index: 255

    Background pixels are excluded from evaluation.
    Invalid predictions on valid GT pixels count as false negatives.
    """

    def __init__(
        self,
        num_classes=19,
        ignore_index=255,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Last column represents invalid/background predictions.
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes + 1),
            dtype=np.int64,
        )

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, prediction, target):
        """
        Args:
            prediction: H x W predicted Train ID mask.
            target: H x W Ground Truth Train ID mask.
        """

        prediction = np.asarray(prediction)
        target = np.asarray(target)

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction={prediction.shape}, target={target.shape}"
            )

        prediction = prediction.astype(np.int64)
        target = target.astype(np.int64)

        # Ignore invalid GT pixels.
        valid_gt = (target >= 0) & (target < self.num_classes)

        gt = target[valid_gt]
        pred = prediction[valid_gt].copy()

        # Invalid predictions become an extra column.
        invalid_pred = (
            (pred < 0) | (pred >= self.num_classes) | (pred == self.ignore_index)
        )

        pred[invalid_pred] = self.num_classes

        # Build confusion matrix.
        indices = gt * (self.num_classes + 1) + pred

        batch_matrix = np.bincount(
            indices,
            minlength=self.num_classes * (self.num_classes + 1),
        ).reshape(
            self.num_classes,
            self.num_classes + 1,
        )

        self.confusion_matrix += batch_matrix

    def compute(self):
        """
        Returns:
            Dictionary containing:
                per_class_iou
                miou
                pixel_accuracy
                confusion_matrix
        """

        cm = self.confusion_matrix

        true_positive = np.diag(cm[:, : self.num_classes])

        gt_count = cm.sum(axis=1)

        pred_count = cm[:, : self.num_classes].sum(axis=0)

        union = gt_count + pred_count - true_positive

        per_class_iou = np.divide(
            true_positive,
            union,
            out=np.full(
                self.num_classes,
                np.nan,
                dtype=np.float64,
            ),
            where=union > 0,
        )

        # Classes absent from GT and prediction are excluded.
        valid_classes = union > 0

        miou = float(np.nanmean(per_class_iou)) if np.any(valid_classes) else 0.0

        total_pixels = gt_count.sum()

        pixel_accuracy = (
            float(true_positive.sum() / total_pixels) if total_pixels > 0 else 0.0
        )

        return {
            "per_class_iou": per_class_iou,
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "confusion_matrix": cm.copy(),
            "valid_pixels": int(total_pixels),
        }


if __name__ == "__main__":
    metrics = SemanticSegmentationMetrics(
        num_classes=19,
        ignore_index=255,
    )

    # Simple test.
    target = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 255],
        ]
    )

    prediction = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 255],
        ]
    )

    metrics.update(prediction, target)

    results = metrics.compute()

    print("========== Metric Test ==========")

    print("mIoU:", results["miou"])

    print("Pixel Accuracy:", results["pixel_accuracy"])

    print("Valid Pixels:", results["valid_pixels"])

    print("\nPer-class IoU:")

    for class_id, iou in enumerate(results["per_class_iou"]):
        if np.isfinite(iou):
            print(f"Class {class_id}: {iou:.4f}")

    assert abs(results["miou"] - 0.75) < 1e-6

    assert abs(results["pixel_accuracy"] - 6 / 7) < 1e-6

    print("\nMetric Test: PASSED")
