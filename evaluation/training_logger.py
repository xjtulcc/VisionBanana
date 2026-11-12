import csv
import json
from pathlib import Path


class TrainingLogger:
    """
    Record training and validation metrics.

    One row per optimizer step or epoch.
    Supports JSON and CSV export.
    """

    FIELDS = [
        "record_type",
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "miou",
        "pixel_accuracy",
        "per_class_iou",
    ]

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = self.output_dir / "metrics.csv"
        self.json_path = self.output_dir / "metrics.json"

        self.records = []

        # Resume existing logs if available.
        if self.json_path.exists():
            with open(
                self.json_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.records = json.load(f)

        # Create CSV header if necessary.
        if not self.csv_path.exists():
            with open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.FIELDS,
                )
                writer.writeheader()

    def _save(self, record):
        self.records.append(record)

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.FIELDS,
            )
            writer.writerow(record)

        with open(
            self.json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.records,
                f,
                indent=4,
                ensure_ascii=False,
            )

    def log_train_step(
        self,
        epoch,
        global_step,
        train_loss,
        learning_rate,
    ):
        record = {
            "record_type": "train",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "val_loss": None,
            "learning_rate": float(learning_rate),
            "miou": None,
            "pixel_accuracy": None,
            "per_class_iou": None,
        }

        self._save(record)

    def log_validation(
        self,
        epoch,
        global_step,
        val_loss,
        miou=None,
        pixel_accuracy=None,
        per_class_iou=None,
    ):
        record = {
            "record_type": "validation",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": None,
            "val_loss": float(val_loss),
            "learning_rate": None,
            "miou": (float(miou) if miou is not None else None),
            "pixel_accuracy": (
                float(pixel_accuracy) if pixel_accuracy is not None else None
            ),
            "per_class_iou": (per_class_iou if per_class_iou is not None else None),
        }

        self._save(record)
