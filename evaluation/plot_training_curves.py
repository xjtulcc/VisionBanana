import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(csv_path):
    records = []

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            records.append(row)

    return records


def number(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_loss(records, output_dir):
    train_steps = []
    train_losses = []

    val_steps = []
    val_losses = []

    for row in records:
        step = number(row["global_step"])

        if row["record_type"] == "train":
            loss = number(row["train_loss"])

            if step is not None and loss is not None:
                train_steps.append(step)
                train_losses.append(loss)

        elif row["record_type"] == "validation":
            loss = number(row["val_loss"])

            if step is not None and loss is not None:
                val_steps.append(step)
                val_losses.append(loss)

    if not train_losses and not val_losses:
        print("No loss records found.")
        return

    plt.figure(figsize=(10, 6))

    if train_losses:
        plt.plot(
            train_steps,
            train_losses,
            label="Training Loss",
            alpha=0.65,
        )

    if val_losses:
        plt.plot(
            val_steps,
            val_losses,
            marker="o",
            label="Validation Loss",
        )

    plt.xlabel("Optimizer Steps")
    plt.ylabel("Diffusion Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=200,
    )

    plt.close()


def plot_learning_rate(records, output_dir):
    steps = []
    learning_rates = []

    for row in records:
        if row["record_type"] != "train":
            continue

        step = number(row["global_step"])
        lr = number(row["learning_rate"])

        if step is not None and lr is not None:
            steps.append(step)
            learning_rates.append(lr)

    if not steps:
        print("No learning-rate records found.")
        return

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        learning_rates,
        label="Learning Rate",
    )

    plt.xlabel("Optimizer Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "learning_rate_curve.png",
        dpi=200,
    )

    plt.close()


def load_semantic_records(records):
    semantic_records = []

    for row in records:
        if row["record_type"] != "validation":
            continue

        miou = number(row["miou"])
        pixel_accuracy = number(row["pixel_accuracy"])

        if miou is None and pixel_accuracy is None:
            continue

        semantic_records.append(
            {
                "epoch": int(row["epoch"]),
                "miou": miou,
                "pixel_accuracy": pixel_accuracy,
                "per_class_iou": (
                    json.loads(row["per_class_iou"]) if row["per_class_iou"] else None
                ),
            }
        )

    return semantic_records


def plot_semantic_metrics(records, output_dir):
    semantic_records = load_semantic_records(records)

    if not semantic_records:
        print("Semantic metrics not available. Skipping mIoU and Pixel Accuracy plots.")
        return

    epochs = [r["epoch"] for r in semantic_records]

    # mIoU.
    miou_points = [
        (r["epoch"], r["miou"]) for r in semantic_records if r["miou"] is not None
    ]

    if miou_points:
        plt.figure(figsize=(9, 5))

        plt.plot(
            [p[0] for p in miou_points],
            [p[1] for p in miou_points],
            marker="o",
            label="Validation mIoU",
        )

        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.title("Validation mIoU")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / "miou_curve.png",
            dpi=200,
        )

        plt.close()

    # Pixel Accuracy.
    pa_points = [
        (r["epoch"], r["pixel_accuracy"])
        for r in semantic_records
        if r["pixel_accuracy"] is not None
    ]

    if pa_points:
        plt.figure(figsize=(9, 5))

        plt.plot(
            [p[0] for p in pa_points],
            [p[1] for p in pa_points],
            marker="o",
            label="Pixel Accuracy",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Pixel Accuracy")
        plt.title("Validation Pixel Accuracy")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / "pixel_accuracy_curve.png",
            dpi=200,
        )

        plt.close()

    # Per-class IoU.
    class_records = [
        r for r in semantic_records if isinstance(r["per_class_iou"], dict)
    ]

    if class_records:
        class_names = list(class_records[0]["per_class_iou"].keys())

        plt.figure(figsize=(13, 7))

        for class_name in class_names:
            x = []
            y = []

            for record in class_records:
                value = number(record["per_class_iou"].get(class_name))

                if value is not None:
                    x.append(record["epoch"])
                    y.append(value)

            if x:
                plt.plot(
                    x,
                    y,
                    marker=".",
                    label=class_name,
                )

        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Per-class IoU")
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_dir / "per_class_iou.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="checkpoints/semantic_lora/training_logs/metrics.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/training_curves",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_metrics(csv_path)

    print("Loaded records:", len(records))

    plot_loss(records, output_dir)

    plot_learning_rate(records, output_dir)

    plot_semantic_metrics(records, output_dir)

    print("\n========== Plotting Completed ==========")
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
