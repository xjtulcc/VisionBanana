import csv
from pathlib import Path

import numpy as np
import torch

from data.cityscapes import CityscapesDataset
from data.cityscapes_palette import (
    CITYSCAPES_PALETTE,
    IGNORE_COLOR,
    IGNORE_INDEX,
    decode_cityscapes_rgb,
    encode_cityscapes_mask,
)

# ==========================================
# 1. Experiment Configuration
# ==========================================

SEED = 42

# Initially test 3 images.
# Increase to 10 or more after confirming correctness.
SAMPLE_COUNT = 3

NOISE_LEVELS = [0, 1, 2, 4, 8, 12, 16]

DISTANCE_THRESHOLDS = [5, 10, 15, 20, 30, 40]

DATASET_ROOT = "datasets/cityscapes"

OUTPUT_DIR = Path("results/cityscapes_threshold")


# ==========================================
# 2. Palette Analysis
# ==========================================


def calculate_palette_distance():

    class_ids = sorted(CITYSCAPES_PALETTE.keys())

    colors = np.array(
        [CITYSCAPES_PALETTE[i] for i in class_ids],
        dtype=np.float32,
    )

    diff = colors[:, None, :] - colors[None, :, :]

    distances = np.sqrt(np.sum(diff**2, axis=2))

    np.fill_diagonal(distances, np.inf)

    min_index = np.argmin(distances)

    i, j = np.unravel_index(
        min_index,
        distances.shape,
    )

    min_distance = distances[i, j]

    print("\n========== Palette Analysis ==========")

    print("Minimum inter-class distance:", min_distance)

    print("Closest classes:", class_ids[i], class_ids[j])

    print("Half minimum distance:", min_distance / 2)

    return min_distance


# ==========================================
# 3. Generate Noisy RGB
# ==========================================


def generate_noisy_rgb(
    rgb,
    original_mask,
    noise_std,
    sample_index,
    noise_index,
):
    """
    Generate one deterministic noisy RGB image.

    The same noisy image is reused for every
    distance threshold.
    """

    valid_mask = original_mask != IGNORE_INDEX

    rng = np.random.default_rng(SEED + sample_index * 1009 + noise_index * 9176)

    noisy_rgb = rgb.astype(np.float32).copy()

    noise = rng.normal(
        loc=0,
        scale=noise_std,
        size=rgb.shape,
    ).astype(np.float32)

    noisy_rgb[valid_mask] += noise[valid_mask]

    noisy_rgb = (
        np.clip(
            noisy_rgb,
            0,
            255,
        )
        .round()
        .astype(np.uint8)
    )

    # Preserve Ignore pixels.
    noisy_rgb[~valid_mask] = IGNORE_COLOR

    return noisy_rgb


# ==========================================
# 4. Calculate Metrics
# ==========================================


def calculate_metrics(
    original_mask,
    recovered_mask,
):

    valid_mask = original_mask != IGNORE_INDEX

    ignore_mask = ~valid_mask

    valid_total = int(valid_mask.sum())

    ignore_total = int(ignore_mask.sum())

    # Pixels correctly recovered.
    correct = recovered_mask[valid_mask] == original_mask[valid_mask]

    correct_total = int(correct.sum())

    # Pixels accepted by Decoder.
    accepted = recovered_mask[valid_mask] != IGNORE_INDEX

    accepted_total = int(accepted.sum())

    # Correctly classified accepted pixels.
    accepted_correct = (
        recovered_mask[valid_mask][accepted] == original_mask[valid_mask][accepted]
    )

    accepted_correct_total = int(accepted_correct.sum())

    # Ignore region accuracy.
    if ignore_total > 0:
        ignore_correct = recovered_mask[ignore_mask] == IGNORE_INDEX

        ignore_accuracy = ignore_correct.mean() * 100

    else:
        ignore_accuracy = 100.0

    # Overall accuracy.
    accuracy = correct_total / valid_total * 100 if valid_total > 0 else 0.0

    # Coverage.
    coverage = accepted_total / valid_total * 100 if valid_total > 0 else 0.0

    # Accuracy among accepted pixels.
    accepted_accuracy = (
        accepted_correct_total / accepted_total * 100 if accepted_total > 0 else 0.0
    )

    return {
        "valid_total": valid_total,
        "correct_total": correct_total,
        "accepted_total": accepted_total,
        "accepted_correct_total": accepted_correct_total,
        "ignore_total": ignore_total,
        "accuracy": accuracy,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "ignore_accuracy": ignore_accuracy,
    }


# ==========================================
# 5. Load Dataset
# ==========================================


def load_dataset():

    dataset = CityscapesDataset(
        root=DATASET_ROOT,
        split="train",
        prompt="Segment all objects in the image.",
        image_size=(512, 1024),
        encode_mask=encode_cityscapes_mask,
    )

    print("\n========== Dataset Information ==========")

    print("Total samples:", len(dataset))

    # Select reproducible samples from the dataset.
    rng = np.random.default_rng(SEED)

    sample_indices = np.sort(
        rng.choice(
            len(dataset),
            size=min(SAMPLE_COUNT, len(dataset)),
            replace=False,
        )
    )

    print("Selected sample indices:", sample_indices.tolist())

    return dataset, sample_indices


# ==========================================
# 6. Run Threshold Grid Experiment
# ==========================================


def run_experiment(dataset, sample_indices):

    # Store aggregated pixel counts for each combination.
    accumulators = {}

    for noise_index, noise_std in enumerate(NOISE_LEVELS):
        for threshold in DISTANCE_THRESHOLDS:
            accumulators[(noise_std, threshold)] = {
                "valid_total": 0,
                "correct_total": 0,
                "accepted_total": 0,
                "accepted_correct_total": 0,
                "ignore_total": 0,
                "ignore_correct_total": 0,
            }

    for sample_index in sample_indices:
        sample_index = int(sample_index)

        print(f"\nProcessing sample {sample_index}...")

        sample = dataset[sample_index]

        original_mask = sample["mask"].cpu().numpy().astype(np.uint8)

        rgb = sample["rgb_target"].permute(1, 2, 0).cpu().numpy()

        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)

        valid_total = int(np.sum(original_mask != IGNORE_INDEX))

        ignore_total = int(np.sum(original_mask == IGNORE_INDEX))

        print("Mask shape:", original_mask.shape)
        print("Valid pixels:", valid_total)

        for noise_index, noise_std in enumerate(NOISE_LEVELS):
            # Important:
            # Generate noise only once for this sample/std.
            # All thresholds use the exact same noisy RGB.
            noisy_rgb = generate_noisy_rgb(
                rgb=rgb,
                original_mask=original_mask,
                noise_std=noise_std,
                sample_index=sample_index,
                noise_index=noise_index,
            )

            for threshold in DISTANCE_THRESHOLDS:
                recovered_mask = decode_cityscapes_rgb(
                    noisy_rgb,
                    max_distance=threshold,
                )

                metrics = calculate_metrics(
                    original_mask,
                    recovered_mask,
                )

                key = (noise_std, threshold)

                acc = accumulators[key]

                acc["valid_total"] += metrics["valid_total"]
                acc["correct_total"] += metrics["correct_total"]

                acc["accepted_total"] += metrics["accepted_total"]

                acc["accepted_correct_total"] += metrics["accepted_correct_total"]

                acc["ignore_total"] += metrics["ignore_total"]

                acc["ignore_correct_total"] += round(
                    metrics["ignore_accuracy"] * metrics["ignore_total"] / 100
                )

            print(f"  Noise Std {noise_std}: completed")

    # Convert aggregated counts into final metrics.
    results = []

    for noise_std in NOISE_LEVELS:
        for threshold in DISTANCE_THRESHOLDS:
            acc = accumulators[(noise_std, threshold)]

            valid_total = acc["valid_total"]
            accepted_total = acc["accepted_total"]

            accuracy = acc["correct_total"] / valid_total * 100 if valid_total else 0.0

            coverage = accepted_total / valid_total * 100 if valid_total else 0.0

            accepted_accuracy = (
                acc["accepted_correct_total"] / accepted_total * 100
                if accepted_total
                else 0.0
            )

            ignore_accuracy = (
                acc["ignore_correct_total"] / acc["ignore_total"] * 100
                if acc["ignore_total"]
                else 100.0
            )

            results.append(
                {
                    "noise_std": noise_std,
                    "max_distance": threshold,
                    "accuracy": accuracy,
                    "coverage": coverage,
                    "accepted_accuracy": accepted_accuracy,
                    "ignore_accuracy": ignore_accuracy,
                }
            )

    return results


# ==========================================
# 7. Save CSV
# ==========================================


def save_csv(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = OUTPUT_DIR / "threshold_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_std",
                "max_distance",
                "accuracy",
                "coverage",
                "accepted_accuracy",
                "ignore_accuracy",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    print("\nCSV saved:", csv_path.resolve())


# ==========================================
# 8. Generate Heatmaps
# ==========================================


def save_heatmaps(results):

    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print("\nMatplotlib is not installed. CSV results are still available.")

        print("Install with: python -m pip install matplotlib")

        return

    metrics = [
        ("accuracy", "Overall Accuracy"),
        ("coverage", "Coverage"),
        ("accepted_accuracy", "Accepted Accuracy"),
    ]

    for metric_name, title in metrics:
        matrix = np.zeros(
            (
                len(NOISE_LEVELS),
                len(DISTANCE_THRESHOLDS),
            )
        )

        for result in results:
            i = NOISE_LEVELS.index(result["noise_std"])

            j = DISTANCE_THRESHOLDS.index(result["max_distance"])

            matrix[i, j] = result[metric_name]

        fig, ax = plt.subplots(figsize=(9, 6))

        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=100,
        )

        ax.set_xticks(range(len(DISTANCE_THRESHOLDS)))

        ax.set_xticklabels(DISTANCE_THRESHOLDS)

        ax.set_yticks(range(len(NOISE_LEVELS)))

        ax.set_yticklabels(NOISE_LEVELS)

        ax.set_xlabel("Max Distance")
        ax.set_ylabel("Noise Std")

        ax.set_title(title)

        for i in range(len(NOISE_LEVELS)):
            for j in range(len(DISTANCE_THRESHOLDS)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Percentage (%)",
        )

        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{metric_name}_heatmap.png"

        fig.savefig(
            output_path,
            dpi=200,
        )

        plt.close(fig)

        print("Heatmap saved:", output_path)


# ==========================================
# 9. Print Results
# ==========================================


def print_results(results):

    print("\n========== Threshold Grid Results ==========")

    print(
        f"{'Noise':>7} "
        f"{'Threshold':>10} "
        f"{'Accuracy':>12} "
        f"{'Coverage':>12} "
        f"{'Accepted':>12} "
        f"{'Ignore':>10}"
    )

    for r in results:
        print(
            f"{r['noise_std']:>7} "
            f"{r['max_distance']:>10} "
            f"{r['accuracy']:>11.4f}% "
            f"{r['coverage']:>11.4f}% "
            f"{r['accepted_accuracy']:>11.4f}% "
            f"{r['ignore_accuracy']:>9.4f}%"
        )


# ==========================================
# 10. Main
# ==========================================


def main():

    calculate_palette_distance()

    dataset, sample_indices = load_dataset()

    results = run_experiment(
        dataset,
        sample_indices,
    )

    print_results(results)

    save_csv(results)

    save_heatmaps(results)

    print("\nThreshold Evaluation Completed.")


if __name__ == "__main__":
    main()
