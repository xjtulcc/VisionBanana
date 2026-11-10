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


def calculate_palette_distance():
    """
    计算不同类别标准 RGB 颜色之间的最小欧氏距离。
    """

    class_ids = sorted(CITYSCAPES_PALETTE.keys())

    colors = np.array([CITYSCAPES_PALETTE[i] for i in class_ids], dtype=np.float32)

    diff = colors[:, None, :] - colors[None, :, :]

    distances = np.sqrt(np.sum(diff**2, axis=2))

    np.fill_diagonal(distances, np.inf)

    min_distance = distances.min()

    i, j = np.unravel_index(np.argmin(distances), distances.shape)

    print("\n========== Palette Analysis ==========")

    print("Minimum inter-class distance:", min_distance)

    print(f"Closest classes: {class_ids[i]} and {class_ids[j]}")

    print("Half minimum distance:", min_distance / 2)

    return min_distance


def evaluate_decoder(
    original_mask,
    rgb,
    noise_std,
    max_distance=15,
    seed=42,
):
    """
    对 RGB Target 添加高斯噪声，
    然后使用最近邻 Decoder 恢复 Mask。
    """

    rng = np.random.default_rng(seed)

    valid_mask = original_mask != IGNORE_INDEX

    # Convert to float for noise addition.
    noisy_rgb = rgb.astype(np.float32).copy()

    # Gaussian noise is applied only to valid pixels.
    noise = rng.normal(loc=0, scale=noise_std, size=rgb.shape).astype(np.float32)

    noisy_rgb[valid_mask] += noise[valid_mask]

    # Clip RGB values.
    noisy_rgb = np.clip(noisy_rgb, 0, 255).round().astype(np.uint8)

    # Preserve Ignore Color exactly.
    noisy_rgb[~valid_mask] = IGNORE_COLOR

    # Decode noisy RGB.
    recovered = decode_cityscapes_rgb(noisy_rgb, max_distance=max_distance)

    recovered = np.asarray(recovered)

    # Calculate valid pixel accuracy.
    correct = recovered[valid_mask] == original_mask[valid_mask]

    valid_accuracy = correct.mean() * 100

    # Coverage: proportion of valid pixels
    # that were not rejected as Ignore.
    accepted = recovered[valid_mask] != IGNORE_INDEX

    coverage = accepted.mean() * 100

    # Accuracy among accepted pixels.
    if accepted.any():
        accepted_accuracy = (
            recovered[valid_mask][accepted] == original_mask[valid_mask][accepted]
        ).mean() * 100

    else:
        accepted_accuracy = 0.0

    # Ignore region accuracy.
    ignore_pixels = ~valid_mask

    if ignore_pixels.any():
        ignore_accuracy = (recovered[ignore_pixels] == IGNORE_INDEX).mean() * 100

    else:
        ignore_accuracy = 100.0

    return {
        "noise_std": noise_std,
        "valid_accuracy": valid_accuracy,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "ignore_accuracy": ignore_accuracy,
    }


def main():

    # 1. Analyze Palette.
    calculate_palette_distance()

    # 2. Load real Cityscapes sample.
    dataset = CityscapesDataset(
        root="datasets/cityscapes",
        split="train",
        prompt="Segment all objects in the image.",
        image_size=(512, 1024),
        encode_mask=encode_cityscapes_mask,
    )

    sample = dataset[0]

    original_mask = sample["mask"].cpu().numpy().astype(np.uint8)

    rgb = sample["rgb_target"].permute(1, 2, 0).cpu().numpy()

    rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)

    print("\n========== Dataset Information ==========")

    print("Mask shape:", original_mask.shape)
    print("RGB shape:", rgb.shape)

    print("Valid pixels:", np.sum(original_mask != IGNORE_INDEX))

    # 3. Test different noise levels.
    noise_levels = [0, 1, 2, 4, 8, 12, 16]

    results = []

    for noise_std in noise_levels:
        result = evaluate_decoder(
            original_mask=original_mask,
            rgb=rgb,
            noise_std=noise_std,
            max_distance=15,
        )

        results.append(result)

    # 4. Print results.
    print("\n========== Robustness Results ==========")

    print(
        f"{'Noise':>8} "
        f"{'Accuracy':>12} "
        f"{'Coverage':>12} "
        f"{'Accepted Acc':>14} "
        f"{'Ignore Acc':>12}"
    )

    for r in results:
        print(
            f"{r['noise_std']:>8.1f} "
            f"{r['valid_accuracy']:>11.4f}% "
            f"{r['coverage']:>11.4f}% "
            f"{r['accepted_accuracy']:>13.4f}% "
            f"{r['ignore_accuracy']:>11.4f}%"
        )

    print("\nRobustness Evaluation Completed.")


if __name__ == "__main__":
    main()
