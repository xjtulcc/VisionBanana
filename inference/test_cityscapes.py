from pathlib import Path

import numpy as np
from PIL import Image

from datasets.cityscapes import (
    CITYSCAPES_PALETTE,
    CityscapesDataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "datasets" / "cityscapes"
RESULT_DIR = PROJECT_ROOT / "results" / "cityscapes"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def colorize_mask(mask):
    """将 Train ID Mask 转换为可视化 RGB 图。"""

    rgb = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    rgb[:] = (1, 1, 1)

    for class_id, color in CITYSCAPES_PALETTE.items():
        rgb[mask == class_id] = color

    return rgb


def main():

    dataset = CityscapesDataset(
        root=DATA_ROOT,
        split="train",
    )

    print("Dataset size:", len(dataset))

    # 先可视化前 3 个样本
    for index in range(min(3, len(dataset))):
        sample = dataset[index]

        image = sample["image"]
        mask = sample["mask"]
        rgb_target = sample["rgb_target"]

        # 保存原图
        Image.fromarray(image).save(RESULT_DIR / f"{index}_image.png")

        # 保存 Train ID 可视化
        Image.fromarray(colorize_mask(mask)).save(RESULT_DIR / f"{index}_mask.png")

        # 保存 RGB Target
        Image.fromarray(rgb_target).save(RESULT_DIR / f"{index}_rgb_target.png")

        print(f"\nSample {index}")
        print("Prompt:", sample["prompt"])
        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        print("RGB Target shape:", rgb_target.shape)
        print("Valid classes:", np.unique(mask).tolist())

    print("\nCityscapes preprocessing completed.")
    print("Results:", RESULT_DIR)


if __name__ == "__main__":
    main()
