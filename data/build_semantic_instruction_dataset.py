import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data.cityscapes_labels import CLASS_NAMES, LABEL_ID_TO_TRAIN_ID
from data.cityscapes_palette import (
    CITYSCAPES_PALETTE,
    decode_cityscapes_rgb,
    encode_cityscapes_mask,
)

# ==========================================
# 1. Configuration
# ==========================================

CITYSCAPES_ROOT = Path("datasets/cityscapes")

OUTPUT_ROOT = Path("datasets/semantic_instruction")

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

IGNORE_INDEX = 255


# ==========================================
# 2. Label ID -> Train ID
# ==========================================

from data.cityscapes_labels import convert_label_ids

# ==========================================
# 3. Build Prompt
# ==========================================


def build_semantic_prompt():

    palette_description = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        color = CITYSCAPES_PALETTE[class_id]

        r, g, b = map(int, color)

        palette_description.append(f"{class_name}: RGB({r}, {g}, {b})")

    palette_text = "; ".join(palette_description)

    prompt = (
        "Generate a semantic segmentation visualization "
        "image for the input street scene. "
        "Preserve the spatial layout and boundaries of "
        "all visible objects. "
        "Represent each semantic category using a uniform "
        "flat RGB color. "
        "Do not use gradients, textures, shadows or "
        "photographic effects. "
        "Use this exact color mapping: " + palette_text + "."
    )

    return prompt


# ==========================================
# 4. Find Paired Cityscapes Files
# ==========================================


def find_cityscapes_pairs(split):

    image_root = CITYSCAPES_ROOT / "leftImg8bit" / split

    label_root = CITYSCAPES_ROOT / "gtFine" / split

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    if not label_root.exists():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    image_paths = sorted(image_root.rglob("*_leftImg8bit.png"))

    pairs = []

    for image_path in image_paths:
        city = image_path.parent.name

        filename = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png",
        )

        label_path = label_root / city / filename

        if not label_path.exists():
            print(f"[WARNING] Missing label: {label_path}")

            continue

        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError(f"No valid image-label pairs found for {split}")

    return pairs


# ==========================================
# 5. Process One Sample
# ==========================================


def process_sample(
    image_path,
    label_path,
    image_output_path,
    target_output_path,
):

    # Load original RGB image.
    image = Image.open(image_path).convert("RGB")

    # Resize image using bilinear interpolation.
    image = image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR,
    )

    # Load original Cityscapes Label IDs.
    label_ids = np.array(
        Image.open(label_path),
        dtype=np.uint8,
    )

    # Convert Label IDs into Train IDs.
    train_mask = convert_label_ids(label_ids)

    # Resize mask using nearest-neighbor interpolation.
    train_mask = np.array(
        Image.fromarray(train_mask).resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )

    # Encode Train ID mask into RGB target.
    rgb_target = encode_cityscapes_mask(train_mask)

    rgb_target = np.asarray(
        rgb_target,
        dtype=np.uint8,
    )

    # Check output dimensions.
    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    )

    if rgb_target.shape != expected_shape:
        raise ValueError(
            f"Unexpected RGB target shape: "
            f"{rgb_target.shape}, expected {expected_shape}"
        )

    # Verify exact round-trip decoding.
    recovered_mask = decode_cityscapes_rgb(
        rgb_target,
        max_distance=40,
    )

    recovered_mask = np.asarray(
        recovered_mask,
        dtype=np.uint8,
    )

    if not np.array_equal(
        recovered_mask,
        train_mask,
    ):
        mismatch = np.sum(recovered_mask != train_mask)

        raise RuntimeError(
            f"Encoder/Decoder round-trip failed. Mismatched pixels: {mismatch}"
        )

    # Save source image.
    image.save(image_output_path)

    # Save semantic RGB target.
    Image.fromarray(rgb_target).save(target_output_path)

    return train_mask, rgb_target


# ==========================================
# 6. Build Dataset
# ==========================================


def build_dataset(
    split,
    max_samples,
):

    print("\n========== Building Dataset ==========")

    print("Split:", split)

    pairs = find_cityscapes_pairs(split)

    print("Available pairs:", len(pairs))

    if max_samples is not None:
        pairs = pairs[:max_samples]

    print("Selected samples:", len(pairs))

    output_dir = OUTPUT_ROOT / split

    image_dir = output_dir / "images"

    target_dir = output_dir / "targets"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = output_dir / "manifest.jsonl"

    prompt = build_semantic_prompt()

    successful = 0

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as manifest:
        for index, (image_path, label_path) in enumerate(pairs):
            sample_id = image_path.name.replace(
                "_leftImg8bit.png",
                "",
            )

            image_output_path = image_dir / f"{sample_id}.png"

            target_output_path = target_dir / f"{sample_id}_target.png"

            print(f"\n[{index + 1}/{len(pairs)}] Processing {sample_id}")

            train_mask, rgb_target = process_sample(
                image_path,
                label_path,
                image_output_path,
                target_output_path,
            )

            record = {
                "sample_id": sample_id,
                "split": split,
                "image": str(image_output_path.as_posix()),
                "target": str(target_output_path.as_posix()),
                "prompt": prompt,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "num_classes": 19,
                "ignore_index": IGNORE_INDEX,
            }

            manifest.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            successful += 1

            valid_pixels = np.sum(train_mask != IGNORE_INDEX)

            print("Valid pixels:", valid_pixels)

            print("RGB target shape:", rgb_target.shape)

            print("Round-trip: PASSED")

    print("\n========== Dataset Completed ==========")

    print("Successful samples:", successful)

    print("Image directory:", image_dir.resolve())

    print("Target directory:", target_dir.resolve())

    print("Manifest:", manifest_path.resolve())


# ==========================================
# 7. Main
# ==========================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    build_dataset(
        split=args.split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
