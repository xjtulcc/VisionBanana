from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from PIL import Image

from data.cityscapes_palette import (
    CITYSCAPES_PALETTE,
    IGNORE_INDEX,
    decode_cityscapes_rgb,
)

# ==========================================
# 1. Configuration
# ==========================================

MODEL_PATH = "models/sdxl-instructpix2pix"

IMAGE_PATH = (
    "datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
)

MASK_PATH = (
    "datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png"
)

OUTPUT_DIR = Path("results/semantic_generation")

IMAGE_SIZE = (1024, 512)

MAX_DISTANCE = 40

DEVICE = "cuda"


# ==========================================
# 2. Build Semantic Prompt
# ==========================================

CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def convert_label_ids_to_train_ids(label_ids):

    label_id_to_train_id = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18,
    }

    train_mask = np.full(
        label_ids.shape,
        255,
        dtype=np.uint8,
    )

    for label_id, train_id in label_id_to_train_id.items():
        train_mask[label_ids == label_id] = train_id

    return train_mask


def build_prompt():

    palette_description = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        color = CITYSCAPES_PALETTE[class_id]

        r, g, b = color

        palette_description.append(f"{class_name}: RGB({r},{g},{b})")

    palette_text = "; ".join(palette_description)

    prompt = (
        "Transform the input street scene into a semantic "
        "segmentation visualization. Preserve the original "
        "spatial layout and object boundaries. Represent each "
        "semantic category using a uniform flat RGB color. "
        "Do not use gradients, textures, shadows or artistic "
        "effects. Use these exact semantic color assignments: "
        + palette_text
        + ". The result must be a clean flat-color semantic "
        "segmentation map."
    )

    return prompt


# ==========================================
# 3. Load Pipeline
# ==========================================


def load_pipeline():

    print("Loading SDXL InstructPix2Pix...")

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # Use CPU offloading for the 8GB GPU.
    pipe.enable_model_cpu_offload()

    print("Pipeline loaded.")

    return pipe


# ==========================================
# 4. Load Input
# ==========================================


def load_input():

    image = Image.open(IMAGE_PATH).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BICUBIC,
    )

    label_ids = np.array(
        Image.open(MASK_PATH),
        dtype=np.uint8,
    )

    gt_mask = convert_label_ids_to_train_ids(label_ids)

    gt_mask = np.array(
        Image.fromarray(gt_mask).resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )

    return image, gt_mask


# ==========================================
# 5. Generate Semantic RGB
# ==========================================


def generate_semantic_rgb(pipe, image):

    prompt = build_prompt()

    print("\nPrompt:")
    print(prompt)

    print("\nGenerating semantic RGB...")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=30,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
        )

    generated_image = result.images[0].convert("RGB")

    # Ensure generated image has the same size as input.
    generated_image = generated_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BICUBIC,
    )

    print("Generated image size:", generated_image.size)

    return generated_image


# ==========================================
# 6. Decode Generated RGB
# ==========================================


def decode_generated_image(generated_image):

    rgb = np.array(
        generated_image,
        dtype=np.uint8,
    )

    decoded_mask = decode_cityscapes_rgb(
        rgb,
        max_distance=MAX_DISTANCE,
    )

    return rgb, np.asarray(decoded_mask)


# ==========================================
# 7. Evaluation
# ==========================================


def evaluate(gt_mask, pred_mask):

    valid = gt_mask != IGNORE_INDEX

    accuracy = (pred_mask[valid] == gt_mask[valid]).mean() * 100

    accepted = pred_mask[valid] != IGNORE_INDEX

    coverage = accepted.mean() * 100

    if accepted.any():
        accepted_accuracy = (
            pred_mask[valid][accepted] == gt_mask[valid][accepted]
        ).mean() * 100

    else:
        accepted_accuracy = 0.0

    # Calculate mIoU for 19 semantic classes.
    ious = []

    for class_id in range(19):
        gt_class = gt_mask == class_id

        pred_class = pred_mask == class_id

        intersection = np.logical_and(
            gt_class,
            pred_class,
        ).sum()

        union = np.logical_or(
            gt_class,
            pred_class,
        ).sum()

        if union > 0:
            ious.append(intersection / union)

    miou = np.mean(ious) * 100 if ious else 0.0

    print("\n========== Evaluation ==========")

    print(f"Pixel Accuracy: {accuracy:.4f}%")

    print(f"Coverage: {coverage:.4f}%")

    print(f"Accepted Accuracy: {accepted_accuracy:.4f}%")

    print(f"mIoU: {miou:.4f}%")

    return {
        "pixel_accuracy": accuracy,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "mIoU": miou,
    }


# ==========================================
# 8. Main
# ==========================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image, gt_mask = load_input()

    image.save(OUTPUT_DIR / "input.png")

    # Load model.
    pipe = load_pipeline()

    # Generate RGB segmentation visualization.
    generated_image = generate_semantic_rgb(
        pipe,
        image,
    )

    generated_image.save(OUTPUT_DIR / "generated_rgb.png")

    # Decode RGB.
    rgb, pred_mask = decode_generated_image(
        generated_image,
    )

    # Save RGB and decoded mask.
    Image.fromarray(rgb).save(OUTPUT_DIR / "generated_rgb_array.png")

    Image.fromarray(pred_mask.astype(np.uint8)).save(OUTPUT_DIR / "decoded_mask.png")

    # Evaluate against Ground Truth.
    evaluate(
        gt_mask,
        pred_mask,
    )

    print("\nGeneration completed.")

    print("Results:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
