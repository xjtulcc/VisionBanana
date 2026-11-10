from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.cityscapes_labels import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    convert_label_ids,
)
from data.cityscapes_palette import encode_cityscapes_mask


class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset for Vision Banana.

    Data flow:
        RGB Image + Label ID
            -> Train ID Mask
            -> Prompt
            -> Optional RGB Target

    Expected directory structure:

        cityscapes/
        ├── leftImg8bit/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── gtFine/
            ├── train/
            ├── val/
            └── test/
    """

    def __init__(
        self,
        root,
        split="train",
        prompt="Segment all objects in the image.",
        image_size=None,
        encode_mask: Optional[Callable] = None,
        return_paths=False,
    ):
        """
        Args:
            root:
                Cityscapes dataset root directory.

            split:
                train, val or test.

            prompt:
                Fixed natural language instruction.

            image_size:
                Optional (height, width).
                Images and masks are resized together.

            encode_mask:
                Optional Semantic Encoder function.
                Input: 2D numpy Train ID mask.
                Output: H x W x 3 RGB encoded target.

            return_paths:
                Whether to return image and mask paths.
        """

        self.root = Path(root)
        self.split = split
        self.prompt = prompt
        self.image_size = image_size
        self.encode_mask = encode_mask
        self.return_paths = return_paths

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}. Choose train, val or test.")

        self.image_dir = self.root / "leftImg8bit" / split
        self.mask_dir = self.root / "gtFine" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Ground Truth directory not found: {self.mask_dir}"
            )

        # Find all original RGB images.
        image_paths = sorted(self.image_dir.rglob("*_leftImg8bit.png"))

        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        # Pair RGB images with corresponding labelIds masks.
        self.samples = []

        missing_masks = []

        for image_path in image_paths:
            # Example:
            # aachen_000000_000019_leftImg8bit.png
            #
            # becomes:
            # aachen_000000_000019_gtFine_labelIds.png

            filename = image_path.name.replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"
            )

            relative_city = image_path.parent.relative_to(self.image_dir)

            mask_path = self.mask_dir / relative_city / filename

            if not mask_path.exists():
                missing_masks.append(mask_path)
                continue

            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No valid image-mask pairs found in split={split}. "
                f"Missing masks: {len(missing_masks)}"
            )

        if missing_masks:
            print(
                f"Warning: {len(missing_masks)} images "
                "have no matching masks and were skipped."
            )

        print(f"Cityscapes {split} dataset initialized.")
        print(f"Valid samples: {len(self.samples)}")
        print(f"Number of classes: {NUM_CLASSES}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        """Load RGB image."""

        image = Image.open(image_path).convert("RGB")

        if self.image_size is not None:
            height, width = self.image_size

            image = image.resize((width, height), resample=Image.Resampling.BILINEAR)

        image = np.array(image, dtype=np.uint8)

        # H x W x 3 -> 3 x H x W
        image = torch.from_numpy(image.copy())
        image = image.permute(2, 0, 1).float() / 255.0

        return image

    def _load_mask(self, mask_path):
        """Load and convert original Label IDs to Train IDs."""

        mask = Image.open(mask_path)

        # Keep original integer Label IDs.
        mask = np.array(mask, dtype=np.uint8)

        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D label mask, got {mask.shape}")

        # Convert Cityscapes Label IDs into 19 Train IDs.
        mask = convert_label_ids(mask)

        if self.image_size is not None:
            height, width = self.image_size

            mask_image = Image.fromarray(mask)

            # NEAREST is essential:
            # semantic class IDs must not be interpolated.
            mask_image = mask_image.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

            mask = np.array(mask_image, dtype=np.uint8)

        return mask

    def _encode_target(self, mask):
        """
        Convert Train ID mask into RGB target.

        The actual encoding implementation is provided
        by the existing Semantic Encoder.
        """

        if self.encode_mask is None:
            return None

        rgb = self.encode_mask(mask)

        rgb = np.asarray(rgb)

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Semantic Encoder must return an H x W x 3 RGB array.")

        if rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"RGB target shape {rgb.shape[:2]} "
                f"does not match mask shape {mask.shape}."
            )

        if rgb.dtype != np.uint8:
            if not np.issubdtype(rgb.dtype, np.number):
                raise TypeError("RGB target must contain numeric values.")

            if not np.all(np.isfinite(rgb)):
                raise ValueError("RGB target contains NaN or Inf.")

            if rgb.min() < 0 or rgb.max() > 255:
                raise ValueError("RGB target values must be in [0, 255].")

            rgb = np.rint(rgb).astype(np.uint8)

        # H x W x 3 -> 3 x H x W
        target = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        )

        return target

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # 1. Load original RGB image.
        image = self._load_image(image_path)

        # 2. Load and convert semantic mask.
        mask = self._load_mask(mask_path)

        # 3. Convert mask into PyTorch Tensor.
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        # 4. Construct prompt.
        prompt = self.prompt

        # 5. Generate RGB supervision target if encoder supplied.
        rgb_target = self._encode_target(mask)

        sample = {
            "image": image,
            "mask": mask_tensor,
            "prompt": prompt,
        }

        # Only include this field when an encoder is supplied.
        if rgb_target is not None:
            sample["rgb_target"] = rgb_target

        if self.return_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)

        return sample


if __name__ == "__main__":
    dataset = CityscapesDataset(
        root="datasets/cityscapes",
        split="train",
        prompt="Segment all objects in the image.",
        image_size=(512, 1024),
        encode_mask=encode_cityscapes_mask,
    )

    sample = dataset[0]

    print("\nSample information:")
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Prompt:", sample["prompt"])

    print("Unique Train IDs:", torch.unique(sample["mask"]).tolist())
    print("RGB target shape:", sample["rgb_target"].shape)
