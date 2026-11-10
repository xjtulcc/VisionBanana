import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SemanticInstructionDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        image_size=(1024, 512),
    ):
        self.manifest_path = Path(manifest_path)

        self.image_size = image_size

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Read manifest.
        self.samples = []

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        if not self.samples:
            raise RuntimeError("Dataset is empty.")

        # Resolve paths relative to project root.
        self.project_root = Path.cwd()

        print("Semantic Instruction Dataset initialized.")

        print("Manifest:", self.manifest_path)

        print("Number of samples:", len(self.samples))

    def __len__(self):

        return len(self.samples)

    def image_to_tensor(self, image):

        # PIL RGB -> NumPy.
        image = np.array(
            image,
            dtype=np.float32,
        )

        # [H, W, C] -> [C, H, W].
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())

        # Normalize [0, 255] -> [-1, 1].
        image = image / 127.5 - 1.0

        return image

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = self.project_root / sample["image"]

        target_path = self.project_root / sample["target"]

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not target_path.exists():
            raise FileNotFoundError(target_path)

        # Load source image.
        source_image = Image.open(image_path).convert("RGB")

        # Load target semantic RGB image.
        target_image = Image.open(target_path).convert("RGB")

        # Ensure identical spatial dimensions.
        source_image = source_image.resize(
            self.image_size,
            Image.Resampling.BILINEAR,
        )

        target_image = target_image.resize(
            self.image_size,
            Image.Resampling.NEAREST,
        )

        # Convert images to tensors.
        source_tensor = self.image_to_tensor(source_image)

        target_tensor = self.image_to_tensor(target_image)

        prompt = sample["prompt"]

        return {
            "pixel_values": target_tensor,
            "conditioning_pixel_values": source_tensor,
            "prompt": prompt,
            "sample_id": sample["sample_id"],
        }


# ==========================================
# DataLoader Factory
# ==========================================


def create_dataloader(
    manifest_path,
    batch_size=2,
    shuffle=True,
    num_workers=0,
):

    dataset = SemanticInstructionDataset(
        manifest_path=manifest_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader
