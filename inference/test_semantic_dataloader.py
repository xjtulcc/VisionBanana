from pathlib import Path

import numpy as np
from PIL import Image

from data.semantic_instruction_dataset import (
    create_dataloader,
)


def tensor_to_image(tensor):

    # [-1, 1] -> [0, 255].
    image = ((tensor.cpu().numpy() + 1.0) * 127.5).clip(0, 255).round().astype(np.uint8)

    # [C, H, W] -> [H, W, C].
    image = image.transpose(1, 2, 0)

    return Image.fromarray(image)


def main():

    manifest_path = "datasets/semantic_instruction/train/manifest.jsonl"

    dataloader = create_dataloader(
        manifest_path=manifest_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print("\n========== DataLoader Test ==========")

    for batch_index, batch in enumerate(dataloader):
        source = batch["conditioning_pixel_values"]

        target = batch["pixel_values"]

        prompts = batch["prompt"]

        sample_ids = batch["sample_id"]

        print(f"\nBatch: {batch_index}")

        print("Sample IDs:", sample_ids)

        print("Source Shape:", source.shape)

        print("Target Shape:", target.shape)

        print("Number of Prompts:", len(prompts))

        print("Source Range:", source.min().item(), source.max().item())

        print("Target Range:", target.min().item(), target.max().item())

        print("\nPrompt:")
        print(prompts[0])

        # Check tensor shapes.
        assert source.shape == target.shape

        assert source.ndim == 4

        assert source.shape[1] == 3

        # Check normalization.
        assert source.min() >= -1.001

        assert source.max() <= 1.001

        assert target.min() >= -1.001

        assert target.max() <= 1.001

        # Save first batch sample for visual inspection.
        output_dir = Path("results/dataloader_test")

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tensor_to_image(source[0]).save(output_dir / "source.png")

        tensor_to_image(target[0]).save(output_dir / "target.png")

        print("\nSaved:")
        print(output_dir.resolve())

        # Only inspect the first batch.
        break

    print("\nDataLoader Test: PASSED")


if __name__ == "__main__":
    main()
