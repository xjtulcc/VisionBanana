import numpy as np
import torch

from data.cityscapes import CityscapesDataset
from data.cityscapes_palette import (
    encode_cityscapes_mask,
    decode_cityscapes_rgb,
)


def main():

    # 1. 加载 Cityscapes 数据集
    dataset = CityscapesDataset(
        root="datasets/cityscapes",
        split="train",
        prompt="Segment all objects in the image.",
        image_size=(512, 1024),
        encode_mask=encode_cityscapes_mask,
        return_paths=True,
    )

    # 2. 获取真实样本
    sample = dataset[0]

    mask = sample["mask"].cpu().numpy().astype(np.uint8)

    rgb_target = sample["rgb_target"].permute(1, 2, 0).cpu().numpy()

    rgb_target = np.rint(rgb_target * 255).clip(0, 255).astype(np.uint8)

    # 3. RGB -> Mask
    recovered_mask = decode_cityscapes_rgb(rgb_target)
    recovered_mask = np.asarray(recovered_mask)

    # 4. 检查尺寸
    print("Original Mask:", mask.shape)
    print("RGB Target:", rgb_target.shape)
    print("Recovered Mask:", recovered_mask.shape)

    assert mask.shape == recovered_mask.shape

    # 5. 检查有效类别准确率
    valid = mask != 255

    correct = mask[valid] == recovered_mask[valid]

    accuracy = correct.mean() * 100

    print(f"\nValid Pixel Accuracy: {accuracy:.6f}%")

    # 6. 检查 Ignore 区域
    ignore_correct = np.all(recovered_mask[~valid] == 255)

    print("Ignore Region Correct:", ignore_correct)

    # 7. 检查错误像素
    mismatch = (mask != recovered_mask) & valid

    mismatch_count = int(mismatch.sum())

    print("Mismatched Pixels:", mismatch_count)

    if mismatch_count > 0:
        positions = np.argwhere(mismatch)

        print("\nMismatch Examples:")

        for y, x in positions[:10]:
            print(
                f"Position ({y}, {x}): "
                f"Original={mask[y, x]}, "
                f"Decoded={recovered_mask[y, x]}, "
                f"RGB={rgb_target[y, x].tolist()}"
            )

    # 8. 最终判断
    if accuracy == 100.0 and ignore_correct:
        print("\nCityscapes Codec Round-trip: PASSED")

    else:
        print("\nCityscapes Codec Round-trip: FAILED")


if __name__ == "__main__":
    main()
