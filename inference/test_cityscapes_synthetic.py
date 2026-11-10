import numpy as np

from datasets.cityscapes import (
    CITYSCAPES_PALETTE,
    convert_label_ids,
)
from decoders.semantic_decoder import decode_rgb
from encoders.semantic_encoder import encode_mask


def main():

    # 模拟 Cityscapes 原始 Label ID
    label_ids = np.array(
        [
            [7, 7, 8, 8],
            [11, 11, 26, 26],
            [23, 21, 24, 25],
            [0, 1, 3, 33],
        ],
        dtype=np.uint8,
    )

    # 1. Label ID -> Train ID
    mask = convert_label_ids(label_ids)

    print("Train ID Mask:")
    print(mask)

    # 2. Train ID -> RGB
    rgb = encode_mask(
        mask,
        class_to_color=CITYSCAPES_PALETTE,
        ignore_index=255,
    )

    print("\nRGB shape:", rgb.shape)

    # 3. RGB -> Train ID
    recovered = decode_rgb(
        rgb,
        class_to_color=CITYSCAPES_PALETTE,
        ignore_index=255,
    )

    # 4. 验证可逆性
    assert np.array_equal(mask, recovered)

    print("\nCityscapes Label Mapping: PASSED")
    print("RGB Encoding: PASSED")
    print("RGB Decoding: PASSED")

    print("\nTrain IDs:", np.unique(mask))


if __name__ == "__main__":
    main()
