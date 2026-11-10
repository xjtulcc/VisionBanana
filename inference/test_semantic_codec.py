from pathlib import Path

import numpy as np
from PIL import Image

from decoders.semantic_decoder import decode_rgb
from encoders.semantic_encoder import (
    DEFAULT_PALETTE,
    encode_mask,
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 创建结果目录
RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 保存 RGB 图像
output_path = RESULT_DIR / "semantic_rgb.png"


def main():

    # 1. 构造模拟语义分割 Mask
    mask = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [3, 3, 1, 1],
            [3, 3, 0, 255],
        ],
        dtype=np.int64,
    )

    print("Original Mask:")
    print(mask)

    # 2. Encoder: Mask -> RGB
    rgb = encode_mask(mask)

    print("\nRGB Shape:", rgb.shape)
    print("RGB Dtype:", rgb.dtype)

    # 3. 保存 RGB 图像
    Image.fromarray(rgb).save(output_path)

    # 4. Decoder: RGB -> Mask
    recovered_mask = decode_rgb(rgb)

    print("\nRecovered Mask:")
    print(recovered_mask)

    # 5. 检查是否完全恢复
    assert np.array_equal(mask, recovered_mask)

    print("\nRound-trip Test: PASSED")

    # 6. 测试未知颜色处理
    test_rgb = rgb.copy()
    test_rgb[0, 0] = (123, 45, 67)

    recovered_ignore = decode_rgb(test_rgb, unknown_color="ignore")

    assert recovered_ignore[0, 0] == 255

    print("Unknown Color Test: PASSED")

    # 7. 输出调色板
    print("\nPalette:")
    for class_id, color in DEFAULT_PALETTE.items():
        print(f"Class {class_id}: RGB {color}")


if __name__ == "__main__":
    main()
