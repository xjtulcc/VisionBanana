"""
Semantic Segmentation RGB Decoder

将 RGB 语义分割图像还原为整数类别 Mask。

Input:
    rgb: [H, W, 3], uint8

Output:
    mask: [H, W], int64
"""

import numpy as np

from encoders.semantic_encoder import (
    DEFAULT_IGNORE_COLOR,
    DEFAULT_IGNORE_INDEX,
    DEFAULT_PALETTE,
    validate_palette,
)


def decode_rgb(
    rgb,
    class_to_color=None,
    ignore_index=DEFAULT_IGNORE_INDEX,
    ignore_color=DEFAULT_IGNORE_COLOR,
    unknown_color="ignore",
):
    """
    将 RGB 图像解码为类别 ID Mask。

    Args:
        rgb: [H, W, 3] RGB 图像
        class_to_color: 类别 ID 到 RGB 颜色的映射
        ignore_index: 忽略像素的类别 ID
        ignore_color: 忽略区域颜色
        unknown_color:
            "ignore": 未知颜色标记为 ignore_index
            "raise": 遇到未知颜色直接报错

    Returns:
        mask: [H, W] int64
    """

    if class_to_color is None:
        class_to_color = DEFAULT_PALETTE

    validate_palette(class_to_color)

    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB image must have shape [H, W, 3].")

    if not np.issubdtype(rgb.dtype, np.integer):
        raise TypeError("RGB values must be integers.")

    if np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("RGB values must be in [0, 255].")

    if unknown_color not in ("ignore", "raise"):
        raise ValueError("unknown_color must be 'ignore' or 'raise'.")

    rgb = rgb.astype(np.uint8)

    ignore_color = tuple(ignore_color)

    # 默认所有像素均为 ignore_index
    mask = np.full(rgb.shape[:2], ignore_index, dtype=np.int64)

    # 识别每一种类别颜色
    for class_id, color in class_to_color.items():
        color = np.asarray(color, dtype=np.uint8)

        matched = np.all(rgb == color, axis=-1)

        mask[matched] = class_id

    # 忽略区域
    ignore_matched = np.all(rgb == np.asarray(ignore_color, dtype=np.uint8), axis=-1)

    mask[ignore_matched] = ignore_index

    # 检查是否存在无法识别的颜色
    recognized = ignore_matched.copy()

    for color in class_to_color.values():
        recognized |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)

    unknown = ~recognized

    if np.any(unknown) and unknown_color == "raise":
        unknown_rgb = np.unique(rgb[unknown].reshape(-1, 3), axis=0)

        raise ValueError(f"Unknown RGB colors: {unknown_rgb[:10].tolist()}")

    return mask
