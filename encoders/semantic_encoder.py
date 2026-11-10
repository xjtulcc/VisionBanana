"""
Semantic Segmentation RGB Encoder

将整数语义分割 Mask 转换为 RGB 可视化图像。

Input:
    mask: [H, W], integer class IDs

Output:
    rgb: [H, W, 3], uint8

注意：
    DEFAULT_PALETTE 是演示调色板，不是论文指定的固定调色板。
"""

import numpy as np

# 演示用调色板，后续可以替换为数据集对应的颜色映射
DEFAULT_PALETTE = {
    0: (0, 0, 0),  # Background
    1: (128, 64, 192),  # Road
    2: (255, 0, 0),  # Car
    3: (0, 255, 0),  # Building
}

DEFAULT_IGNORE_INDEX = 255
DEFAULT_IGNORE_COLOR = (1, 1, 1)


def validate_palette(class_to_color):
    """检查类别 ID 和 RGB 颜色映射是否合法。"""

    if not class_to_color:
        raise ValueError("Palette cannot be empty.")

    colors = []

    for class_id, color in class_to_color.items():
        if not isinstance(class_id, (int, np.integer)):
            raise TypeError(f"Invalid class ID: {class_id}")

        if len(color) != 3:
            raise ValueError(f"Invalid RGB color: {color}")

        if any(
            not isinstance(v, (int, np.integer)) or not 0 <= v <= 255 for v in color
        ):
            raise ValueError(f"RGB values must be integers in [0, 255]: {color}")

        colors.append(tuple(int(v) for v in color))

    if len(colors) != len(set(colors)):
        raise ValueError("Different classes cannot share the same RGB color.")


def encode_mask(
    mask,
    class_to_color=None,
    ignore_index=DEFAULT_IGNORE_INDEX,
    ignore_color=DEFAULT_IGNORE_COLOR,
):
    """
    将二维语义分割 Mask 编码为 RGB 图像。

    Args:
        mask: [H, W] 整数类别矩阵
        class_to_color: 类别 ID 到 RGB 颜色的映射
        ignore_index: 忽略像素的类别 ID
        ignore_color: 忽略像素对应的 RGB 颜色

    Returns:
        rgb: [H, W, 3] uint8
    """

    if class_to_color is None:
        class_to_color = DEFAULT_PALETTE

    validate_palette(class_to_color)

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must have shape [H, W].")

    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Mask must contain integer class IDs.")

    ignore_color = tuple(ignore_color)

    if len(ignore_color) != 3 or any(
        not isinstance(v, (int, np.integer)) or not 0 <= v <= 255 for v in ignore_color
    ):
        raise ValueError("ignore_color must be a valid RGB tuple.")

    palette_colors = {tuple(int(v) for v in color) for color in class_to_color.values()}

    if ignore_color in palette_colors:
        raise ValueError("Ignore color conflicts with a class color.")

    known = mask == ignore_index

    for class_id in class_to_color:
        known |= mask == class_id

    if not np.all(known):
        unknown_ids = np.unique(mask[~known])
        raise ValueError(f"Unknown class IDs: {unknown_ids.tolist()}")

    # 初始化 RGB 图像
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # 忽略区域使用专属颜色
    rgb[:] = ignore_color

    # 类别 ID -> RGB
    for class_id, color in class_to_color.items():
        rgb[mask == class_id] = color

    return rgb
