import numpy as np


# Cityscapes 原始 Label ID -> 连续 Train ID
# 19 个有效语义类别，其他类别统一设为 255 (ignore)

LABEL_ID_TO_TRAIN_ID = {
    7: 0,  # road
    8: 1,  # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}


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


NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


def convert_label_ids(label_ids: np.ndarray) -> np.ndarray:
    """
    将 Cityscapes 原始 Label ID 转换为连续 Train ID。

    Args:
        label_ids: 原始二维 Label ID 数组。

    Returns:
        train_ids: 连续类别 ID，范围 0~18，
                   无效或忽略类别为 255。
    """
    train_ids = np.full(label_ids.shape, IGNORE_INDEX, dtype=np.uint8)

    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        train_ids[label_ids == label_id] = train_id

    return train_ids
