import numpy as np

from encoders.semantic_encoder import encode_mask

# ==========================================
# 1. Cityscapes 19-class RGB Palette
# ==========================================

CITYSCAPES_PALETTE = {
    0: (0, 0, 0),  # road
    1: (128, 64, 192),  # sidewalk
    2: (255, 0, 0),  # building
    3: (0, 255, 0),  # wall
    4: (255, 255, 0),  # fence
    5: (0, 128, 255),  # pole
    6: (255, 128, 0),  # traffic light
    7: (128, 0, 255),  # traffic sign
    8: (0, 255, 255),  # vegetation
    9: (255, 0, 128),  # terrain
    10: (128, 255, 0),  # sky
    11: (0, 128, 128),  # person
    12: (128, 0, 0),  # rider
    13: (0, 0, 255),  # car
    14: (255, 128, 128),  # truck
    15: (128, 128, 0),  # bus
    16: (128, 128, 255),  # train
    17: (255, 0, 255),  # motorcycle
    18: (0, 255, 128),  # bicycle
}


IGNORE_INDEX = 255
IGNORE_COLOR = (1, 1, 1)


# ==========================================
# 2. Encoder
# ==========================================


def encode_cityscapes_mask(mask):
    """
    Train ID Mask -> RGB Target
    """

    return encode_mask(
        mask,
        class_to_color=CITYSCAPES_PALETTE,
        ignore_index=IGNORE_INDEX,
        ignore_color=IGNORE_COLOR,
    )


# ==========================================
# 3. Decoder
# ==========================================


def decode_cityscapes_rgb(
    rgb,
    max_distance=None,
):
    """
    RGB Target -> Cityscapes Train ID Mask.

    Uses nearest-color matching in RGB space.

    Args:
        rgb:
            H x W x 3 RGB array.
            Supports uint8 [0, 255] or float [0, 1].

        max_distance:
            Maximum allowed Euclidean RGB distance.
            Pixels exceeding this threshold become IGNORE_INDEX.
            None means always select the nearest class.

    Returns:
        H x W uint8 Train ID mask.
    """

    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB input, got {rgb.shape}")

    if not np.issubdtype(rgb.dtype, np.number):
        raise TypeError("RGB input must be numeric.")

    if not np.all(np.isfinite(rgb)):
        raise ValueError("RGB input contains NaN or Inf.")

    # Convert normalized float RGB to uint8.
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("RGB values must be in [0, 1] or [0, 255].")

        if rgb.max() <= 1.0:
            rgb = np.rint(rgb * 255)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    else:
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("RGB values must be in [0, 255].")

        rgb = rgb.astype(np.uint8)

    # Flatten image pixels.
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.int32)

    # Build palette arrays.
    class_ids = np.array(
        sorted(CITYSCAPES_PALETTE.keys()),
        dtype=np.uint8,
    )

    colors = np.array(
        [CITYSCAPES_PALETTE[int(i)] for i in class_ids],
        dtype=np.int32,
    )

    # Calculate squared Euclidean distance.
    # Shape: [num_pixels, 19]
    diff = pixels[:, None, :] - colors[None, :, :]

    distances_squared = np.sum(
        diff**2,
        axis=2,
    )

    # Find nearest class color.
    nearest_indices = np.argmin(
        distances_squared,
        axis=1,
    )

    min_distances = np.sqrt(distances_squared[np.arange(len(pixels)), nearest_indices])

    decoded = class_ids[nearest_indices].copy()

    # Restore exact Ignore Color.
    ignore_pixels = np.all(
        pixels == np.array(IGNORE_COLOR, dtype=np.int32),
        axis=1,
    )

    decoded[ignore_pixels] = IGNORE_INDEX

    # Reject colors that are too far from every valid class color.
    if max_distance is not None:
        invalid_pixels = min_distances > max_distance

        decoded[invalid_pixels] = IGNORE_INDEX

    return decoded.reshape(h, w)


# ==========================================
# 4. Palette Information
# ==========================================


def get_cityscapes_palette():
    """
    Return a copy of the Cityscapes RGB Palette.
    """
    return CITYSCAPES_PALETTE.copy()
