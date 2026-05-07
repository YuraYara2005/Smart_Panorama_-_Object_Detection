"""
Blending module for the Smart Panorama pipeline.
Composites two aligned images into a single seamless panorama using
distance-transform alpha feathering to hide exposure differences at the seam.
"""

import cv2
import numpy as np

# Epsilon to prevent division-by-zero in fully black canvas regions
_BLEND_EPSILON = 1e-7

def create_weight_mask(img: np.ndarray) -> np.ndarray:
    """
    Builds a normalized float weight mask [0.0, 1.0] for an image based on
    the Euclidean distance of each pixel to the nearest black border.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a numpy ndarray.")

    # 1. Create a binary mask (255 for valid pixels, 0 for black borders)
    if img.ndim == 2:
        gray = img.astype(np.uint8)
    else:
        src = img if img.dtype == np.uint8 else img.astype(np.uint8)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # 2. Guard against completely black images
    if cv2.countNonZero(binary_mask) == 0:
        return np.zeros(img.shape[:2], dtype=np.float32)

    # 3. Compute distance transform
    dist = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)

    # 4. Normalize to [0.0, 1.0]
    dist_max = float(dist.max())
    if dist_max < _BLEND_EPSILON:
        return np.zeros_like(dist)

    return dist / dist_max


def feather_blend(warped_img1: np.ndarray, translated_img2: np.ndarray) -> np.ndarray:
    """
    Blends two identically-shaped images using their distance-transform weight masks.
    """
    if warped_img1.shape != translated_img2.shape:
        raise ValueError("Input images must have identical shapes for blending.")

    # 1. Generate weight masks
    mask1 = create_weight_mask(warped_img1)
    mask2 = create_weight_mask(translated_img2)

    # 2. Expand masks to 3D for broadcasting over color channels
    if warped_img1.ndim == 3:
        mask1_3d = mask1[:, :, np.newaxis]
        mask2_3d = mask2[:, :, np.newaxis]
    else:
        mask1_3d, mask2_3d = mask1, mask2

    # 3. Convert images to float32 for precise arithmetic
    img1_f = warped_img1.astype(np.float32)
    img2_f = translated_img2.astype(np.float32)

    # 4. Perform weighted pixel-wise addition
    numerator = (img1_f * mask1_3d) + (img2_f * mask2_3d)

    # Add epsilon to the denominator to prevent division by zero on black background pixels
    total_weight = mask1 + mask2
    if warped_img1.ndim == 3:
        denominator = total_weight[:, :, np.newaxis] + _BLEND_EPSILON
    else:
        denominator = total_weight + _BLEND_EPSILON

    blended_f = numerator / denominator

    # 5. Clip to valid pixel range and convert back to uint8
    return np.clip(blended_f, 0, 255).astype(np.uint8)