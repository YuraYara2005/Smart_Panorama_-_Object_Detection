"""
OpenCV drawing utilities for visually debugging keypoint matches
and bounding-box geometry in the Smart Panorama pipeline.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_ransac_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    mask: np.ndarray,
) -> np.ndarray:

    # cv2.drawMatches requires a flat Python list of 0/1 integers
    matches_mask: Optional[List[int]] = mask.ravel().tolist() if mask is not None else None

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=matches_mask,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)


def draw_warped_outline(
    img_canvas: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:

    """Draw a closed polygon over the transformed image corners on a canvas copy.
    Returns the annotated canvas without modifying the original array."""
    vis = img_canvas.copy()

    # polylines requires shape (N, 1, 2) int32
    pts = np.int32(corners).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

    return vis
