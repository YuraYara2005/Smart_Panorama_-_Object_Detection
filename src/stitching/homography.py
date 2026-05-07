"""
Homography estimation module for the Smart Panorama & Object Recognition pipeline.

Computes a robust 3x3 Homography transformation matrix from raw keypoint matches 
using the RANSAC algorithm. Includes geometric sanity checks to prevent degenerate warps.
"""

import logging
from typing import List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants for safety checks
_HOMOGRAPHY_SAFE_MIN_POINTS: int = 10
_DET_NEAR_ZERO_THRESH: float = 1e-3

def calculate_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a 3x3 Homography matrix from raw keypoint matches using RANSAC.
    """
    if not matches:
        raise ValueError("Cannot compute homography: the 'matches' list is empty.")

    n_matches: int = len(matches)
    if n_matches < _HOMOGRAPHY_SAFE_MIN_POINTS:
        raise ValueError(
            f"Insufficient matches. Got {n_matches}, require at least {_HOMOGRAPHY_SAFE_MIN_POINTS} "
            f"for a stable matrix."
        )

    # Extract matched (x, y) pixel coordinates
    src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography via RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None:
        raise RuntimeError(
            "cv2.findHomography returned None. The scene may be degenerate "
            "(e.g., the camera did not move or images do not overlap)."
        )

    # Defensive fallback if mask is None but H is valid
    if mask is None:
        mask = np.ones((n_matches, 1), dtype=np.uint8)

    n_inliers: int = int(mask.sum())
    logger.info(f"Homography estimated — inliers: {n_inliers} / {n_matches} ({(100.0 * n_inliers / n_matches):.1f}%)")

    return H, mask

def is_homography_valid(H: np.ndarray) -> bool:
    """
    Perform a geometric sanity check on a 3x3 Homography matrix to prevent 
    reflections or near-singular collapses.
    """
    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
        return False

    # Check determinant of the top-left 2x2 sub-matrix (rotation + scale)
    top_left_2x2: np.ndarray = H[:2, :2]
    det: float = float(np.linalg.det(top_left_2x2))

    # Negative determinant -> reflection (inside-out)
    if det <= 0:
        logger.warning(f"Homography rejected: determinant is {det:.6f} (reflection detected).")
        return False

    # Near-zero determinant -> singular/degenerate collapse
    if det < _DET_NEAR_ZERO_THRESH:
        logger.warning(f"Homography rejected: determinant is {det:.6f} (near-singular collapse).")
        return False

    return True