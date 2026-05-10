import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum safe number of matches
_MIN_MATCHES = 12

# Geometric thresholds
_MAX_SCALE = 4.0
_MAX_PERSPECTIVE = 0.002
_MIN_DET = 1e-3
_MAX_DET = 4.0


def calculate_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:

    if matches is None or len(matches) < _MIN_MATCHES:
        raise ValueError(
            f"Need at least {_MIN_MATCHES} matches. "
            f"Got {0 if matches is None else len(matches)}."
        )

    # Extract point coordinates
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_thresh
    )

    if H is None:
        raise RuntimeError("Failed to compute homography.")

    if mask is None:
        mask = np.ones((len(matches), 1), dtype=np.uint8)

    # Keep only inlier matches
    inlier_matches = [
        matches[i]
        for i in range(len(matches))
        if mask[i]
    ]

    inliers = int(mask.sum())

    logger.info(
        f"Homography estimated. "
        f"Inliers: {inliers}/{len(matches)}"
    )

    print("\n========== HOMOGRAPHY ==========")
    print(H)

    print("\nPerspective Terms:")
    print("H[2,0] =", H[2, 0])
    print("H[2,1] =", H[2, 1])

    sx = np.linalg.norm(H[0, :2])
    sy = np.linalg.norm(H[1, :2])

    print("\nScale:")
    print("Scale X =", sx)
    print("Scale Y =", sy)

    print("================================\n")

    return H, mask, inlier_matches


def is_homography_valid(H: np.ndarray) -> bool:

    if H is None:
        return False

    if not isinstance(H, np.ndarray):
        return False

    if H.shape != (3, 3):
        return False

    # Rotation + Scale block
    top_left = H[:2, :2]

    det = float(np.linalg.det(top_left))

    # Reflection or collapse
    if det <= 0:
        logger.warning(f"Rejected: negative determinant ({det:.6f})")
        return False

    if det < _MIN_DET:
        logger.warning(f"Rejected: near-singular determinant ({det:.6f})")
        return False

    if det > _MAX_DET:
        logger.warning(f"Rejected: excessive scaling determinant ({det:.6f})")
        return False

    # Perspective sanity
    p1 = abs(float(H[2, 0]))
    p2 = abs(float(H[2, 1]))

    if p1 > _MAX_PERSPECTIVE or p2 > _MAX_PERSPECTIVE:
        logger.warning(
            f"Rejected: excessive perspective "
            f"({p1:.6f}, {p2:.6f})"
        )
        return False

    # Scale sanity
    sx = np.linalg.norm(H[0, :2])
    sy = np.linalg.norm(H[1, :2])

    if sx > _MAX_SCALE or sy > _MAX_SCALE:
        logger.warning(
            f"Rejected: excessive scale ({sx:.2f}, {sy:.2f})"
        )
        return False

    return True