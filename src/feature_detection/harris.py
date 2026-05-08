import cv2
import numpy as np
from typing import List, Tuple


# =====================================================
# INTERNAL HELPER
# =====================================================

def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale float32.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    return gray.astype(np.float32)


# =====================================================
# HARRIS RESPONSE
# =====================================================

def compute_harris_response(
    image: np.ndarray,
    block_size: int = 3,
    ksize: int = 3,
    k: float = 0.04
) -> np.ndarray:
    """
    Compute Harris corner response map.

    R = det(M) - k(trace(M)^2)
    """

    gray = _to_gray(image)

    response = cv2.cornerHarris(
        gray,
        blockSize=block_size,
        ksize=ksize,
        k=k
    )

    return response


# =====================================================
# HARRIS DETECTOR
# =====================================================

def detect_harris_corners(
    image: np.ndarray,
    block_size: int = 3,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    nms_radius: int = 5
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect Harris corners with:
    - thresholding
    - non-maximum suppression
    """

    response = compute_harris_response(
        image,
        block_size,
        ksize,
        k
    )

    # =================================================
    # THRESHOLDING
    # =================================================

    threshold = threshold_ratio * response.max()

    corners_mask = (
        response > threshold
    ).astype(np.uint8)

    # =================================================
    # NON-MAXIMUM SUPPRESSION
    # =================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * nms_radius + 1,
         2 * nms_radius + 1)
    )

    dilated = cv2.dilate(
        response,
        kernel
    )

    local_max_mask = (
        (response == dilated) &
        (corners_mask > 0)
    ).astype(np.uint8)

    ys, xs = np.where(local_max_mask > 0)

    # =================================================
    # CONVERT TO KEYPOINTS
    # =================================================

    keypoints = [
        cv2.KeyPoint(
            float(x),
            float(y),
            float(nms_radius * 2)
        )
        for y, x in zip(ys, xs)
    ]

    # Sort strongest corners first
    keypoints.sort(
        key=lambda kp: -response[
            int(kp.pt[1]),
            int(kp.pt[0])
        ]
    )

    return keypoints, response


# =====================================================
# VISUALIZATION
# =====================================================

def draw_harris_corners(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    max_corners: int = 500
) -> np.ndarray:
    """
    Draw Harris corners on image.
    """

    if image.ndim == 3:
        output = image.copy()
    else:
        output = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    for kp in keypoints[:max_corners]:

        x = int(kp.pt[0])
        y = int(kp.pt[1])

        # Outer circle
        cv2.circle(
            output,
            (x, y),
            4,
            (0, 0, 255),
            1
        )

        # Center point
        cv2.circle(
            output,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    return output