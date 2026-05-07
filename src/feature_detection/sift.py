import cv2
import numpy as np
from typing import Tuple, List


def detect_sift_features(image: np.ndarray):
    """
    Detect SIFT keypoints and descriptors.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def draw_keypoints(image: np.ndarray,
                   keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    Draw SIFT keypoints on image.
    """

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return output