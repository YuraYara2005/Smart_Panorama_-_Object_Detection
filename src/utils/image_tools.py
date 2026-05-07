"""
Utility functions for cleaning, padding, and formatting image arrays
in the Smart Panorama pipeline.
"""

import cv2
import numpy as np


def crop_black_borders(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img  # entire image is black — return as-is

    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]


def pad_image(img: np.ndarray, padding: int = 50) -> np.ndarray:
    return cv2.copyMakeBorder(
        img,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img  # unexpected format — return unchanged