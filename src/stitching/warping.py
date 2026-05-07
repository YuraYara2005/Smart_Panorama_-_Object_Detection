"""
Warping module for the Smart Panorama pipeline.
Applies a Homography matrix to warp a source image onto a dynamically-sized
canvas, ensuring no pixels are lost into negative coordinates.
"""
import logging
from typing import Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

ImageShape = Tuple[int, int]
CanvasSize = Tuple[int, int]


def calculate_panorama_canvas(
    img1_shape: ImageShape,
    img2_shape: ImageShape,
    H: np.ndarray,
) -> Tuple[CanvasSize, np.ndarray]:
    """
    Computes the new canvas dimensions and the translation matrix (T) required
    to hold both images without clipping any negative coordinates.
    """
    if not isinstance(H, np.ndarray) or H.shape != (3, 3):
        raise ValueError("H must be a valid (3, 3) numpy array.")

    h1, w1 = img1_shape
    h2, w2 = img2_shape

    # 1. Define corners of Image 1 and project them through H
    corners_img1 = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    if warped_corners is None:
        raise RuntimeError("cv2.perspectiveTransform failed. H may be degenerate.")

    # 2. Combine with Image 2 corners to find the absolute bounding box
    corners_img2 = np.float32([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]])
    all_corners = np.vstack([warped_corners.reshape(4, 2), corners_img2])

    x_min, y_min = all_corners[:, 0].min(), all_corners[:, 1].min()
    x_max, y_max = all_corners[:, 0].max(), all_corners[:, 1].max()

    # 3. Calculate translation offsets (clip at 0 to avoid unnecessary shifts)
    tx = float(max(0.0, -x_min))
    ty = float(max(0.0, -y_min))

    # 4. Build translation matrix T
    translation_matrix = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 5. Calculate final canvas size
    new_width = int(np.ceil(x_max + tx)) + 1
    new_height = int(np.ceil(y_max + ty)) + 1

    return (new_width, new_height), translation_matrix


def warp_and_position_images(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps img1 onto the shared canvas and places img2 at its translated position.
    Returns two identically-shaped images ready for blending.
    """
    if img1 is None or img2 is None:
        raise ValueError("Input images cannot be None.")

    # 1. Get canvas geometry
    canvas_size, T = calculate_panorama_canvas(img1.shape[:2], img2.shape[:2], H)
    new_width, new_height = canvas_size

    # 2. Compose the transform (T @ H) to warp and translate in a single step
    T_H = np.dot(T, H)

    # 3. Warp Image 1
    warped_img1 = cv2.warpPerspective(
        img1, T_H, (new_width, new_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 4. Position Image 2 on a blank canvas
    tx, ty = int(round(T[0, 2])), int(round(T[1, 2]))

    canvas_shape = (new_height, new_width) if img2.ndim == 2 else (new_height, new_width, img2.shape[2])
    translated_img2 = np.zeros(canvas_shape, dtype=img2.dtype)

    # Safely slice Image 2 into the canvas
    row_end = min(ty + img2.shape[0], new_height)
    col_end = min(tx + img2.shape[1], new_width)
    translated_img2[ty:row_end, tx:col_end] = img2[:(row_end - ty), :(col_end - tx)]

    return warped_img1, translated_img2


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    Finds the largest axis-aligned rectangle fully inside the valid
    (non-black) region of the panorama and returns it cropped.
    Works for any warp direction — horizontal, vertical, or mixed.
    """
    if image is None:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape
    best_area = 0
    best_rect = (0, 0, w, h)  # fallback: full image

    # For every possible top and bottom row pair,
    # find the widest horizontal span fully inside the mask
    # Use row histograms for efficiency
    heights = np.zeros(w, dtype=np.int32)

    for row in range(h):
        # Update histogram: how many consecutive valid rows ending at this row per column
        heights = np.where(mask[row] == 1, heights + 1, 0)

        # Largest rectangle in this histogram (stack-based O(w))
        stack = []
        for col in range(w + 1):
            h_cur = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] > h_cur:
                s_col, s_h = stack.pop()
                area = (col - s_col) * s_h
                if area > best_area:
                    best_area = area
                    # Rectangle: bottom = row, top = row - s_h + 1
                    best_rect = (s_col, row - s_h + 1, col, row + 1)
                start = s_col
            stack.append((start, h_cur))

    x1, y1, x2, y2 = best_rect
    return image[y1:y2, x1:x2]