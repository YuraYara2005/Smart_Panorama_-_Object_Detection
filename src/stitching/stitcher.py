"""
Orchestrator for the stitching module. Connects homography estimation,
warping, and blending into a single end-to-end pipeline.
"""
from typing import List, Tuple
import cv2
import numpy as np

from src.stitching.homography import calculate_homography, is_homography_valid
from src.stitching.warping import warp_and_position_images
from src.stitching.blending import feather_blend


def stitch_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if img1 is None or img2 is None:
        raise ValueError("Input images cannot be None.")

    print(f"[Stitcher] Starting pipeline. Raw matches: {len(matches) if matches else 0}")

    # Stage 1 — Homography estimation
    H, mask = calculate_homography(kp1, kp2, matches)

    # Stage 2 — Geometric validation

    if not is_homography_valid(H):
        raise RuntimeError(
            "Homography matrix is mathematically degenerate and cannot be used. "
            "The estimated matrix encodes a reflection or a near-singular collapse. "
            "Possible causes: lack of visual overlap, pure forward camera movement, "
            "or a symmetric pattern confusing the feature matcher."
        )

    # Stage 3 — Warping

    print("[Stitcher] Matrix valid. Warping images onto shared canvas...")
    warped_img1, translated_img2 = warp_and_position_images(img1, img2, H)


    # Stage 4 — Blending

    print("[Stitcher] Feather-blending aligned images...")
    panorama = feather_blend(warped_img1, translated_img2)

    print(f"[Stitcher] Success! Final panorama shape: {panorama.shape}")

    return panorama, H, mask