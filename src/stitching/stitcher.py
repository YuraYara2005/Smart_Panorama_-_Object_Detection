from typing import List, Tuple

import cv2
import numpy as np

from src.stitching.homography import (
    calculate_homography,
    is_homography_valid
)

from src.stitching.warping import (
    warp_and_position_images,
    crop_black_borders
)

from src.stitching.blending import feather_blend


def stitch_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if img1 is None or img2 is None:
        raise ValueError("Images cannot be None.")

    print(f"\n[Stitcher] Raw matches: {len(matches)}")

    # ===============================
    # Stage 1 — Homography
    # ===============================

    H, mask, inlier_matches = calculate_homography(
        kp1,
        kp2,
        matches,
        ransac_thresh=3.0
    )
    print("Inlier ratio:", len(inlier_matches) / len(matches))
    print(f"[Stitcher] Inlier matches: {len(inlier_matches)}")

    # ===============================
    # Stage 2 — Validation
    # ===============================

    if not is_homography_valid(H):
        raise RuntimeError(
            "Homography rejected due to unstable geometry."
        )

    print("[Stitcher] Homography accepted.")

    # ===============================
    # Stage 3 — Warping
    # ===============================

    warped_img1, translated_img2 = warp_and_position_images(
        img1,
        img2,
        H
    )

    # ===============================
    # Stage 4 — Blending
    # ===============================

    panorama = feather_blend(
        warped_img1,
        translated_img2
    )

    # ===============================
    # Stage 5 — Crop black borders
    # ===============================

    panorama = crop_black_borders(panorama)

    print(f"[Stitcher] Final panorama shape: {panorama.shape}")

    return panorama, H, mask