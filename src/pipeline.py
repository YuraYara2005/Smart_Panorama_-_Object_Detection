import cv2
import numpy as np
import yaml
from typing import Dict, List, Optional

from src.preprocessing.filters import (
    gaussian_filter_cv,
    median_filter_cv,
    add_noise
)
from src.preprocessing.evaluation import compute_all_metrics
from src.pyramids.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian
)
from src.feature_detection.detector import detect_features
from src.matching.matcher import match_features
from src.stitching.stitcher import stitch_image_pair
from src.segmentation.segmenter import segment

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Loads the config.yaml file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_preprocessing(image: np.ndarray, settings: Dict) -> Dict:
    """
    Runs Gaussian and Median filtering on the image.
    Returns original, filtered images and their metrics.
    """
    # Add noise first so we can measure how well filters clean it
    noisy = add_noise(
        image,
        noise_type=settings["noise_type"],
        intensity=settings["noise_intensity"]
    )

    # Apply filters
    gaussian_result = gaussian_filter_cv(
        noisy,
        kernel_size=settings["gaussian_kernel_size"],
        sigma=settings["gaussian_sigma"]
    )

    median_result = median_filter_cv(
        noisy,
        kernel_size=settings["median_kernel_size"]
    )

    # Compute metrics comparing filtered results to original clean image
    metrics_gaussian = compute_all_metrics(
        image, gaussian_result, label="Gaussian Filter"
    )
    metrics_median = compute_all_metrics(
        image, median_result, label="Median Filter"
    )

    return {
        "original":        image,
        "noisy":           noisy,
        "gaussian":        gaussian_result,
        "median":          median_result,
        "metrics_gaussian": metrics_gaussian,
        "metrics_median":   metrics_median,
    }


def run_pyramids(image: np.ndarray, settings: Dict) -> Dict:
    """
    Builds Gaussian and Laplacian pyramids and reconstructs the image.
    """
    gaussian_pyramid = build_gaussian_pyramid(
        image,
        levels=settings["pyramid_levels"],
        sigma=settings["pyramid_sigma"]
    )

    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

    reconstructed = reconstruct_from_laplacian(laplacian_pyramid)

    return {
        "original":          image,
        "gaussian_pyramid":  gaussian_pyramid,
        "laplacian_pyramid": laplacian_pyramid,
        "reconstructed":     reconstructed,
    }


def run_pipeline(images: List[np.ndarray], settings: Dict) -> Dict:
    """
    Main orchestrator function.
    Runs all active modules in order and collects results.
    """
    if not images:
        return {}

    # Use the first image for preprocessing and pyramids
    image = images[0]

    results = {}

    # ── Step 1: Preprocessing ──
    if settings.get("run_preprocessing", False):
        print("[Pipeline] Running preprocessing...")
        results["preprocessing"] = run_preprocessing(image, settings)
        print("[Pipeline] Preprocessing done.")

    # ── Step 2: Pyramids ──
    if settings.get("run_pyramids", False):
        print("[Pipeline] Running pyramids...")
        results["pyramids"] = run_pyramids(image, settings)
        print("[Pipeline] Pyramids done.")

    # ── Step 3: Feature Detection ──
    if settings.get("run_feature_detection", False):

        print("[Pipeline] Running feature detection...")

        if len(images) < 2:
            raise ValueError(
                "Need at least 2 images for feature detection."
            )

        img1 = images[0]
        img2 = images[1]

        # Detect features
        kp1, des1 = detect_features(
            img1,
            method="sift"
        )

        kp2, des2 = detect_features(
            img2,
            method="sift"
        )

        # Match descriptors
        matches = match_features(des1, des2)

        results["features"] = {
            "kp1": kp1,
            "kp2": kp2,
            "des1": des1,
            "des2": des2,
            "matches": matches
        }

        print(f"[Pipeline] Matches found: {len(matches)}")
        print("[Pipeline] Feature detection done.")

    # ── Step 4: Stitching ──
    if settings.get("run_stitching", False):

        print("[Pipeline] Running stitching...")

        if len(images) < 2:
            raise ValueError(
                "At least 2 images are required for stitching."
            )

        panorama = images[0]
        H = None
        mask = None

        for i in range(1, len(images)):

            next_image = images[i]

            # Detect features
            kp1, des1 = detect_features(
                panorama,
                method="sift"
            )

            kp2, des2 = detect_features(
                next_image,
                method="sift"
            )

            # Match descriptors
            matches = match_features(des1, des2)

            print(
                f"[Pipeline] Stitching image {i} "
                f"with {len(matches)} matches."
            )

            # Stitch current panorama with next image
            panorama, H, mask = stitch_image_pair(
                panorama,
                next_image,
                kp1,
                kp2,
                matches
            )

        results["stitching"] = {
            "panorama": panorama,
            "homography": H,
            "mask": mask
        }

        print("[Pipeline] Stitching done.")

    # ── Step 5: Segmentation ──
    if settings.get("run_segmentation", False):

        print("[Pipeline] Running segmentation...")

        if "stitching" not in results:
            raise RuntimeError(
                "Stitching must run before segmentation."
            )

        panorama = results["stitching"]["panorama"]

        segmentation_result = segment(
            panorama,
            method="kmeans"
        )

        results["segmentation"] = segmentation_result

        print("[Pipeline] Segmentation done.")

    # ── Step 6: Classification (not implemented yet) ──
    if settings.get("run_classification", False):
        print("[Pipeline] Classification not implemented yet.")
        pass

    return results
