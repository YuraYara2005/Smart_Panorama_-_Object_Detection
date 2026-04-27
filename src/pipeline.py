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

    # ── Step 3: Feature Detection (not implemented yet) ──
    if settings.get("run_feature_detection", False):
        print("[Pipeline] Feature detection not implemented yet.")
        pass

    # ── Step 4: Stitching (not implemented yet) ──
    if settings.get("run_stitching", False):
        print("[Pipeline] Stitching not implemented yet.")
        pass

    # ── Step 5: Segmentation (not implemented yet) ──
    if settings.get("run_segmentation", False):
        print("[Pipeline] Segmentation not implemented yet.")
        pass

    # ── Step 6: Classification (not implemented yet) ──
    if settings.get("run_classification", False):
        print("[Pipeline] Classification not implemented yet.")
        pass

    return results
