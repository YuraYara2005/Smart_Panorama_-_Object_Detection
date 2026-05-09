import cv2
import numpy as np
import yaml
from typing import Dict, List

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
from src.feature_detection.sift import detect_sift_features, draw_keypoints
from src.matching.matcher import match_features, draw_matches
from src.stitching.stitcher import stitch_image_pair
from src.segmentation.segmenter import segment
from src.classification.predict import PanoramaObjectClassifier


def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_preprocessing(image: np.ndarray, settings: Dict) -> Dict:
    noisy = add_noise(
        image,
        noise_type=settings["noise_type"],
        intensity=settings["noise_intensity"]
    )
    gaussian_result = gaussian_filter_cv(
        noisy,
        kernel_size=settings["gaussian_kernel_size"],
        sigma=settings["gaussian_sigma"]
    )
    median_result = median_filter_cv(
        noisy,
        kernel_size=settings["median_kernel_size"]
    )
    metrics_gaussian = compute_all_metrics(image, gaussian_result, label="Gaussian Filter")
    metrics_median   = compute_all_metrics(image, median_result,   label="Median Filter")

    return {
        "original":         image,
        "noisy":            noisy,
        "gaussian":         gaussian_result,
        "median":           median_result,
        "metrics_gaussian": metrics_gaussian,
        "metrics_median":   metrics_median,
    }


def run_pyramids(image: np.ndarray, settings: Dict) -> Dict:
    gaussian_pyramid  = build_gaussian_pyramid(
        image,
        levels=settings["pyramid_levels"],
        sigma=settings["pyramid_sigma"]
    )
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    reconstructed     = reconstruct_from_laplacian(laplacian_pyramid)

    return {
        "original":          image,
        "gaussian_pyramid":  gaussian_pyramid,
        "laplacian_pyramid": laplacian_pyramid,
        "reconstructed":     reconstructed,
    }


def run_feature_detection(images: List[np.ndarray]) -> Dict:
    if len(images) < 2:
        raise ValueError("Need at least 2 images for feature detection and matching.")

    img1, img2 = images[0], images[1]

    # Detect features
    kp1, des1 = detect_sift_features(img1)
    kp2, des2 = detect_sift_features(img2)

    # Match features
    matches = match_features(des1, des2)

    # Draw keypoints and matches for visualization
    kp_img1    = draw_keypoints(img1, kp1)
    kp_img2    = draw_keypoints(img2, kp2)
    match_img  = draw_matches(img1, kp1, img2, kp2, matches)

    return {
        "kp1":       kp1,
        "kp2":       kp2,
        "des1":      des1,
        "des2":      des2,
        "matches":   matches,
        "kp_img1":   kp_img1,
        "kp_img2":   kp_img2,
        "match_img": match_img,
    }


def run_stitching(images: List[np.ndarray], feature_results: Dict) -> Dict:
    img1, img2 = images[0], images[1]

    panorama, H, mask = stitch_image_pair(
        img1,
        img2,
        feature_results["kp1"],
        feature_results["kp2"],
        feature_results["matches"]
    )

    return {
        "panorama":   panorama,
        "homography": H,
        "mask":       mask,
    }


def run_segmentation(panorama: np.ndarray, settings: Dict) -> Dict:
    results = {}

    # KMeans segmentation
    kmeans_result = segment(panorama, method="kmeans")
    results["kmeans_segmented"] = kmeans_result["segmented_image"]
    results["kmeans_labels"]    = kmeans_result["labels"]

    # Watershed segmentation
    watershed_result = segment(panorama, method="watershed")
    results["watershed_segmented"] = watershed_result["segmented_image"]
    results["watershed_labels"]    = watershed_result["labels"]

    return results

#i added these we can delete them if they miss things up
def extract_masks_from_labels(label_map):

    masks = []

    unique_labels = np.unique(label_map)

    for label in unique_labels:

        if label <= 1:
            continue

        mask = (label_map == label).astype("uint8")

        if mask.sum() < 300:
            continue

        masks.append(mask)

    return masks

def draw_predictions(image, predictions):

    output = image.copy()

    for pred in predictions:

        x, y, w, h = pred["bbox"]

        label = pred["label"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output

def run_pipeline(images: List[np.ndarray], settings: Dict) -> Dict:
    if not images:
        return {}

    results = {}

    # ── Step 1: Preprocessing ──
    if settings.get("run_preprocessing", False):
        print("[Pipeline] Running preprocessing...")
        results["preprocessing"] = run_preprocessing(images[0], settings)
        print("[Pipeline] Preprocessing done.")

    # ── Step 2: Pyramids ──
    if settings.get("run_pyramids", False):
        print("[Pipeline] Running pyramids...")
        results["pyramids"] = run_pyramids(images[0], settings)
        print("[Pipeline] Pyramids done.")

    # ── Step 3: Feature Detection & Matching ──
    if settings.get("run_feature_detection", False):
        if len(images) < 2:
            print("[Pipeline] Need at least 2 images for feature detection. Skipping.")
        else:
            print("[Pipeline] Running feature detection and matching...")
            results["feature_detection"] = run_feature_detection(images)
            print("[Pipeline] Feature detection done.")

    # ── Step 4: Stitching ──
    if settings.get("run_stitching", False):
        if "feature_detection" not in results:
            print("[Pipeline] Stitching requires feature detection. Skipping.")
        else:
            print("[Pipeline] Running stitching...")
            results["stitching"] = run_stitching(images, results["feature_detection"])
            print("[Pipeline] Stitching done.")

    # ── Step 5: Segmentation ──
    if settings.get("run_segmentation", False):
        if "stitching" not in results:
            print("[Pipeline] Segmentation requires stitching. Skipping.")
        else:
            print("[Pipeline] Running segmentation...")
            results["segmentation"] = run_segmentation(
                results["stitching"]["panorama"], settings
            )
            print("[Pipeline] Segmentation done.")

    # ── Step 6: Classification ──
    if settings.get("run_classification", False):

        if "segmentation" not in results:

            print("[Pipeline] Classification requires segmentation.")

        else:

            print("[Pipeline] Running classification...")

            label_map = results["segmentation"]["watershed_labels"]

            panorama = results["stitching"]["panorama"]

            masks = extract_masks_from_labels(label_map)

            classifier = PanoramaObjectClassifier(
                model_dir="data/classification"
            )

            predictions = classifier.predict_objects(
                panorama,
                masks
            )

            classified_image = draw_predictions(
                panorama,
                predictions
            )

            results["classified_panorama"] = classified_image

            results["classification"] = predictions

            print(
                f"[Pipeline] Classified {len(predictions)} objects."
            )

    return results
