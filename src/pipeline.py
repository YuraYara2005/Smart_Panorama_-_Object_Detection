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
from src.feature_detection.harris import detect_harris_corners, draw_harris_corners
from src.matching.matcher import match_features, draw_matches
from src.stitching.stitcher import stitch_image_pair
from src.segmentation.segmenter import segment
from src.segmentation.metrics import iou


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


def run_feature_detection(images: List[np.ndarray], settings: Dict) -> Dict:
    if len(images) < 2:
        raise ValueError("Need at least 2 images for feature detection and matching.")

    img1, img2 = images[0], images[1]

    # ── SIFT ──
    kp1, des1 = detect_sift_features(img1)
    kp2, des2 = detect_sift_features(img2)
    matches    = match_features(des1, des2)
    kp_img1    = draw_keypoints(img1, kp1)
    kp_img2    = draw_keypoints(img2, kp2)
    match_img  = draw_matches(img1, kp1, img2, kp2, matches)

    # ── Harris corners (both images, configurable threshold) ──
    threshold_ratio = settings.get("harris_threshold", 0.01)

    harris_kp1, _ = detect_harris_corners(img1, threshold_ratio=threshold_ratio)
    harris_kp2, _ = detect_harris_corners(img2, threshold_ratio=threshold_ratio)
    harris_img1   = draw_harris_corners(img1, harris_kp1)
    harris_img2   = draw_harris_corners(img2, harris_kp2)

    return {
        "kp1":          kp1,
        "kp2":          kp2,
        "des1":         des1,
        "des2":         des2,
        "matches":      matches,
        "kp_img1":      kp_img1,
        "kp_img2":      kp_img2,
        "match_img":    match_img,
        # Harris results
        "harris_kp1":   harris_kp1,
        "harris_kp2":   harris_kp2,
        "harris_img1":  harris_img1,
        "harris_img2":  harris_img2,
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

    # Compute inlier ratio for matching accuracy metric
    n_matches = len(feature_results["matches"])
    n_inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = round(n_inliers / max(n_matches, 1), 4)

    return {
        "panorama":     panorama,
        "homography":   H,
        "mask":         mask,
        "n_matches":    n_matches,
        "n_inliers":    n_inliers,
        "inlier_ratio": inlier_ratio,
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

    # IoU between kmeans and watershed as a quantitative cross-method metric
    kmeans_fg    = (results["kmeans_labels"] > 0).astype(np.uint8)
    watershed_fg = (results["watershed_labels"] > 1).astype(np.uint8)
    results["cross_iou"] = round(float(iou(kmeans_fg, watershed_fg)), 4)

    return results


def run_classification(panorama: np.ndarray, settings: Dict) -> Dict:
    """
    Runs inference using a pre-trained model saved by train.py.
    Segments a few crops from the panorama and classifies each one.
    Falls back gracefully if no model files are found.
    """
    import os, joblib

    model_dir = settings.get("classification_model_dir", "data/classification")

    required = [
        os.path.join(model_dir, "random_forest.joblib"),
        os.path.join(model_dir, "scaler.joblib"),
        os.path.join(model_dir, "pca.joblib"),
        os.path.join(model_dir, "label_encoder.joblib"),
    ]

    for path in required:
        if not os.path.exists(path):
            return {
                "error": (
                    f"Model file not found: {path}. "
                    "Run train.py first to generate model files."
                )
            }

    try:
        from src.classification.features import extract_object_features
        from src.segmentation.segmenter import segment as seg_fn

        rf            = joblib.load(required[0])
        scaler        = joblib.load(required[1])
        pca           = joblib.load(required[2])
        label_encoder = joblib.load(required[3])

        # Use kmeans to get a few object crops from the panorama
        seg_out   = seg_fn(panorama, method="kmeans")
        label_map = seg_out["labels"]

        predictions = []
        for label_id in np.unique(label_map)[:6]:   # at most 6 crops
            mask = (label_map == label_id).astype(np.uint8)
            if mask.sum() < 500:
                continue
            ys, xs = np.where(mask > 0)
            x, y   = int(xs.min()), int(ys.min())
            w, h   = int(xs.max()) - x + 1, int(ys.max()) - y + 1
            if w < 24 or h < 24:
                continue
            try:
                feat    = extract_object_features(panorama, mask, (x, y, w, h))
                feat_s  = scaler.transform(feat.reshape(1, -1))
                feat_p  = pca.transform(feat_s)
                pred_id = rf.predict(feat_p)[0]
                proba   = rf.predict_proba(feat_p)[0]
                label   = label_encoder.inverse_transform([pred_id])[0]
                conf    = round(float(proba.max()), 3)

                crop = panorama[y:y + h, x:x + w]
                predictions.append({
                    "label":      label,
                    "confidence": conf,
                    "crop":       crop,
                    "bbox":       (x, y, w, h),
                })
            except Exception:
                continue

        return {"predictions": predictions}

    except Exception as exc:
        return {"error": str(exc)}


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

    # ── Step 3: Feature Detection & Matching (SIFT + Harris) ──
    if settings.get("run_feature_detection", False):
        if len(images) < 2:
            print("[Pipeline] Need at least 2 images for feature detection. Skipping.")
        else:
            print("[Pipeline] Running feature detection and matching...")
            results["feature_detection"] = run_feature_detection(images, settings)
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
        panorama = None
        if "stitching" in results:
            panorama = results["stitching"]["panorama"]
        elif images:
            panorama = images[0]

        if panorama is not None:
            print("[Pipeline] Running classification...")
            results["classification"] = run_classification(panorama, settings)
            print("[Pipeline] Classification done.")

    return results
