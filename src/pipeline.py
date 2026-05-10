import cv2
import numpy as np
import yaml
from typing import Dict, List

from src.preprocessing.filters import (
    gaussian_filter_cv,
    median_filter_cv,
    add_noise
)

from src.preprocessing.evaluation import (
    compute_all_metrics
)

from src.pyramids.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian
)

from src.feature_detection.sift import (
    detect_sift_features,
    draw_keypoints
)

from src.feature_detection.harris import (
    detect_harris_corners,
    draw_harris_corners
)

from src.matching.matcher import (
    match_features,
    draw_matches
)

from src.stitching.stitcher import (
    stitch_image_pair
)

from src.segmentation.segmenter import (
    segment
)

from src.segmentation.metrics import (
    iou
)


# =====================================================
# CONFIG
# =====================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =====================================================
# PREPROCESSING
# =====================================================

def run_preprocessing(image: np.ndarray, settings: Dict) -> Dict:

    noisy = add_noise(
        image,
        noise_type=settings.get(
            "noise_type",
            "gaussian"
        ),
        intensity=settings.get(
            "noise_intensity",
            25
        )
    )

    gaussian_result = gaussian_filter_cv(
        noisy,
        kernel_size=settings.get(
            "gaussian_kernel_size",
            5
        ),
        sigma=settings.get(
            "gaussian_sigma",
            1.0
        )
    )

    median_result = median_filter_cv(
        noisy,
        kernel_size=settings.get(
            "median_kernel_size",
            5
        )
    )

    metrics_gaussian = compute_all_metrics(
        image,
        gaussian_result,
        label="Gaussian Filter"
    )

    metrics_median = compute_all_metrics(
        image,
        median_result,
        label="Median Filter"
    )

    return {
        "original": image,
        "noisy": noisy,
        "gaussian": gaussian_result,
        "median": median_result,
        "metrics_gaussian": metrics_gaussian,
        "metrics_median": metrics_median,
    }


# =====================================================
# PYRAMIDS
# =====================================================

def run_pyramids(image: np.ndarray, settings: Dict) -> Dict:

    levels = settings.get(
        "pyramid_levels",
        4
    )

    sigma = settings.get(
        "pyramid_sigma",
        1.0
    )

    gaussian_pyramid = build_gaussian_pyramid(
        image,
        levels=levels,
        sigma=sigma
    )

    laplacian_pyramid = build_laplacian_pyramid(
        gaussian_pyramid
    )

    reconstructed = reconstruct_from_laplacian(
        laplacian_pyramid
    )

    return {
        "original": image,
        "gaussian_pyramid": gaussian_pyramid,
        "laplacian_pyramid": laplacian_pyramid,
        "reconstructed": reconstructed,
    }


# =====================================================
# FEATURE DETECTION
# =====================================================

def run_feature_detection(
    images: List[np.ndarray],
    settings: Dict
) -> Dict:

    if len(images) < 2:
        raise ValueError(
            "Need at least 2 images."
        )

    img1, img2 = images[0], images[1]

    # ── SIFT ──
    kp1, des1 = detect_sift_features(
        img1
    )

    kp2, des2 = detect_sift_features(
        img2
    )

    matches = match_features(
        des1,
        des2
    )

    kp_img1 = draw_keypoints(
        img1,
        kp1
    )

    kp_img2 = draw_keypoints(
        img2,
        kp2
    )

    match_img = draw_matches(
        img1,
        kp1,
        img2,
        kp2,
        matches
    )

    # ── Harris ──
    threshold_ratio = settings.get(
        "harris_threshold",
        0.01
    )

    harris_kp1, _ = detect_harris_corners(
        img1,
        threshold_ratio=threshold_ratio
    )

    harris_kp2, _ = detect_harris_corners(
        img2,
        threshold_ratio=threshold_ratio
    )

    harris_img1 = draw_harris_corners(
        img1,
        harris_kp1
    )

    harris_img2 = draw_harris_corners(
        img2,
        harris_kp2
    )

    return {
        "kp1": kp1,
        "kp2": kp2,
        "des1": des1,
        "des2": des2,
        "matches": matches,
        "kp_img1": kp_img1,
        "kp_img2": kp_img2,
        "match_img": match_img,
        "harris_kp1": harris_kp1,
        "harris_kp2": harris_kp2,
        "harris_img1": harris_img1,
        "harris_img2": harris_img2,
    }


# =====================================================
# STITCHING
# =====================================================

def run_stitching(
    images: List[np.ndarray],
    feature_results: Dict
) -> Dict:

    img1, img2 = images[0], images[1]

    panorama, H, mask = stitch_image_pair(
        img1,
        img2,
        feature_results["kp1"],
        feature_results["kp2"],
        feature_results["matches"]
    )

    n_matches = len(
        feature_results["matches"]
    )

    n_inliers = int(
        mask.sum()
    ) if mask is not None else 0

    inlier_ratio = round(
        n_inliers / max(n_matches, 1),
        4
    )

    return {
        "panorama": panorama,
        "homography": H,
        "mask": mask,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
    }


# =====================================================
# SEGMENTATION
# =====================================================

def run_segmentation(
    panorama: np.ndarray,
    settings: Dict
) -> Dict:

    results = {}

    max_width = settings.get(
        "segmentation_max_width",
        1000
    )

    h, w = panorama.shape[:2]

    working_image = panorama.copy()

    if w > max_width:

        scale = max_width / float(w)

        working_image = cv2.resize(
            panorama,
            (
                max_width,
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    # KMeans
    kmeans_result = segment(
        working_image,
        method="kmeans"
    )

    # Watershed
    watershed_result = segment(
        working_image,
        method="watershed"
    )

    results["kmeans_segmented"] = kmeans_result[
        "segmented_image"
    ]

    results["kmeans_labels"] = kmeans_result[
        "labels"
    ]

    results["watershed_segmented"] = watershed_result[
        "segmented_image"
    ]

    results["watershed_labels"] = watershed_result[
        "labels"
    ]

    # IoU
    kmeans_fg = (
        results["kmeans_labels"] > 0
    ).astype(np.uint8)

    watershed_fg = (
        results["watershed_labels"] > 1
    ).astype(np.uint8)

    results["cross_iou"] = round(
        float(
            iou(
                kmeans_fg,
                watershed_fg
            )
        ),
        4
    )

    return results


# =====================================================
# CLASSIFICATION (YOLO)
# =====================================================

def run_classification(
    panorama: np.ndarray,
    settings: Dict
) -> Dict:

    try:
        from ultralytics import YOLO

    except ImportError:
        return {
            "error": (
                "Ultralytics not installed. "
                "Run: pip install ultralytics"
            )
        }

    print(
        "[Pipeline] YOLO classification backend ACTIVE"
    )

    # Smaller + safer for Colab/local
    model_name = settings.get(
        "yolo_model",
        "yolov8n.pt"
    )

    confidence_threshold = settings.get(
        "classification_confidence",
        0.05
    )

    original_h, original_w = panorama.shape[:2]

    max_detect_width = settings.get(
        "classification_max_width",
        1920
    )

    detection_image = panorama.copy()

    scale_x = 1.0
    scale_y = 1.0

    # Resize large panoramas
    if original_w > max_detect_width:

        scale = max_detect_width / float(
            original_w
        )

        resized_w = max_detect_width
        resized_h = int(
            original_h * scale
        )

        detection_image = cv2.resize(
            panorama,
            (
                resized_w,
                resized_h
            ),
            interpolation=cv2.INTER_AREA
        )

        scale_x = original_w / float(
            resized_w
        )

        scale_y = original_h / float(
            resized_h
        )

    # Optional debug save
    cv2.imwrite(
        "debug_panorama.jpg",
        detection_image
    )

    try:
        model = YOLO(
            model_name
        )

    except Exception as exc:
        return {
            "error": (
                f"Failed to load YOLO model: {str(exc)}"
            )
        }

    try:
        yolo_results = model(
            detection_image,
            conf=confidence_threshold,
            verbose=False
        )

    except Exception as exc:
        return {
            "error": (
                f"YOLO inference failed: {str(exc)}"
            )
        }

    predictions = []

    for result in yolo_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            class_name = model.names[
                cls_id
            ]

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Scale back
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)

            w = x2 - x1
            h = y2 - y1

            if w < 2 or h < 2:
                continue

            crop = panorama[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            predictions.append({
                "label": class_name,
                "confidence": confidence,
                "crop": crop,
                "bbox": (
                    x1,
                    y1,
                    w,
                    h
                ),
            })

    predictions.sort(
        key=lambda p: p["confidence"],
        reverse=True
    )

    print(
        f"[Pipeline] YOLO detections: {len(predictions)}"
    )

    return {
        "predictions": predictions
    }


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_pipeline(
    images: List[np.ndarray],
    settings: Dict
) -> Dict:

    if not images:
        return {}

    results = {}

    # ── Step 1 ──
    if settings.get(
        "run_preprocessing",
        False
    ):

        print(
            "[Pipeline] Running preprocessing..."
        )

        results["preprocessing"] = run_preprocessing(
            images[0],
            settings
        )

        print(
            "[Pipeline] Preprocessing done."
        )

    # ── Step 2 ──
    if settings.get(
        "run_pyramids",
        False
    ):

        print(
            "[Pipeline] Running pyramids..."
        )

        results["pyramids"] = run_pyramids(
            images[0],
            settings
        )

        print(
            "[Pipeline] Pyramids done."
        )

    # ── Step 3 ──
    if settings.get(
        "run_feature_detection",
        False
    ):

        if len(images) < 2:

            print(
                "[Pipeline] Need at least 2 images for feature detection."
            )

        else:

            print(
                "[Pipeline] Running feature detection and matching..."
            )

            results["feature_detection"] = run_feature_detection(
                images,
                settings
            )

            print(
                "[Pipeline] Feature detection done."
            )

    # ── Step 4 ──
    if settings.get(
        "run_stitching",
        False
    ):

        if "feature_detection" not in results:

            print(
                "[Pipeline] Stitching requires feature detection."
            )

        else:

            print(
                "[Pipeline] Running stitching..."
            )

            results["stitching"] = run_stitching(
                images,
                results["feature_detection"]
            )

            print(
                "[Pipeline] Stitching done."
            )

    # ── Step 5 ──
    if settings.get(
        "run_segmentation",
        False
    ):

        if "stitching" not in results:

            print(
                "[Pipeline] Segmentation requires stitching."
            )

        else:

            print(
                "[Pipeline] Running segmentation..."
            )

            results["segmentation"] = run_segmentation(
                results["stitching"]["panorama"],
                settings
            )

            print(
                "[Pipeline] Segmentation done."
            )

    # ── Step 6 ──
    if settings.get(
        "run_classification",
        False
    ):

        panorama = None

        if "stitching" in results:

            panorama = results["stitching"][
                "panorama"
            ]

        elif images:

            panorama = images[0]

        if panorama is not None:

            print(
                "[Pipeline] Running classification..."
            )

            results["classification"] = run_classification(
                panorama,
                settings
            )

            print(
                "[Pipeline] Classification done."
            )

    return results