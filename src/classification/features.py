from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def get_feature_names() -> List[str]:
    names = [
        "mask_area_ratio",
        "bbox_area_ratio",
        "aspect_ratio",
        "extent",
        "solidity",
        "perimeter_ratio",
        "equivalent_diameter_ratio",
        "circularity",
        "axis_ratio",
        "edge_density",
    ]
    names.extend([f"hu_moment_{index}" for index in range(1, 8)])
    names.extend(["mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v"])
    names.extend([f"h_hist_{index}" for index in range(8)])
    names.extend([f"s_hist_{index}" for index in range(4)])
    names.extend([f"v_hist_{index}" for index in range(4)])
    names.extend(["gray_mean", "gray_std"])
    names.extend([f"lbp_bin_{index}" for index in range(10)])
    names.extend([f"hog_{index}" for index in range(144)])
    return names


def extract_object_features(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    x, y, w, h = bbox
    mask = mask.astype(np.uint8)
    crop = image[y : y + h, x : x + w]
    crop_mask = mask[y : y + h, x : x + w]
    if crop.size == 0 or int(crop_mask.sum()) == 0:
        raise ValueError("Cannot extract features from an empty object mask.")

    features: List[float] = []
    features.extend(_shape_features(mask, crop_mask, w, h))
    features.extend(_hu_moments(crop_mask))
    features.extend(_color_features(crop, crop_mask))
    features.extend(_texture_features(crop, crop_mask))
    features.extend(_hog_features(crop, crop_mask))
    return np.asarray(features, dtype=np.float32)


def _shape_features(
    full_mask: np.ndarray,
    crop_mask: np.ndarray,
    width: int,
    height: int,
) -> List[float]:
    total_area = float(full_mask.shape[0] * full_mask.shape[1])
    object_area = float(crop_mask.sum())
    bbox_area = float(max(width * height, 1))
    aspect_ratio = float(width) / max(float(height), 1.0)
    extent = object_area / bbox_area

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    hull_area = object_area
    axis_ratio = 1.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest, True))
        hull = cv2.convexHull(largest)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        points = largest.reshape(-1, 2).astype(np.float32)
        if len(points) >= 5:
            covariance = np.cov(points.T)
            eigenvalues, _ = np.linalg.eigh(covariance)
            eigenvalues = np.sort(np.maximum(eigenvalues, 1e-8))[::-1]
            axis_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))

    solidity = object_area / hull_area
    equivalent_diameter = np.sqrt((4.0 * object_area) / np.pi)
    diagonal = np.sqrt(float(full_mask.shape[0] ** 2 + full_mask.shape[1] ** 2))
    circularity = (4.0 * np.pi * object_area) / max(perimeter * perimeter, 1.0)

    edges = cv2.Canny(crop_mask * 255, 50, 150)
    edge_density = float((edges > 0).sum()) / max(object_area, 1.0)

    return [
        object_area / max(total_area, 1.0),
        bbox_area / max(total_area, 1.0),
        aspect_ratio,
        extent,
        solidity,
        perimeter / max(diagonal, 1.0),
        equivalent_diameter / max(diagonal, 1.0),
        circularity,
        axis_ratio,
        edge_density,
    ]


def _hu_moments(mask: np.ndarray) -> List[float]:
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    transformed = [-np.sign(value) * np.log10(abs(value) + 1e-12) for value in hu]
    return [float(value) for value in transformed]


def _color_features(crop: np.ndarray, crop_mask: np.ndarray) -> List[float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv[crop_mask > 0]
    if pixels.size == 0:
        return [0.0] * 22

    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)

    histograms: List[float] = []
    bins_and_ranges = [
        (8, [0, 180]),
        (4, [0, 256]),
        (4, [0, 256]),
    ]
    for channel_index, (bin_count, value_range) in enumerate(bins_and_ranges):
        hist = cv2.calcHist([hsv], [channel_index], crop_mask, [bin_count], value_range)
        hist = hist.flatten().astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        histograms.extend(hist.tolist())

    return [
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(stds[0]),
        float(stds[1]),
        float(stds[2]),
        *histograms,
    ]


def _texture_features(crop: np.ndarray, crop_mask: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    masked_gray = gray[crop_mask > 0]
    if masked_gray.size == 0:
        return [0.0] * 12

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_values = lbp[crop_mask > 0]
    histogram, _ = np.histogram(lbp_values, bins=np.arange(0, 11), density=False)
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)

    return [
        float(masked_gray.mean()),
        float(masked_gray.std()),
        *histogram.tolist(),
    ]


def _hog_features(crop: np.ndarray, crop_mask: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray[crop_mask == 0] = 0.0
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32).tolist()
