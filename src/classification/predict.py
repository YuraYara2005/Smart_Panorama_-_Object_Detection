from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np


if __package__ in (None, ""):

    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.segmentation.segmenter import segment

    from src.segmentation.grabcut import (
        refine_mask_grabcut
    )

    from src.segmentation.filtering import (
        filter_mask
    )

    from src.classification.cnn_features import (
        extract_cnn_features
    )

else:

    from ..segmentation.segmenter import segment

    from ..segmentation.grabcut import (
        refine_mask_grabcut
    )

    from ..segmentation.filtering import (
        filter_mask
    )

    from .cnn_features import (
        extract_cnn_features
    )


# =====================================================
# ARGUMENTS
# =====================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="Panorama object inference"
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--model",
        default="data/classification/adaboost.joblib"
    )

    parser.add_argument(
        "--label-encoder",
        default="data/classification/label_encoder.joblib"
    )

    parser.add_argument(
        "--scaler",
        default="data/classification/scaler.joblib"
    )

    parser.add_argument(
        "--pca",
        default="data/classification/pca.joblib"
    )

    parser.add_argument(
        "--segmentation-method",
        choices=["watershed", "kmeans"],
        default="watershed"
    )

    parser.add_argument(
        "--output",
        default="prediction_result.jpg"
    )

    return parser.parse_args()


# =====================================================
# HELPERS
# =====================================================


def bbox_from_mask(mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    return (
        x_min,
        y_min,
        x_max - x_min + 1,
        y_max - y_min + 1
    )


# =====================================================
# MAIN
# =====================================================


def main():

    args = parse_args()

    image = cv2.imread(args.image)

    if image is None:

        print("ERROR: could not load image")

        return

    # =================================================
    # LOAD TRAINED COMPONENTS
    # =================================================

    model = joblib.load(args.model)

    label_encoder = joblib.load(
        args.label_encoder
    )

    scaler = joblib.load(args.scaler)

    pca = joblib.load(args.pca)

    # =================================================
    # SEGMENTATION
    # =================================================

    segmentation_output = segment(
        image,
        method=args.segmentation_method
    )

    label_map = segmentation_output["labels"]

    unique_labels = np.unique(label_map)

    output = image.copy()

    kernel = np.ones((3, 3), np.uint8)

    predictions_count = 0

    # =================================================
    # CANDIDATE EXTRACTION
    # =================================================

    for label in unique_labels:

        if args.segmentation_method == "watershed":

            if label <= 1:
                continue

        cluster_mask = (
            label_map == label
        ).astype(np.uint8)

        if int(cluster_mask.sum()) < 500:
            continue

        cluster_mask = cv2.morphologyEx(
            cluster_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        cluster_mask = cv2.morphologyEx(
            cluster_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        component_count, component_labels = (
            cv2.connectedComponents(
                cluster_mask
            )
        )

        for component_index in range(
            1,
            component_count
        ):

            component_mask = (
                component_labels == component_index
            ).astype(np.uint8)

            component_mask = refine_mask_grabcut(
                image,
                component_mask
            )

            if not filter_mask(component_mask):
                continue

            bbox = bbox_from_mask(component_mask)

            if bbox is None:
                continue

            x, y, w, h = bbox

            # Skip tiny objects
            if w < 30 or h < 30:
                continue

            # =================================================
            # CNN FEATURES
            # =================================================

            feature_vector = extract_cnn_features(
                image,
                bbox
            )

            if feature_vector is None:
                continue

            feature_vector = scaler.transform(
                [feature_vector]
            )

            feature_vector = pca.transform(
                feature_vector
            )

            # =================================================
            # CLASSIFICATION
            # =================================================

            prediction = model.predict(
                feature_vector
            )[0]

            class_name = (
                label_encoder.inverse_transform(
                    [prediction]
                )[0]
            )

            predictions_count += 1

            # =================================================
            # VISUALIZATION
            # =================================================

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                class_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =================================================
    # SAVE OUTPUT
    # =================================================

    cv2.imwrite(args.output, output)

    print(
        f"Saved prediction result to: {args.output}"
    )

    print(
        f"Detected objects: {predictions_count}"
    )
# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":

    main()
