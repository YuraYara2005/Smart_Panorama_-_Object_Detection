from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):

    project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classification.dataset import CocoObjectDatasetBuilder, ObjectSample
    from src.classification.features import extract_object_features

else:

    from .dataset import CocoObjectDatasetBuilder, ObjectSample
    from .features import extract_object_features


# =====================================================
# ARGUMENTS
# =====================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train object classifiers"
    )

    parser.add_argument("--dataset-root", required=True)

    parser.add_argument(
        "--images-subdir",
        default="images/val2017"
    )

    parser.add_argument(
        "--annotations-file",
        default="annotations/instances_val2017.json"
    )

    parser.add_argument(
        "--segmentation-method",
        choices=["watershed", "kmeans"],
        default="watershed"
    )

    parser.add_argument(
        "--category-names",
        default=""
    )

    parser.add_argument(
        "--top-k-categories",
        type=int,
        default=6
    )

    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=300
    )

    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output-dir",
        default="data/classification"
    )

    return parser.parse_args()


# =====================================================
# HELPERS
# =====================================================

def parse_csv_argument(value: str):

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def build_feature_matrix(
    samples: List[ObjectSample]
) -> Tuple[np.ndarray, np.ndarray]:

    image_cache: Dict[str, np.ndarray] = {}

    features = []

    labels = []

    for sample in samples:

        image = image_cache.get(sample.image_path)

        if image is None:

            image = cv2.imread(sample.image_path)

            if image is None:
                continue

            image_cache[sample.image_path] = image

        feature_vector = extract_object_features(
            image=image,
            mask=sample.mask,
            bbox=sample.bbox
        )

        features.append(feature_vector)

        labels.append(sample.category_name)

    return np.vstack(features), np.asarray(labels)


def evaluate_model(
    model_name,
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    label_encoder
):

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "estimator": model
    }


def save_manifest(samples, output_dir):

    manifest_path = output_dir / "dataset_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "image_id",
            "annotation_id",
            "category_name",
            "image_path",
            "bbox",
            "match_iou"
        ])

        for sample in samples:

            writer.writerow([
                sample.image_id,
                sample.annotation_id,
                sample.category_name,
                sample.image_path,
                sample.bbox,
                sample.match_iou
            ])


# =====================================================
# MAIN
# =====================================================

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():

        output_dir = (
            Path(__file__).resolve().parents[2]
            / output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =================================================
    # DATASET
    # =================================================

    builder = CocoObjectDatasetBuilder(
        dataset_root=args.dataset_root,
        images_subdir=args.images_subdir,
        annotations_file=args.annotations_file,
        segmentation_method=args.segmentation_method,
        category_names=parse_csv_argument(args.category_names),
        top_k_categories=args.top_k_categories,
        max_samples_per_category=args.max_samples_per_category,
        min_iou=args.min_iou
    )

    samples, selected_categories = builder.build()

    print(f"\nCurated dataset size: {len(samples)}")

    print(
        "Selected categories:",
        ", ".join(selected_categories.values())
    )

    # =================================================
    # FEATURES
    # =================================================

    X, labels = build_feature_matrix(samples)

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(labels)

    # =================================================
    # SPLIT
    # =================================================

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # =================================================
    # SCALING
    # =================================================

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    # =================================================
    # PCA
    # =================================================

    pca = PCA(n_components=0.95)

    x_train = pca.fit_transform(x_train)

    x_test = pca.transform(x_test)

    print(f"\nPCA components: {x_train.shape[1]}")

    # =================================================
    # MODELS
    # =================================================

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1
    )

    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=3
        ),
        n_estimators=250,
        learning_rate=0.3,
        random_state=args.random_state
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        random_state=args.random_state
    )

    # =================================================
    # EVALUATION
    # =================================================

    results = [

        evaluate_model(
            model_name="random_forest",
            model=random_forest,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder
        ),

        evaluate_model(
            model_name="adaboost",
            model=adaboost,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder
        ),

        evaluate_model(
            model_name="gradient_boosting",
            model=gradient_boosting,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder
        )
    ]

    # =================================================
    # CROSS VALIDATION
    # =================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.random_state
    )

    rf_cv_scores = cross_val_score(
        random_forest,
        x_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    print(
        f"\nRandom Forest CV F1: "
        f"{rf_cv_scores.mean():.4f}"
    )

    # =================================================
    # BEST MODEL
    # =================================================

    best_result = max(
        results,
        key=lambda item: item["f1"]
    )

    print(
        f"\nBest Model: {best_result['model_name']}"
    )

    # =================================================
    # SAVE METRICS
    # =================================================

    metrics_payload = {
        "dataset_size": len(samples),
        "selected_categories": selected_categories,
        "best_model": best_result["model_name"],
        "results": [
            {
                key: value
                for key, value in result.items()
                if key != "estimator"
            }
            for result in results
        ]
    }

    save_manifest(
        samples,
        output_dir
    )

    metrics_path = output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8"
    )

    # =================================================
    # SAVE MODELS
    # =================================================

    for result in results:

        joblib.dump(
            result["estimator"],
            output_dir /
            f"{result['model_name']}.joblib"
        )

    joblib.dump(
        scaler,
        output_dir / "scaler.joblib"
    )

    joblib.dump(
        pca,
        output_dir / "pca.joblib"
    )

    joblib.dump(
        label_encoder,
        output_dir / "label_encoder.joblib"
    )

    # =================================================
    # SUMMARY
    # =================================================

    print("\n==============================")

    for result in results:

        print(f"\n{result['model_name']}")

        print(
            f"Accuracy : {result['accuracy']:.4f}"
        )

        print(
            f"Precision: {result['precision']:.4f}"
        )

        print(
            f"Recall   : {result['recall']:.4f}"
        )

        print(
            f"F1 Score : {result['f1']:.4f}"
        )
    print("\nSaved everything to:")
    print(output_dir)
# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":

    main()