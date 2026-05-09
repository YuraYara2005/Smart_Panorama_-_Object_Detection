from __future__ import annotations

import joblib
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.classification.features import extract_object_features


class PanoramaObjectClassifier:

    def __init__(self, model_dir: str):

        model_dir = Path(model_dir)

        self.model = joblib.load(model_dir / "random_forest.joblib")

        self.scaler = joblib.load(model_dir / "scaler.joblib")

        self.pca = joblib.load(model_dir / "pca.joblib")

        self.label_encoder = joblib.load(
            model_dir / "label_encoder.joblib"
        )

    def predict_objects(
        self,
        image: np.ndarray,
        masks: List[np.ndarray]
    ) -> List[Dict]:

        predictions = []

        for mask in masks:

            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min = int(xs.min())
            x_max = int(xs.max())

            y_min = int(ys.min())
            y_max = int(ys.max())

            bbox = (
                x_min,
                y_min,
                x_max - x_min + 1,
                y_max - y_min + 1
            )

            try:

                feature_vector = extract_object_features(
                    image=image,
                    mask=mask,
                    bbox=bbox
                )

            except Exception:
                continue

            feature_vector = feature_vector.reshape(1, -1)

            feature_vector = self.scaler.transform(feature_vector)

            feature_vector = self.pca.transform(feature_vector)

            prediction = self.model.predict(feature_vector)[0]

            label = self.label_encoder.inverse_transform(
                [prediction]
            )[0]

            predictions.append({
                "label": label,
                "bbox": bbox,
                "mask": mask
            })


        return predictions