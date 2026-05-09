from __future__ import annotations
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple
from src.segmentation.grabcut import refine_mask_grabcut
from src.segmentation.filtering import filter_mask
import cv2
import numpy as np
from tqdm import tqdm
from src.segmentation.metrics import iou
from src.segmentation.segmenter import segment


@dataclass
class ObjectSample:
    image_id: int
    annotation_id: int
    category_id: int
    category_name: str
    image_path: str
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    match_iou: float
    segmentation_method: str


class CocoObjectDatasetBuilder:
    def __init__(
        self,
        dataset_root: str | Path,
        images_subdir: str = "images/val2017",
        annotations_file: str = "annotations/instances_val2017.json",
        segmentation_method: str = "watershed",
        top_k_categories: int = 6,
        min_category_instances: int = 200,
        max_samples_per_category: int = 60,
        min_mask_area: int = 300,
        min_iou: float = 0.10,
        max_mask_fraction: float = 0.65,
        annotation_fallback: bool = False,
        category_names: Optional[Sequence[str]] = None,
        exclude_category_names: Optional[Sequence[str]] = None,
        category_selection_strategy: str = "separable",
        min_category_median_area: float = 0.01,
        min_bbox_area_ratio: float = 0.003,
        min_bbox_side: int = 24,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / images_subdir
        self.annotations_path = self.dataset_root / annotations_file
        self.segmentation_method = segmentation_method
        self.top_k_categories = top_k_categories
        self.min_category_instances = min_category_instances
        self.max_samples_per_category = max_samples_per_category
        self.min_mask_area = min_mask_area
        self.min_iou = min_iou
        self.max_mask_fraction = max_mask_fraction
        self.annotation_fallback = annotation_fallback
        self.category_names = [name.strip() for name in (category_names or []) if name.strip()]
        self.exclude_category_names = {
            name.strip() for name in (exclude_category_names or []) if name.strip()
        }
        self.category_selection_strategy = category_selection_strategy
        self.min_category_median_area = min_category_median_area
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.min_bbox_side = min_bbox_side

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.images = {item["id"]: item for item in data["images"]}
        self.image_areas = {
            item["id"]: float(item["width"] * item["height"])
            for item in data["images"]
        }
        self.categories = {item["id"]: item["name"] for item in data["categories"]}
        self.category_ids_by_name = {
            item["name"]: item["id"]
            for item in data["categories"]
        }
        self.annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
        self.category_annotation_areas: Dict[int, List[float]] = defaultdict(list)
        self.category_counts = Counter()

        for annotation in data["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)
            if annotation.get("iscrowd", 0):
                continue
            bbox = annotation.get("bbox", [0, 0, 0, 0])
            area_ratio = (bbox[2] * bbox[3]) / max(self.image_areas[annotation["image_id"]], 1.0)
            self.category_annotation_areas[annotation["category_id"]].append(area_ratio)
            self.category_counts[annotation["category_id"]] += 1

    def select_categories(self) -> Dict[int, str]:
        if self.category_names:
            selected = []
            missing = []
            for name in self.category_names:
                category_id = self.category_ids_by_name.get(name)
                if category_id is None:
                    missing.append(name)
                    continue
                if self.category_counts[category_id] < self.min_category_instances:
                    continue
                selected.append(category_id)
            if missing:
                raise ValueError(f"Unknown COCO categories: {', '.join(missing)}")
            if not selected:
                raise ValueError(
                    "None of the requested categories met the minimum instance threshold."
                )
            return {category_id: self.categories[category_id] for category_id in selected}

        candidates = self._rank_categories(self.category_selection_strategy)
        if not candidates and self.category_selection_strategy == "separable":
            candidates = self._rank_categories("frequency")

        chosen = [category_id for _, _, category_id in candidates[: self.top_k_categories]]
        if not chosen:
            raise ValueError(
                "No categories met the selection criteria. Lower thresholds or choose category_names explicitly."
            )

        return {category_id: self.categories[category_id] for category_id in chosen}

    # def build(self, max_images: Optional[int] = None) -> Tuple[List[ObjectSample], Dict[int, str]]:
    #     selected_categories = self.select_categories()
    #     category_counts = {category_id: 0 for category_id in selected_categories}
    #     samples: List[ObjectSample] = []
    #     image_ids = sorted(self.images)
    #     if max_images is not None:
    #         image_ids = image_ids[:max_images]
    #
    #     progress = tqdm(image_ids, desc="Building classification dataset", unit="image")
    #     for image_id in progress:
    #         if all(
    #             count >= self.max_samples_per_category
    #             for count in category_counts.values()
    #         ):
    #             break
    #
    #         image_meta = self.images[image_id]
    #         image_path = self.images_dir / image_meta["file_name"]
    #         if not image_path.exists():
    #             continue
    #
    #         image = cv2.imread(str(image_path))
    #         if image is None:
    #             continue
    #         height, width = image.shape[:2]
    #         image_area = float(height * width)
    #
    #         annotations = [
    #             annotation
    #             for annotation in self.annotations_by_image.get(image_id, [])
    #             if annotation["category_id"] in selected_categories
    #             and not annotation.get("iscrowd", 0)
    #             and self._annotation_is_usable(annotation, image_area)
    #         ]
    #         if not annotations:
    #             continue
    #
    #         candidates = self._extract_candidates(image)
    #         matches = self._match_candidates_to_annotations(
    #             candidates=candidates,
    #             annotations=annotations,
    #             height=height,
    #             width=width,
    #         )
    #
    #         matched_annotation_ids = set()
    #         for annotation, candidate_mask, match_score in matches:
    #             category_id = annotation["category_id"]
    #             if category_counts[category_id] >= self.max_samples_per_category:
    #                 continue
    #
    #             x, y, w, h = self._bbox_from_mask(candidate_mask)
    #             samples.append(
    #                 ObjectSample(
    #                     image_id=image_id,
    #                     annotation_id=annotation["id"],
    #                     category_id=category_id,
    #                     category_name=selected_categories[category_id],
    #                     image_path=str(image_path),
    #                     bbox=(x, y, w, h),
    #                     mask=candidate_mask.astype(np.uint8),
    #                     match_iou=float(match_score),
    #                     segmentation_method=self.segmentation_method,
    #                 )
    #             )
    #             matched_annotation_ids.add(annotation["id"])
    #             category_counts[category_id] += 1
    #
    #         if self.annotation_fallback:
    #             for annotation in annotations:
    #                 if annotation["id"] in matched_annotation_ids:
    #                     continue
    #                 category_id = annotation["category_id"]
    #                 if category_counts[category_id] >= self.max_samples_per_category:
    #                     continue
    #                 mask = self._annotation_to_mask(annotation, height, width)
    #                 if mask is None or int(mask.sum()) < self.min_mask_area:
    #                     continue
    #                 x, y, w, h = self._bbox_from_mask(mask)
    #                 samples.append(
    #                     ObjectSample(
    #                         image_id=image_id,
    #                         annotation_id=annotation["id"],
    #                         category_id=category_id,
    #                         category_name=selected_categories[category_id],
    #                         image_path=str(image_path),
    #                         bbox=(x, y, w, h),
    #                         mask=mask.astype(np.uint8),
    #                         match_iou=1.0,
    #                         segmentation_method=f"{self.segmentation_method}_fallback",
    #                     )
    #                 )
    #                 category_counts[category_id] += 1
    #
    #         progress.set_postfix(
    #             {
    #                 "samples": len(samples),
    #                 "categories": sum(1 for count in category_counts.values() if count > 0),
    #             }
    #         )
    #
    #     filtered_samples = [
    #         sample
    #         for sample in samples
    #         if category_counts[sample.category_id] > 1
    #     ]
    #     return filtered_samples, selected_categories
    def build(self, max_images: Optional[int] = None) -> Tuple[List[ObjectSample], Dict[int, str]]:

        selected_categories = self.select_categories()

        category_counts = {
            category_id: 0
            for category_id in selected_categories
        }

        samples: List[ObjectSample] = []

        image_ids = sorted(self.images)

        if max_images is not None:
            image_ids = image_ids[:max_images]

        progress = tqdm(
            image_ids,
            desc="Building classification dataset",
            unit="image"
        )

        for image_id in progress:

            if all(
                    count >= self.max_samples_per_category
                    for count in category_counts.values()
            ):
                break

            image_meta = self.images[image_id]

            image_path = self.images_dir / image_meta["file_name"]

            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            height, width = image.shape[:2]

            image_area = float(height * width)

            annotations = [
                annotation
                for annotation in self.annotations_by_image.get(image_id, [])
                if (
                        annotation["category_id"] in selected_categories
                        and not annotation.get("iscrowd", 0)
                        and self._annotation_is_usable(annotation, image_area)
                )
            ]

            if not annotations:
                continue

            # =============================================
            # USE COCO BOUNDING BOXES DIRECTLY
            # =============================================

            for annotation in annotations:

                category_id = annotation["category_id"]

                if (
                        category_counts[category_id]
                        >= self.max_samples_per_category
                ):
                    continue

                x, y, w, h = map(
                    int,
                    annotation["bbox"]
                )

                if w <= 0 or h <= 0:
                    continue

                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8
                )

                mask[y:y + h, x:x + w] = 1

                samples.append(
                    ObjectSample(
                        image_id=image_id,
                        annotation_id=annotation["id"],
                        category_id=category_id,
                        category_name=selected_categories[category_id],
                        image_path=str(image_path),
                        bbox=(x, y, w, h),
                        mask=mask.astype(np.uint8),
                        match_iou=1.0,
                        segmentation_method="coco_bbox"
                    )
                )

                category_counts[category_id] += 1

            progress.set_postfix({
                "samples": len(samples),
                "categories": sum(
                    1
                    for count in category_counts.values()
                    if count > 0
                ),
            })

        filtered_samples = [
            sample
            for sample in samples
            if category_counts[sample.category_id] > 1
        ]

        return filtered_samples, selected_categories
    def _rank_categories(self, strategy: str) -> List[Tuple[float, int, int]]:
        ranked: List[Tuple[float, int, int]] = []
        for category_id, count in self.category_counts.items():
            if count < self.min_category_instances:
                continue
            category_name = self.categories[category_id]
            if category_name in self.exclude_category_names:
                continue
            median_area = median(self.category_annotation_areas[category_id])
            if strategy == "separable":
                if median_area < self.min_category_median_area:
                    continue
                score = count * median_area
            else:
                score = float(count)
            ranked.append((score, count, category_id))

        ranked.sort(reverse=True)
        return ranked

    def _annotation_is_usable(self, annotation: dict, image_area: float) -> bool:
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        width = float(bbox[2])
        height = float(bbox[3])
        return (
            width >= self.min_bbox_side
            and height >= self.min_bbox_side
            and (width * height) / max(image_area, 1.0) >= self.min_bbox_area_ratio
        )

    def _extract_candidates(self,image: np.ndarray) -> List[np.ndarray]:
        segmentation_output = segment(image,method=self.segmentation_method)
        label_map = segmentation_output["labels"]
        if self.segmentation_method == "watershed":
            raw_labels = [label for label in np.unique(label_map)if label > 1]
        else:
            raw_labels = list(np.unique(label_map))
        image_area = image.shape[0] * image.shape[1]
        kernel = np.ones((3, 3), dtype=np.uint8)
        candidates: List[np.ndarray] = []
        for label in raw_labels:
            cluster_mask = (label_map == label).astype(np.uint8)
            # Remove tiny masks
            if int(cluster_mask.sum()) < self.min_mask_area:
                continue
            # Morphological cleanup
            cluster_mask = cv2.morphologyEx(cluster_mask,cv2.MORPH_OPEN,kernel)
            cluster_mask = cv2.morphologyEx(cluster_mask,cv2.MORPH_CLOSE, kernel)
            component_count, component_labels = cv2.connectedComponents( cluster_mask)
            for component_index in range(1, component_count):
                component_mask = (component_labels == component_index).astype(np.uint8)
                # GrabCut Refinement
                component_mask = refine_mask_grabcut(image,component_mask)
                # Remove bad masks
                if not filter_mask(component_mask):
                    continue
                area = int(component_mask.sum())
                if area < self.min_mask_area:
                    continue
                if area / max(image_area, 1) > self.max_mask_fraction:
                    continue
                if self._touches_border_too_much(component_mask):
                    continue
                candidates.append(component_mask)
        return candidates
    def _match_candidates_to_annotations(
        self,
        candidates: Sequence[np.ndarray],
        annotations: Sequence[dict],
        height: int,
        width: int,
    ) -> List[Tuple[dict, np.ndarray, float]]:
        scored_matches: List[Tuple[float, int, int]] = []
        annotation_masks: List[Optional[np.ndarray]] = []

        for annotation in annotations:
            annotation_masks.append(self._annotation_to_mask(annotation, height, width))

        for candidate_index, candidate_mask in enumerate(candidates):
            for annotation_index, annotation_mask in enumerate(annotation_masks):
                if annotation_mask is None:
                    continue
                score = float(iou(candidate_mask, annotation_mask))
                if score >= self.min_iou:
                    scored_matches.append((score, candidate_index, annotation_index))

        scored_matches.sort(reverse=True, key=lambda item: item[0])
        used_candidates = set()
        used_annotations = set()
        matches: List[Tuple[dict, np.ndarray, float]] = []

        for score, candidate_index, annotation_index in scored_matches:
            if candidate_index in used_candidates or annotation_index in used_annotations:
                continue
            used_candidates.add(candidate_index)
            used_annotations.add(annotation_index)
            matches.append((annotations[annotation_index], candidates[candidate_index], score))

        return matches

    def _annotation_to_mask(
        self,
        annotation: dict,
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        segmentation = annotation.get("segmentation")
        if not segmentation:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
            return mask if int(mask.sum()) > 0 else None

        counts = segmentation.get("counts")
        if isinstance(counts, list):
            return self._decode_uncompressed_rle(segmentation, height, width)

        return None

    @staticmethod
    def _decode_uncompressed_rle(segmentation: dict, height: int, width: int) -> np.ndarray:
        counts = segmentation["counts"]
        values: List[int] = []
        current = 0
        for run_length in counts:
            values.extend([current] * run_length)
            current = 1 - current
        array = np.array(values, dtype=np.uint8)
        if array.size < height * width:
            array = np.pad(array, (0, height * width - array.size))
        return array[: height * width].reshape((width, height)).T

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

    @staticmethod
    def _touches_border_too_much(mask: np.ndarray, border: int = 2) -> bool:
        top = mask[:border, :].sum()
        bottom = mask[-border:, :].sum()
        left = mask[:, :border].sum()
        right = mask[:, -border:].sum()
        border_pixels = int(top + bottom + left + right)
        return border_pixels > 0.25 * max(int(mask.sum()), 1)
