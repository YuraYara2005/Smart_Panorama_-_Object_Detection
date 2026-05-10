def run_classification(panorama: np.ndarray, settings: Dict) -> Dict:
    """
    Classification backend using YOLO for robust panorama object recognition.
    Keeps output schema compatible with existing Streamlit display.
    """

    try:
        from ultralytics import YOLO

    except ImportError:
        return {
            "error": (
                "Ultralytics YOLO not installed. "
                "Run: pip install ultralytics"
            )
        }

    # ==========================================
    # SETTINGS
    # ==========================================

    model_name = settings.get(
        "yolo_model",
        "yolov8m.pt"
    )

    confidence_threshold = settings.get(
        "classification_confidence",
        0.20
    )

    # Allowed categories for project
    allowed_classes = {
        "car",
        "truck",
        "bus",
        "person",
        "bicycle",
        "motorcycle",
        "airplane",
        "bottle"
    }

    # ==========================================
    # LOAD MODEL
    # ==========================================

    try:
        model = YOLO(model_name)

    except Exception as exc:
        return {
            "error": f"Failed to load YOLO model: {str(exc)}"
        }

    # ==========================================
    # INFERENCE
    # ==========================================

    try:
        results = model(
            panorama,
            conf=confidence_threshold,
            verbose=False
        )

    except Exception as exc:
        return {
            "error": f"Inference failed: {str(exc)}"
        }

    predictions = []

    # ==========================================
    # PARSE DETECTIONS
    # ==========================================

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[
                cls_id
            ]

            if class_name not in allowed_classes:
                continue

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Safety bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(panorama.shape[1], x2)
            y2 = min(panorama.shape[0], y2)

            w = x2 - x1
            h = y2 - y1

            if w < 10 or h < 10:
                continue

            crop = panorama[
                y1:y2,
                x1:x2
            ]

            predictions.append({
                "label": class_name,
                "confidence": round(confidence, 3),
                "crop": crop,
                "bbox": (x1, y1, w, h),
            })

    # ==========================================
    # SORT BEST FIRST
    # ==========================================

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return {
        "predictions": predictions
    }