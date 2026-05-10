from __future__ import annotations

import argparse
import cv2

from ultralytics import YOLO


ALLOWED_CLASSES = {
    "car",
    "bus",
    "truck",
    "bicycle",
    "bottle",
    "airplane"
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--output",
        default="prediction_dnn_result.jpg"
    )

    parser.add_argument(
        "--model",
        default="yolov8x.pt"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.20
    )

    return parser.parse_args()


def main():

    args = parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:
        print("ERROR: Could not load image.")
        return

    model = YOLO(
        args.model
    )

    results = model(
        image,
        conf=args.confidence
    )

    output = image.copy()

    detection_count = 0

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[
                cls_id
            ]

            if class_name not in ALLOWED_CLASSES:
                continue

            conf = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            label = (
                f"{class_name} "
                f"{conf:.2f}"
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detection_count += 1

    cv2.imwrite(
        args.output,
        output
    )

    print(
        f"Saved to: {args.output}"
    )

    print(
        f"Detected objects: {detection_count}"
    )


if __name__ == "__main__":
    main()