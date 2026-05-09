import cv2
import numpy as np

def kmeans_segment(image, k=4):

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2
    )

    compactness, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    segmented_data = centers[
        labels.flatten()
    ]

    segmented_image = segmented_data.reshape(
        image.shape
    )

    label_map = labels.reshape(
        image.shape[:2]
    )

    return (
        segmented_image,
        label_map,
        compactness
    )