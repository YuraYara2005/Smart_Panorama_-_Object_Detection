import cv2
import numpy as np

def kmeans_segment(image, k=5):

    # Convert image into pixel vectors
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # KMeans stopping criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers back to uint8
    centers = np.uint8(centers)

    # Rebuild image
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    # Label map
    label_map = labels.reshape(image.shape[:2])

    return segmented_image, label_map, compactness