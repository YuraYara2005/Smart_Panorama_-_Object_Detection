import cv2
import numpy as np


def watershed_segment(image):
    """
    Perform Watershed segmentation and generate:
    - segmented visualization
    - marker labels
    - binary masks for each region
    """

    if image is None:
        raise ValueError("Input image is None.")

    img = image.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # THRESHOLDING
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # NOISE REMOVAL
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # SURE BACKGROUND
    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    # SURE FOREGROUND
    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.7 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # UNKNOWN REGION
    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # MARKER LABELING
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    # Shift labels so background is 1
    markers = markers + 1

    # Unknown region = 0
    markers[unknown == 255] = 0

    # APPLY WATERSHED
    markers = cv2.watershed(
        img,
        markers
    )

    # CREATE VISUALIZATION
    segmented_color = img.copy()

    # Watershed boundaries in red
    segmented_color[markers == -1] = [0, 0, 255]

    # CREATE REGION MASKS
    unique_regions = np.unique(markers)

    masks = {}

    for region in unique_regions:

        # Skip boundaries and background
        if region <= 1:
            continue

        mask = np.zeros(
            markers.shape,
            dtype=np.uint8
        )

        mask[markers == region] = 255

        masks[int(region)] = mask

    return segmented_color, markers, masks