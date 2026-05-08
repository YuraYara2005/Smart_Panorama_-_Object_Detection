import cv2
import numpy as np
def filter_mask(mask):

    area = np.sum(mask)

    if area < 500:
        return False

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return False

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)

    contour_area = cv2.contourArea(cnt)

    solidity = contour_area / (
        hull_area + 1e-6
    )

    if solidity < 0.4:
        return False

    return True