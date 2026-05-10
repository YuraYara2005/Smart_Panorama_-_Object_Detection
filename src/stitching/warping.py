import cv2
import numpy as np


def calculate_panorama_canvas(img1_shape, img2_shape, H):

    h1, w1 = img1_shape
    h2, w2 = img2_shape

    corners_img1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    warped_corners_img1 = cv2.perspectiveTransform(
        corners_img1,
        H
    )

    corners_img2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 2)

    all_corners = np.vstack([
        warped_corners_img1.reshape(4, 2),
        corners_img2
    ])

    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    width = int(x_max - x_min)
    height = int(y_max - y_min)

    return (width, height), T


def warp_and_position_images(img1, img2, H):

    canvas_size, T = calculate_panorama_canvas(
        img1.shape[:2],
        img2.shape[:2],
        H
    )

    width, height = canvas_size

    H_total = T @ H

    warped_img1 = cv2.warpPerspective(
        img1,
        H_total,
        (width, height)
    )

    translated_img2 = np.zeros_like(warped_img1)

    tx = int(T[0, 2])
    ty = int(T[1, 2])

    translated_img2[
        ty:ty + img2.shape[0],
        tx:tx + img2.shape[1]
    ] = img2

    return warped_img1, translated_img2


def crop_black_borders(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    coords = cv2.findNonZero(mask)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    cropped = image[y:y+h, x:x+w]

    return cropped