"""
Mathematical helper functions for calculating coordinate transformations 
and bounding boxes during image stitching.
"""

from typing import Tuple
import cv2
import numpy as np

def get_transformed_corners(image_shape: Tuple[int, int], H: np.ndarray) -> np.ndarray:

    h, w = image_shape
    
    # Define the 4 corners of the original image
    corners = np.float32([
        [0, 0], 
        [w - 1, 0], 
        [w - 1, h - 1], 
        [0, h - 1]
    ]).reshape(-1, 1, 2)
    
    # Apply the homography matrix to find their new positions
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    return transformed_corners


def get_bounding_box(corners: np.ndarray) -> Tuple[int, int, int, int]:

    # Flatten the array to easily extract X and Y columns
    pts = corners.reshape(-1, 2)
    
    # Use floor and ceil to safely encompass fractional pixel coordinates
    x_min = int(np.floor(pts[:, 0].min()))
    y_min = int(np.floor(pts[:, 1].min()))
    x_max = int(np.ceil(pts[:, 0].max()))
    y_max = int(np.ceil(pts[:, 1].max()))
    
    return x_min, y_min, x_max, y_max