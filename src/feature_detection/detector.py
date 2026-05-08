from .sift import detect_sift_features
from .harris import detect_harris_corners


def detect_features(
    image,
    method="sift"
):
    """
    Generic feature detector dispatcher.
    
    - Harris output CANNOT go into the matcher directly.
    """

    if method == "sift":

        keypoints, descriptors = (
            detect_sift_features(image)
        )

        return keypoints, descriptors

    elif method == "harris":

        keypoints, response = (
            detect_harris_corners(image)
        )

        return keypoints, response

    raise ValueError(
        f"Unknown feature detection method: {method}"
    )