from .sift import detect_sift_features


def detect_features(image, method="sift"):

    if method == "sift":
        return detect_sift_features(image)

    raise ValueError(
        f"Unknown feature detection method: {method}"
    )