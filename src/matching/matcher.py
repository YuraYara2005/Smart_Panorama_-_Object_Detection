import cv2
import numpy as np


def match_features(desc1, desc2, ratio=0.75):
    """
    Match descriptors using BFMatcher + Lowe ratio test.
    """

    # Safety checks
    if desc1 is None or desc2 is None:
        raise ValueError("Descriptors cannot be None.")

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for pair in matches:

        # Some pairs may contain less than 2 matches
        if len(pair) < 2:
            continue

        m, n = pair

        # Lowe ratio test
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    max_matches=50
):
    """
    Draw feature matches between two images.
    """

    matched = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched