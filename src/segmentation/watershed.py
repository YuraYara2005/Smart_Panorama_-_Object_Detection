import cv2
import numpy as np

from .segmentation_preprocessing import preprocess_image


def watershed_segment(image):

    img = preprocess_image(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Smooth image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Background area
    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=1
    )

    # Distance transform
    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # Less aggressive foreground threshold
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.3 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # Connected components
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(
        img,
        markers
    )

    # Build segmented image
    segmented = np.zeros_like(img)

    segmented[markers > 1] = image[markers > 1]

    return segmented, markers
# import cv2
# import numpy as np
#
# def watershed_segment(image):
#     img = image.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Thresholding
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # Noise removal
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#
#     # Background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
#
#     # Foreground area (Distance Transform)
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
#
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
#
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
#     # Mark the region of unknown with zero
#     markers[unknown == 255] = 0
#
#     # Apply watershed
#     markers = cv2.watershed(img, markers)
#
#     # Create a colored version for visualization
#     # We use a color map that makes label 1 (BG) and label 2+ (FG) distinct
#     display_markers = np.int32(markers)
#     # Map -1 to 0, and scale others for visibility
#     vis_img = np.zeros_like(img)
#     vis_img[markers == -1] = [0, 0, 255] # Red boundaries
#
#     # Apply a Jet colormap to the labels for the "segmented_image" output
#     norm_markers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     segmented_color = cv2.applyColorMap(norm_markers, cv2.COLORMAP_JET)
#
#     return segmented_color, markers