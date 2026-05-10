import cv2
import numpy as np

from src.feature_detection.sift import detect_sift_features
from src.matching.matcher import match_features

from src.stitching.stitcher import stitch_image_pair

from src.segmentation.segmenter import segment


# =====================================================
# LOAD IMAGES
# =====================================================

img1 = cv2.imread("data/image1.jpg")
img2 = cv2.imread("data/image2.jpg")

if img1 is None or img2 is None:
    print("ERROR: images not loaded!")
    exit()

print("Images loaded.")


# =====================================================
# FEATURE DETECTION
# =====================================================

kp1, des1 = detect_sift_features(img1)
kp2, des2 = detect_sift_features(img2)

print("Keypoints detected.")
print("Image 1:", len(kp1))
print("Image 2:", len(kp2))


# =====================================================
# FEATURE MATCHING
# =====================================================

matches = match_features(des1, des2)

print("Good matches:", len(matches))


# =====================================================
# STITCHING
# =====================================================

panorama, H, mask = stitch_image_pair(
    img1,
    img2,
    kp1,
    kp2,
    matches
)

cv2.imwrite("panorama_test.jpg", panorama)

print("Saved panorama_test.jpg")


# =====================================================
# KMEANS SEGMENTATION
# =====================================================

kmeans_result = segment(
    panorama,
    method="kmeans"
)

kmeans_segmented = kmeans_result["segmented_image"]
kmeans_labels = kmeans_result["labels"]

cv2.imwrite(
    "kmeans_segmented.jpg",
    kmeans_segmented
)

print("Saved kmeans_segmented.jpg")


# =====================================================
# SAVE KMEANS CLUSTERS
# =====================================================

unique_labels = np.unique(kmeans_labels)

for i in unique_labels:

    mask_img = (
        (kmeans_labels == i)
        .astype("uint8")
    ) * 255

    cv2.imwrite(
        f"kmeans_cluster_{i}.png",
        mask_img
    )

print("Saved KMeans cluster masks.")


# =====================================================
# WATERSHED SEGMENTATION
# =====================================================

watershed_result = segment(
    panorama,
    method="watershed"
)

watershed_segmented = watershed_result["segmented_image"]
watershed_labels = watershed_result["labels"]

cv2.imwrite(
    "watershed_segmented.jpg",
    watershed_segmented
)

print("Saved watershed_segmented.jpg")


# =====================================================
# WATERSHED BOUNDARIES
# =====================================================

boundary_mask = np.zeros(
    panorama.shape[:2],
    dtype=np.uint8
)

boundary_mask[
    watershed_labels == -1
] = 255

kernel = np.ones((3, 3), np.uint8)

boundary_mask = cv2.dilate(
    boundary_mask,
    kernel,
    iterations=1
)

cv2.imwrite(
    "watershed_boundaries.jpg",
    boundary_mask
)

print("Saved watershed_boundaries.jpg")


print("\nFULL PIPELINE TEST COMPLETED SUCCESSFULLY!")