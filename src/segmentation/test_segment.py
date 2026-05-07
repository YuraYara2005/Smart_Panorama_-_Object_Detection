"""
TESTING SEGMENTATION

"""
import cv2
import numpy as np

from src.segmentation.segmenter import segment
from src.segmentation.metrics import iou

# ---------------------------------------------------
# Load Image
# ---------------------------------------------------

img = cv2.imread("panorama.jpg")

if img is None:
    print("ERROR: image not loaded!")
    exit()

# ===================================================
# KMEANS SEGMENTATION
# ===================================================

print("\n--- Running KMeans Segmentation ---")

kmeans_result = segment(img, method="kmeans")

kmeans_segmented = kmeans_result["segmented_image"]
kmeans_labels = kmeans_result["labels"]

# Save segmented image
cv2.imwrite("segmented_kmeans.jpg", kmeans_segmented)

print("Saved segmented_kmeans.jpg")

# Save cluster masks
unique_labels = np.unique(kmeans_labels)

print("KMeans Clusters Found:", unique_labels)

for i in unique_labels:

    mask = (kmeans_labels == i).astype("uint8") * 255

    cv2.imwrite(f"kmeans_cluster_{i}.png", mask)

print("KMeans cluster masks saved!")

# ---------------------------------------------------
# IoU Evaluation
# ---------------------------------------------------

gt = cv2.imread("mask.png", 0)

if gt is not None:

    gt = (gt > 127)

    # CHANGE after inspecting cluster masks
    chosen_cluster = 3

    pred_mask = (kmeans_labels == chosen_cluster)

    score = iou(pred_mask, gt)

    print("KMeans IoU:", score)

else:
    print("No mask.png found — skipping IoU evaluation.")

# ===================================================
# WATERSHED SEGMENTATION
# ===================================================

print("\n--- Running Watershed Segmentation ---")

watershed_result = segment(img, method="watershed")

watershed_segmented = watershed_result["segmented_image"]
watershed_labels = watershed_result["labels"]

# Save watershed result
cv2.imwrite("watershed_result.jpg", watershed_segmented)

print("Saved watershed_result.jpg")

# ---------------------------------------------------
# Save Watershed Boundary Mask
# ---------------------------------------------------

boundary_mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Watershed boundaries
boundary_mask[watershed_labels == -1] = 255

# Thicken boundaries so they are visible
kernel = np.ones((3, 3), np.uint8)

boundary_mask = cv2.dilate(
    boundary_mask,
    kernel,
    iterations=1
)

cv2.imwrite("watershed_boundaries.png", boundary_mask)

print("Saved watershed_boundaries.png")

# ---------------------------------------------------
# Save Foreground Regions
# ---------------------------------------------------

foreground_mask = np.zeros(img.shape[:2], dtype=np.uint8)

foreground_mask = np.zeros(watershed_labels.shape, dtype=np.uint8)

unique_regions = np.unique(watershed_labels)

for label in unique_regions:

    if label <= 1:
        continue

    foreground_mask[watershed_labels == label] = 255

cv2.imwrite("watershed_foreground.png", foreground_mask)

print("Saved watershed_foreground.png")

print("\nSegmentation testing completed successfully!")