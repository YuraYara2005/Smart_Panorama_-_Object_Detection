import numpy as np

def iou(mask1, mask2):
    # Convert to boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    union_sum = np.sum(union)

    if union_sum == 0:
        return 0.0  # avoid division by zero

    return np.sum(intersection) / union_sum