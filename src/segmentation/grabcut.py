import cv2
import numpy as np


def refine_mask_grabcut(image, mask):

    gc_mask = np.where(
        mask > 0,
        cv2.GC_PR_FGD,
        cv2.GC_BGD
    ).astype("uint8")

    bgdModel = np.zeros((1, 65), np.float64)

    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    refined = np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype("uint8")

    return refined