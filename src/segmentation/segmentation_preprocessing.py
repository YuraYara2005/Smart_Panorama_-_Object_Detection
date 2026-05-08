import cv2

def preprocess_image(image):

    filtered = cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return enhanced