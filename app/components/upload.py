import streamlit as st
import cv2
import numpy as np
from PIL import Image


def upload_images() -> list:
    """
    Renders the image upload widget.
    Returns a list of images as numpy arrays (BGR format for OpenCV).
    """
    st.header("Upload Images")

    uploaded_files = st.file_uploader(
        "Upload one or more images to stitch into a panorama",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    images = []

    if uploaded_files:
        st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")

        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files):
            # Convert uploaded file to numpy array (BGR for OpenCV)
            pil_image = Image.open(file).convert("RGB")
            np_image = np.array(pil_image)
            bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            images.append(bgr_image)

            # Show thumbnail preview
            with cols[i % 4]:
                st.image(pil_image, caption=file.name, use_container_width=True)
    else:
        st.info("Please upload at least 2 images to create a panorama.")

    return images
