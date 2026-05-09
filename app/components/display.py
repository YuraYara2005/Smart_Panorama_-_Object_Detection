import streamlit as st
import cv2
import numpy as np
from typing import Optional, List, Dict


def display_image(image: np.ndarray, title: str = "") -> None:
    """
    Displays a single BGR image in the Streamlit app.
    """
    if image is None:
        return

    # Convert BGR to RGB for display
    if image.ndim == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    if title:
        st.subheader(title)
    st.image(rgb_image, width="stretch")


def display_image_grid(images: List[np.ndarray], titles: List[str]) -> None:
    """
    Displays multiple images side by side in a grid.
    """
    if not images:
        return

    cols = st.columns(min(len(images), 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        with cols[i % 4]:
            display_image(img, title=title)


def display_metrics(metrics: Dict) -> None:
    """
    Displays evaluation metrics (MSE, PSNR, SSIM, SNR) in a clean table.
    """
    if not metrics:
        return

    st.subheader("Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE",  f"{metrics.get('MSE',  0):.4f}")
    col2.metric("PSNR", f"{metrics.get('PSNR', 0):.2f} dB")
    col3.metric("SSIM", f"{metrics.get('SSIM', 0):.4f}")
    col4.metric("SNR",  f"{metrics.get('SNR',  0):.2f} dB")


def display_pipeline_results(results: Dict) -> None:
    """
    Displays all pipeline results section by section.
    Sections are only shown if their results exist.
    """
    if not results:
        st.warning("No results to display yet. Upload images and run the pipeline.")
        return

    # ── Preprocessing ──
    if "preprocessing" in results:
        st.markdown("---")
        st.header("Preprocessing Results")
        prep = results["preprocessing"]

        display_image_grid(
            images=[prep.get("original"), prep.get("gaussian"), prep.get("median")],
            titles=["Original", "Gaussian Filter", "Median Filter"]
        )

        if "metrics_gaussian" in prep:
            st.write("**Gaussian Filter Metrics:**")
            display_metrics(prep["metrics_gaussian"])

        if "metrics_median" in prep:
            st.write("**Median Filter Metrics:**")
            display_metrics(prep["metrics_median"])

    # ── Pyramids ──
    if "pyramids" in results:
        st.markdown("---")
        st.header("Pyramid Results")
        pyr = results["pyramids"]

        if "gaussian_pyramid" in pyr:
            st.subheader("Gaussian Pyramid")
            display_image_grid(
                images=pyr["gaussian_pyramid"],
                titles=[f"Level {i}" for i in range(len(pyr["gaussian_pyramid"]))]
            )

        if "reconstructed" in pyr:
            st.subheader("Laplacian Reconstruction")
            display_image_grid(
                images=[pyr.get("original"), pyr.get("reconstructed")],
                titles=["Original", "Reconstructed"]
            )

    # ── Placeholders for future modules ──
    if "stitching" in results:
        st.markdown("---")
        st.header("Panorama Stitching Result")
        display_image(results["stitching"].get("panorama"), title="Stitched Panorama")

    if "segmentation" in results:
        st.markdown("---")
        st.header("Segmentation Result")
        display_image(results["segmentation"].get("segmented_image"), title="Segmented Image")

    if "classification" in results:
        st.markdown("---")
        st.header("Classification Result")
        st.write(results["classification"].get("labels", "No labels yet."))