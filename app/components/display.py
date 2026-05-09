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
    st.image(rgb_image, use_column_width=True)


def display_image_grid(images: List[np.ndarray], titles: List[str]) -> None:
    """
    Displays multiple images side by side in a grid.
    """
    if not images:
        return

    # Filter out None images
    valid = [(img, title) for img, title in zip(images, titles) if img is not None]
    if not valid:
        return

    cols = st.columns(min(len(valid), 4))
    for i, (img, title) in enumerate(valid):
        with cols[i % 4]:
            display_image(img, title=title)


def display_metrics(metrics: Dict) -> None:
    """
    Displays evaluation metrics (MSE, PSNR, SSIM, SNR) in a clean table.
    """
    if not metrics:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE",  f"{metrics.get('MSE',  0):.4f}")
    col2.metric("PSNR", f"{metrics.get('PSNR', 0):.2f} dB")
    col3.metric("SSIM", f"{metrics.get('SSIM', 0):.4f}")
    col4.metric("SNR",  f"{metrics.get('SNR',  0):.2f} dB")


def display_pipeline_results(results: Dict) -> None:
    """
    Displays all pipeline results section by section.
    """
    if not results:
        st.warning("No results to display yet. Upload images and run the pipeline.")
        return

    # ── Preprocessing ──
    if "preprocessing" in results:
        st.markdown("---")
        st.header("🔧 Preprocessing Results")
        prep = results["preprocessing"]

        display_image_grid(
            images=[
                prep.get("original"),
                prep.get("noisy"),
                prep.get("gaussian"),
                prep.get("median")
            ],
            titles=[
                "Original",
                "Noisy",
                "Gaussian Filter",
                "Median Filter"
            ]
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
        st.header("🔺 Pyramid Results")
        pyr = results["pyramids"]

        if "gaussian_pyramid" in pyr:
            st.subheader("Gaussian Pyramid")
            display_image_grid(
                images=pyr["gaussian_pyramid"],
                titles=[f"Level {i}" for i in range(len(pyr["gaussian_pyramid"]))]
            )

        if "laplacian_pyramid" in pyr:
            st.subheader("Laplacian Pyramid")
            # Normalize laplacian levels for display
            normalized = []
            for level in pyr["laplacian_pyramid"]:
                level_f = level.astype(np.float32)
                level_min, level_max = level_f.min(), level_f.max()
                if level_max > level_min:
                    level_f = (level_f - level_min) / (level_max - level_min) * 255
                normalized.append(level_f.astype(np.uint8))
            display_image_grid(
                images=normalized,
                titles=[f"Level {i}" for i in range(len(normalized))]
            )

        if "reconstructed" in pyr:
            st.subheader("Laplacian Reconstruction")
            display_image_grid(
                images=[pyr.get("original"), pyr.get("reconstructed")],
                titles=["Original", "Reconstructed"]
            )

    # ── Feature Detection ──
    if "feature_detection" in results:
        st.markdown("---")
        st.header("🔍 Feature Detection & Matching")
        feat = results["feature_detection"]

        display_image_grid(
            images=[
                feat.get("kp_img1"),
                feat.get("kp_img2")
            ],
            titles=[
                "Keypoints — Image 1",
                "Keypoints — Image 2"
            ]
        )

        if "match_img" in feat:
            st.subheader("Feature Matches")
            display_image(feat["match_img"])
            st.write(f"**Total good matches:** {len(feat.get('matches', []))}")

    # ── Stitching ──
    if "stitching" in results:
        st.markdown("---")
        st.header("🖼️ Panorama Stitching")
        stitch = results["stitching"]

        if "panorama" in stitch:
            display_image(stitch["panorama"], title="Stitched Panorama")

    # ── Segmentation ──
    if "segmentation" in results:
        st.markdown("---")
        st.header("🎨 Segmentation Results")
        seg = results["segmentation"]

        if "kmeans_segmented" in seg:
            st.subheader("KMeans Segmentation")
            display_image(seg["kmeans_segmented"])

        if "watershed_segmented" in seg:
            st.subheader("Watershed Segmentation")
            display_image(seg["watershed_segmented"])

            # Show watershed boundaries if labels exist
            if "watershed_labels" in seg:
                boundary_mask = np.zeros(
                    seg["watershed_labels"].shape,
                    dtype=np.uint8
                )
                boundary_mask[seg["watershed_labels"] == -1] = 255
                kernel = np.ones((3, 3), np.uint8)
                boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)
                st.subheader("Watershed Boundaries")
                st.image(boundary_mask, use_column_width=True)
