import streamlit as st
import yaml
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def render_controls(config: Dict) -> Dict:
    st.sidebar.title("⚙️ Pipeline Controls")

    settings = {}

    # ── Preprocessing ──
    st.sidebar.markdown("---")
    st.sidebar.header("Preprocessing")

    settings["run_preprocessing"] = st.sidebar.checkbox(
        "Run Preprocessing",
        value=config["pipeline"]["run_preprocessing"]
    )

    if settings["run_preprocessing"]:
        st.sidebar.subheader("Gaussian Filter")
        settings["gaussian_kernel_size"] = st.sidebar.select_slider(
            "Gaussian Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=config["preprocessing"]["gaussian"]["kernel_size"]
        )
        settings["gaussian_sigma"] = st.sidebar.slider(
            "Gaussian Sigma",
            min_value=0.1,
            max_value=5.0,
            value=float(config["preprocessing"]["gaussian"]["sigma"]),
            step=0.1
        )

        st.sidebar.subheader("Median Filter")
        settings["median_kernel_size"] = st.sidebar.select_slider(
            "Median Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=config["preprocessing"]["median"]["kernel_size"]
        )

        st.sidebar.subheader("Noise")
        settings["noise_type"] = st.sidebar.selectbox(
            "Noise Type",
            options=["gaussian", "salt_pepper"],
            index=0 if config["preprocessing"]["noise"]["type"] == "gaussian" else 1
        )
        settings["noise_intensity"] = st.sidebar.slider(
            "Noise Intensity",
            min_value=1.0,
            max_value=100.0,
            value=float(config["preprocessing"]["noise"]["intensity"]),
            step=1.0
        )

    # ── Pyramids ──
    st.sidebar.markdown("---")
    st.sidebar.header("Pyramids")

    settings["run_pyramids"] = st.sidebar.checkbox(
        "Run Pyramids",
        value=config["pipeline"]["run_pyramids"]
    )

    if settings["run_pyramids"]:
        settings["pyramid_levels"] = st.sidebar.slider(
            "Pyramid Levels",
            min_value=2,
            max_value=8,
            value=config["pyramids"]["levels"],
            step=1
        )
        settings["pyramid_sigma"] = st.sidebar.slider(
            "Pyramid Sigma",
            min_value=0.1,
            max_value=3.0,
            value=float(config["pyramids"]["sigma"]),
            step=0.1
        )

    # ── Feature Detection ──
    st.sidebar.markdown("---")
    st.sidebar.header("Feature Detection & Matching")

    settings["run_feature_detection"] = st.sidebar.checkbox(
        "Run Feature Detection",
        value=config["pipeline"]["run_feature_detection"]
    )

    if settings["run_feature_detection"]:
        st.sidebar.info("Upload at least 2 images for feature detection and matching.")

    # ── Stitching ──
    st.sidebar.markdown("---")
    st.sidebar.header("Stitching")

    settings["run_stitching"] = st.sidebar.checkbox(
        "Run Stitching",
        value=config["pipeline"]["run_stitching"]
    )

    if settings["run_stitching"]:
        st.sidebar.info("Stitching requires feature detection to be enabled.")

    # ── Segmentation ──
    st.sidebar.markdown("---")
    st.sidebar.header("Segmentation")

    settings["run_segmentation"] = st.sidebar.checkbox(
        "Run Segmentation",
        value=config["pipeline"]["run_segmentation"]
    )

    if settings["run_segmentation"]:
        settings["segmentation_method"] = st.sidebar.selectbox(
            "Segmentation Method",
            options=["both", "kmeans", "watershed"],
            index=0
        )
        st.sidebar.info("Segmentation runs on the stitched panorama.")

    # ── Classification (not ready yet) ──
    st.sidebar.markdown("---")
    st.sidebar.header("Classification")
    st.sidebar.checkbox(
        "Run Classification",
        value=False,
        disabled=True,
        help="Not implemented yet."
    )

    # ── Run Button ──
    st.sidebar.markdown("---")
    settings["run_pipeline"] = st.sidebar.button(
        "▶️ Run Pipeline",
        use_container_width=True
    )

    return settings
