import streamlit as st
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Loads the config.yaml file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def render_controls(config: Dict) -> Dict:
    """
    Renders all sidebar controls and returns the updated settings.
    """
    st.sidebar.title("⚙️ Pipeline Controls")

    settings = {}

    # ── Preprocessing Controls ──
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
            value=config["preprocessing"]["gaussian"]["sigma"],
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
            value=config["preprocessing"]["noise"]["intensity"],
            step=1.0
        )

    # ── Pyramid Controls ──
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
            value=config["pyramids"]["sigma"],
            step=0.1
        )

    # ── Placeholder controls for future modules ──
    st.sidebar.markdown("---")
    st.sidebar.header("Coming Soon")

    if config["pipeline"]["run_feature_detection"]:
        settings["run_feature_detection"] = st.sidebar.checkbox(
            "Run Feature Detection", value=True
        )

    if config["pipeline"]["run_stitching"]:
        settings["run_stitching"] = st.sidebar.checkbox(
            "Run Stitching", value=True
        )

    if config["pipeline"]["run_segmentation"]:
        settings["run_segmentation"] = st.sidebar.checkbox(
            "Run Segmentation", value=True
        )

    if config["pipeline"]["run_classification"]:
        settings["run_classification"] = st.sidebar.checkbox(
            "Run Classification", value=True
        )

    # ── Run Button ──
    st.sidebar.markdown("---")
    settings["run_pipeline"] = st.sidebar.button(
        "▶️ Run Pipeline",
        use_container_width=True
    )

    return settings
