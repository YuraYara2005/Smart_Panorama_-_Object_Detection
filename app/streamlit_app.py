import streamlit as st
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.upload import upload_images
from app.components.controls import load_config, render_controls
from app.components.display import display_pipeline_results
from src.pipeline import run_pipeline


def main():
    # ── Page Setup ──
    st.set_page_config(
        page_title="Smart Panorama & Object Recognition",
        page_icon="🖼️",
        layout="wide"
    )

    st.title("🖼️ Smart Panorama & Object Recognition System")
    st.markdown("Upload images, configure the pipeline from the sidebar, and click **Run Pipeline**.")

    # ── Load Config ──
    config = load_config("config.yaml")

    # ── Sidebar Controls ──
    settings = render_controls(config)

    # ── Image Upload ──
    st.markdown("---")
    images = upload_images()

    # ── Run Pipeline ──
    if settings.get("run_pipeline"):
        if not images:
            st.error("Please upload at least one image before running the pipeline.")
        else:
            with st.spinner("Running pipeline... please wait."):
                results = run_pipeline(images, settings)

            st.success("Pipeline completed!")

            # ── Display Results ──
            display_pipeline_results(results)

    else:
        st.info("Configure the settings in the sidebar and click ▶️ Run Pipeline to start.")


if __name__ == "__main__":
    main()
