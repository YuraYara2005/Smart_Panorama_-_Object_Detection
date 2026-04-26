"""
Part 1 Runner — Preprocessing & Multi-Scale Analysis
======================================================
Runs the full Part 1 pipeline on a batch of panorama dataset images:

  1. Load images from dataset
  2. Add synthetic noise (Gaussian + Salt & Pepper)
  3. Apply Gaussian and Median filters (manual + OpenCV)
  4. Compare visually and numerically (MSE, PSNR, SSIM, SNR)
  5. Build Gaussian & Laplacian pyramids
  6. Reconstruct from Laplacian pyramid and measure quality
  7. Detect blobs at multiple scales
  8. Save all output figures to outputs/part1/

Usage
-----
    python run_part1.py --data_dir data/panorama_sets --n_images 1

Arguments
---------
    --data_dir   Path to folder of images (panorama dataset or any folder of JPEGs/PNGs).
    --n_images   Number of images to process (default: 10).
    --output_dir Directory for saved figures (default: outputs/part1).
    --kernel     Kernel size for filters (default: 5, must be odd).
    --sigma      Gaussian sigma (default: 1.5).
    --pyr_levels Number of pyramid levels (default: 5).
    --img_size   Resize shortest edge to this (default: 512, 0=no resize).
    --grayscale  Process images in grayscale mode.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing.filters import *
from src.preprocessing.evaluation import *

from src.pyramids.pyramid import *
    



# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def load_images(data_dir: str, n: int, img_size: int, grayscale: bool):
    """Load up to n images from data_dir, optionally resize and grayscale."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    paths = sorted([p for p in data_path.rglob('*') if p.suffix.lower() in SUPPORTED_EXT])
    if not paths:
        raise ValueError(f"No supported images found in {data_dir}")

    paths = paths[:n]
    images = []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size > 0:
            h, w = img.shape[:2]
            scale = img_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        images.append((p.name, img))

    print(f"[Loader] Loaded {len(images)} images from {data_dir}")
    return images


def safe_manual_gaussian(image, kernel_size, sigma):
    """Run manual Gaussian; skip if image is too large (slow O(n²) conv)."""
    h, w = image.shape[:2]
    if h * w > 640 * 640:
        print(f"  [Info] Image {h}×{w} too large for manual conv — using OpenCV fallback.")
        return gaussian_filter_cv(image, kernel_size, sigma)
    return gaussian_filter_manual(image, kernel_size, sigma)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing(image, name, kernel, sigma, out_dir, idx):
    """Full filter comparison pipeline for one image."""
    sub_dir = out_dir / "filters"

    # ── Noise variants ──
    noisy_gauss = add_noise(image, 'gaussian', intensity=25)
    noisy_sp    = add_noise(image, 'salt_pepper', intensity=30)

    for noise_label, noisy in [("gaussian_noise", noisy_gauss),
                                ("salt_pepper_noise", noisy_sp)]:

        print(f"\n  [{name}] Noise: {noise_label}")

        # Apply all four filter variants
        g_manual = safe_manual_gaussian(noisy, kernel, sigma)
        g_cv     = gaussian_filter_cv(noisy, kernel, sigma)
        m_manual = median_filter_manual(noisy, kernel)
        m_cv     = median_filter_cv(noisy, kernel)

        filter_results = [
            ("Gaussian Manual",  g_manual),
            ("Gaussian OpenCV",  g_cv),
            ("Median Manual",    m_manual),
            ("Median OpenCV",    m_cv),
        ]

        # Compute metrics vs. clean original
        metrics = [compute_all_metrics(image, out, label=lbl)
                   for lbl, out in filter_results]
        print_metrics_table(metrics)

        # Save visual comparison
        plot_filter_comparison(
            original=image,
            noisy=noisy,
            results=filter_results,
            metrics=metrics,
            save_path=str(sub_dir / f"img{idx:03d}_{noise_label}_comparison.png"),
            title=f"{name} — {noise_label} | k={kernel} σ={sigma}",
        )

        # Save metric bar chart
        plot_metric_bars(
            metrics,
            save_path=str(sub_dir / f"img{idx:03d}_{noise_label}_metrics.png"),
        )

    # Kernel size sensitivity (Gaussian, gaussian noise)
    plot_noise_sensitivity(
        original=image,
        filter_fn=lambda img, k: gaussian_filter_cv(img, k, sigma),
        filter_name="Gaussian Filter",
        kernel_sizes=[3, 5, 7, 11],
        noise_type='gaussian',
        save_path=str(sub_dir / f"img{idx:03d}_gaussian_sensitivity.png"),
    )

    # Kernel size sensitivity (Median, salt & pepper)
    plot_noise_sensitivity(
        original=image,
        filter_fn=lambda img, k: median_filter_cv(img, k),
        filter_name="Median Filter",
        kernel_sizes=[3, 5, 7, 11],
        noise_type='salt_pepper',
        save_path=str(sub_dir / f"img{idx:03d}_median_sensitivity.png"),
    )


def run_pyramids(image, name, levels, sigma, out_dir, idx):
    """Full pyramid analysis pipeline for one image."""
    sub_dir = out_dir / "pyramids"

    # ── Gaussian Pyramid ──
    g_pyr = build_gaussian_pyramid(image, levels=levels, sigma=sigma)
    visualise_gaussian_pyramid(
        g_pyr,
        save_path=str(sub_dir / f"img{idx:03d}_gaussian_pyramid.png"),
        title=f"{name} — Gaussian Pyramid ({levels} levels)",
    )

    # ── Laplacian Pyramid ──
    l_pyr = build_laplacian_pyramid(g_pyr)
    visualise_laplacian_pyramid(
        l_pyr,
        save_path=str(sub_dir / f"img{idx:03d}_laplacian_pyramid.png"),
        title=f"{name} — Laplacian Pyramid ({levels} levels)",
    )

    # ── Reconstruction ──
    recon = reconstruct_from_laplacian(l_pyr)
    visualise_reconstruction(
        original=image,
        reconstructed=recon,
        save_path=str(sub_dir / f"img{idx:03d}_reconstruction.png"),
    )

    # ── Multi-scale blob detection ──
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Limit resolution for speed
    small = cv2.resize(gray, (256, 256))
    blobs = detect_blobs_multiscale(small, min_sigma=1.0, max_sigma=10.0,
                                     num_sigma=6, threshold=0.03)
    # Scale blob coordinates back to original size
    scale_r = gray.shape[0] / 256
    scale_c = gray.shape[1] / 256
    blobs_scaled = [(int(r * scale_r), int(c * scale_c), s)
                    for r, c, s in blobs]
    visualise_multiscale_detection(
        image=image,
        blobs=blobs_scaled,
        max_display=150,
        save_path=str(sub_dir / f"img{idx:03d}_blobs.png"),
    )
    print(f"  [{name}] Blobs detected: {len(blobs_scaled)}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Part 1: Preprocessing & Pyramids")
    parser.add_argument('--data_dir',   default='data/panorama_sets',
                        help='Path to image directory')
    parser.add_argument('--n_images',   type=int, default=10,
                        help='Number of images to process')
    parser.add_argument('--output_dir', default='outputs/part1',
                        help='Directory for output figures')
    parser.add_argument('--kernel',     type=int, default=5,
                        help='Filter kernel size (odd integer)')
    parser.add_argument('--sigma',      type=float, default=1.5,
                        help='Gaussian sigma')
    parser.add_argument('--pyr_levels', type=int, default=5,
                        help='Number of pyramid levels')
    parser.add_argument('--img_size',   type=int, default=512,
                        help='Resize smallest dimension to this (0=no resize)')
    parser.add_argument('--grayscale',  action='store_true',
                        help='Process images in grayscale')
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "filters").mkdir(parents=True, exist_ok=True)
    (out_dir / "pyramids").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(" PART 1 — Preprocessing & Multi-Scale Analysis")
    print(f"{'='*60}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Images     : {args.n_images}")
    print(f"  Kernel     : {args.kernel}  Sigma: {args.sigma}")
    print(f"  Pyr levels : {args.pyr_levels}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*60}\n")

    images = load_images(args.data_dir, args.n_images, args.img_size, args.grayscale)

    total_start = time.time()

    for idx, (name, image) in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] Processing: {name}  {image.shape}")
        t0 = time.time()

        run_preprocessing(image, name, args.kernel, args.sigma, out_dir, idx)
        run_pyramids(image, name, args.pyr_levels, args.sigma, out_dir, idx)

        print(f"  Done in {time.time() - t0:.1f}s")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Part 1 complete — {len(images)} images in {elapsed:.1f}s")
    print(f"  Outputs saved to: {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()