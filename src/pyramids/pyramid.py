import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple


#  GAUSSIAN PYRAMID

def build_gaussian_pyramid(image: np.ndarray,
                            levels: int = 5,
                            sigma: float = 1.0) -> List[np.ndarray]:

    pyramid = [image.copy()]
    current = image.copy()

    for _ in range(levels - 1):
        # Blur then halve resolution
        blurred = cv2.GaussianBlur(current, (5, 5), sigmaX=sigma, sigmaY=sigma)
        downsampled = cv2.pyrDown(blurred)
        pyramid.append(downsampled)
        current = downsampled

    return pyramid


#  LAPLACIAN PYRAMID

def build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:

    laplacian_pyramid = []
    levels = len(gaussian_pyramid)

    for i in range(levels - 1):
        g_current = gaussian_pyramid[i].astype(np.int16)
        g_up = cv2.pyrUp(gaussian_pyramid[i + 1])

        # Match spatial dimensions (pyrUp may add 1 pixel)
        h, w = g_current.shape[:2]
        g_up = g_up[:h, :w].astype(np.int16)

        lap = g_current - g_up
        laplacian_pyramid.append(lap)

    # Append the coarsest Gaussian level as the residual
    laplacian_pyramid.append(gaussian_pyramid[-1].astype(np.int16))

    return laplacian_pyramid


def reconstruct_from_laplacian(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:

    reconstructed = laplacian_pyramid[-1].astype(np.float32)

    for lap in reversed(laplacian_pyramid[:-1]):
        upsampled = cv2.pyrUp(reconstructed.astype(np.uint8)).astype(np.float32)
        h, w = lap.shape[:2]
        upsampled = upsampled[:h, :w]
        reconstructed = upsampled + lap.astype(np.float32)

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


#  SCALE-SPACE EXTREMA (keypoint candidates)

def find_scale_space_extrema(gaussian_pyramid: List[np.ndarray],
                              threshold: float = 10.0
                              ) -> List[Tuple[int, int, int, float]]:

    if gaussian_pyramid[0].ndim == 3:
        gp = [cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) for g in gaussian_pyramid]
    else:
        gp = [g.copy() for g in gaussian_pyramid]

    # Build DoG images
    dogs = []
    for i in range(len(gp) - 1):
        up = cv2.resize(gp[i + 1], (gp[i].shape[1], gp[i].shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        dog = gp[i].astype(np.float32) - up.astype(np.float32)
        dogs.append(dog)

    keypoints = []
    for s in range(1, len(dogs) - 1):
        prev_d = cv2.resize(dogs[s - 1], (dogs[s].shape[1], dogs[s].shape[0]))
        curr_d = dogs[s]
        next_d = cv2.resize(dogs[s + 1], (dogs[s].shape[1], dogs[s].shape[0]))

        h, w = curr_d.shape
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                val = curr_d[r, c]
                if abs(val) < threshold:
                    continue
                neighbourhood = np.array([
                    prev_d[r-1:r+2, c-1:c+2],
                    curr_d[r-1:r+2, c-1:c+2],
                    next_d[r-1:r+2, c-1:c+2],
                ])
                is_max = val == neighbourhood.max()
                is_min = val == neighbourhood.min()
                if is_max or is_min:
                    keypoints.append((s, r, c, float(val)))

    return keypoints


#  BLOB DETECTION (LoG across scales)

def detect_blobs_multiscale(image: np.ndarray,
                             min_sigma: float = 1.0,
                             max_sigma: float = 16.0,
                             num_sigma: int = 8,
                             threshold: float = 0.05
                             ) -> List[Tuple[int, int, float]]:

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32) / 255.0

    sigmas = np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma)
    scale_responses = []

    for sigma in sigmas:
        k = int(2 * np.ceil(3 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray_f, (k, k), sigmaX=sigma)
        log_response = (sigma ** 2) * cv2.Laplacian(blurred, cv2.CV_32F)
        scale_responses.append(log_response)

    scale_cube = np.stack(scale_responses, axis=0)   # (num_sigma, H, W)

    blobs = []
    for s_idx in range(1, num_sigma - 1):
        layer = scale_cube[s_idx]
        for r in range(1, layer.shape[0] - 1):
            for c in range(1, layer.shape[1] - 1):
                val = abs(layer[r, c])
                if val < threshold:
                    continue
                neighbourhood = np.abs(scale_cube[s_idx-1:s_idx+2,
                                                   r-1:r+2, c-1:c+2])
                if val == neighbourhood.max():
                    blobs.append((r, c, sigmas[s_idx]))

    return blobs


#  VISUALISATION

def visualise_gaussian_pyramid(pyramid: List[np.ndarray],
                                save_path: Optional[str] = None,
                                title: str = "Gaussian Pyramid") -> None:

    n = len(pyramid)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if n == 1:
        axes = [axes]

    for idx, (ax, level) in enumerate(zip(axes, pyramid)):
        img_show = cv2.cvtColor(level, cv2.COLOR_BGR2RGB) if level.ndim == 3 else level
        ax.imshow(img_show, cmap='gray' if level.ndim == 2 else None)
        h, w = level.shape[:2]
        ax.set_title(f"Level {idx}\n{w}×{h}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Pyramid] Saved Gaussian pyramid → {save_path}")
    plt.close()


def visualise_laplacian_pyramid(laplacian_pyramid: List[np.ndarray],
                                 save_path: Optional[str] = None,
                                 title: str = "Laplacian Pyramid") -> None:

    n = len(laplacian_pyramid)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if n == 1:
        axes = [axes]

    for idx, (ax, level) in enumerate(zip(axes, laplacian_pyramid)):
        # Normalise to [0, 255] for display
        lap_disp = level.astype(np.float32)
        lap_min, lap_max = lap_disp.min(), lap_disp.max()
        if lap_max > lap_min:
            lap_disp = (lap_disp - lap_min) / (lap_max - lap_min) * 255
        lap_disp = lap_disp.astype(np.uint8)

        if lap_disp.ndim == 3:
            lap_disp = cv2.cvtColor(lap_disp, cv2.COLOR_BGR2RGB)

        ax.imshow(lap_disp, cmap='gray' if level.ndim == 2 else None)
        h, w = level.shape[:2]
        ax.set_title(f"Level {idx}\n{w}×{h}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Pyramid] Saved Laplacian pyramid → {save_path}")
    plt.close()


def visualise_reconstruction(original: np.ndarray,
                              reconstructed: np.ndarray,
                              save_path: Optional[str] = None) -> None:

    from src.preprocessing.evaluation import mse, psnr, ssim

    err = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    if err.ndim == 3:
        err_display = err.mean(axis=2)
    else:
        err_display = err

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Laplacian Pyramid Reconstruction Quality", fontsize=13, fontweight='bold')

    def _rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    axes[0].imshow(_rgb(original), cmap='gray' if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(_rgb(reconstructed), cmap='gray' if reconstructed.ndim == 2 else None)
    m_psnr = psnr(original, reconstructed)
    m_ssim = ssim(original, reconstructed)
    axes[1].set_title(f"Reconstructed\nPSNR={m_psnr:.2f} dB  SSIM={m_ssim:.4f}")
    axes[1].axis('off')

    im = axes[2].imshow(err_display, cmap='hot', vmin=0, vmax=20)
    axes[2].set_title(f"Absolute Error\nmax={err_display.max():.2f}")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Pyramid] Saved reconstruction comparison → {save_path}")
    plt.close()


def visualise_multiscale_detection(image: np.ndarray,
                                    blobs: List[Tuple[int, int, float]],
                                    max_display: int = 200,
                                    save_path: Optional[str] = None) -> None:

    display = image.copy()
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    for (r, c, sigma) in blobs[:max_display]:
        radius = int(np.sqrt(2) * sigma)
        cv2.circle(display, (int(c), int(r)), max(radius, 2), (0, 255, 0), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Multi-Scale Blob Detection — {len(blobs)} blobs found", fontweight='bold')

    axes[0].imshow(cv2.cvtColor(image if image.ndim == 3 else
                                 cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                                 cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected Blobs (top {min(max_display, len(blobs))})")
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Pyramid] Saved blob detection → {save_path}")
    plt.close()
