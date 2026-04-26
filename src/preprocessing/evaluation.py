import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from skimage.metrics import structural_similarity as sk_ssim


#  NUMERIC METRICS

def mse(image_a: np.ndarray, image_b: np.ndarray) -> float:

    a = image_a.astype(np.float64)
    b = image_b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(image_a: np.ndarray, image_b: np.ndarray,
         max_pixel: float = 255.0) -> float:
    
    error = mse(image_a, image_b)
    if error == 0.0:
        return float('inf')
    return 10.0 * np.log10((max_pixel ** 2) / error)


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    
    # Convert to grayscale for SSIM if colour
    if image_a.ndim == 3:
        a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    else:
        a_gray = image_a
        b_gray = image_b

    score, _ = sk_ssim(a_gray.astype(np.float64),
                        b_gray.astype(np.float64),
                        data_range=255.0,
                        full=True)
    return float(score)


def snr(original: np.ndarray, filtered: np.ndarray) -> float:
    
    signal = original.astype(np.float64)
    noise = signal - filtered.astype(np.float64)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_all_metrics(original: np.ndarray,
                         filtered: np.ndarray,
                         label: str = "") -> Dict[str, float]:
    
    return {
        'label': label,
        'MSE':  round(mse(original, filtered), 4),
        'PSNR': round(psnr(original, filtered), 4),
        'SSIM': round(ssim(original, filtered), 4),
        'SNR':  round(snr(original, filtered), 4),
    }


def print_metrics_table(results: List[Dict]) -> None:
    
    header = f"{'Filter':<30} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>8} {'SNR (dB)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['label']:<30} {r['MSE']:>10.4f} {r['PSNR']:>12.4f} "
              f"{r['SSIM']:>8.4f} {r['SNR']:>10.4f}")
    print("=" * len(header) + "\n")


#  VISUAL COMPARISON

def plot_filter_comparison(original: np.ndarray,
                            noisy: np.ndarray,
                            results: List[Tuple[str, np.ndarray]],
                            metrics: Optional[List[Dict]] = None,
                            save_path: Optional[str] = None,
                            title: str = "Filter Comparison") -> None:
    
    n_cols = 2 + len(results)   # original + noisy + each filter
    fig = plt.figure(figsize=(4 * n_cols, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    def _to_rgb(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    images = [("Original", original), ("Noisy Input", noisy)] + results
    metric_map = {}
    if metrics:
        for m in metrics:
            metric_map[m['label']] = m

    for idx, (name, img) in enumerate(images):
        ax = fig.add_subplot(1, n_cols, idx + 1)
        ax.imshow(_to_rgb(img), cmap='gray' if img.ndim == 2 else None)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axis('off')

        if name in metric_map:
            m = metric_map[name]
            metric_str = (f"PSNR: {m['PSNR']:.2f} dB\n"
                          f"SSIM: {m['SSIM']:.4f}\n"
                          f"MSE:  {m['MSE']:.2f}")
            ax.set_xlabel(metric_str, fontsize=8, ha='center',
                          color='#333333', labelpad=6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Saved comparison figure → {save_path}")

    plt.close()


def plot_noise_sensitivity(original: np.ndarray,
                            filter_fn,
                            filter_name: str,
                            kernel_sizes: List[int],
                            noise_type: str = 'gaussian',
                            save_path: Optional[str] = None) -> None:
    
    from .filters import add_noise

    noisy = add_noise(original, noise_type=noise_type)

    n = len(kernel_sizes)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))
    fig.suptitle(f"{filter_name} — Kernel Size Sensitivity ({noise_type} noise)",
                 fontsize=13, fontweight='bold')

    def _to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    # Top row: images; Bottom row: difference maps
    axes[0, 0].imshow(_to_rgb(noisy), cmap='gray' if noisy.ndim == 2 else None)
    axes[0, 0].set_title("Noisy Input", fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(_to_rgb(original), cmap='gray' if original.ndim == 2 else None)
    axes[1, 0].set_title("Clean Reference", fontweight='bold')
    axes[1, 0].axis('off')

    for col, ks in enumerate(kernel_sizes, start=1):
        filtered = filter_fn(noisy, ks)
        m = compute_all_metrics(original, filtered, label=str(ks))

        axes[0, col].imshow(_to_rgb(filtered), cmap='gray' if filtered.ndim == 2 else None)
        axes[0, col].set_title(f"k={ks}", fontweight='bold')
        axes[0, col].axis('off')
        axes[0, col].set_xlabel(f"PSNR={m['PSNR']:.2f}\nSSIM={m['SSIM']:.4f}",
                                fontsize=8, ha='center')

        diff = np.abs(original.astype(np.float32) - filtered.astype(np.float32))
        if diff.ndim == 3:
            diff = diff.mean(axis=2)
        im = axes[1, col].imshow(diff, cmap='hot', vmin=0, vmax=50)
        axes[1, col].set_title(f"Error Map k={ks}")
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Saved sensitivity figure → {save_path}")
    plt.close()


def plot_metric_bars(metrics_list: List[Dict],
                     save_path: Optional[str] = None) -> None:
    
    labels = [m['label'] for m in metrics_list]
    psnr_vals = [m['PSNR'] for m in metrics_list]
    ssim_vals = [m['SSIM'] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Filter Metric Comparison", fontsize=13, fontweight='bold')

    bars1 = ax1.bar(x, psnr_vals, width, color='steelblue', edgecolor='white')
    ax1.set_title("PSNR (dB) — Higher is better")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylabel("dB")
    for bar, val in zip(bars1, psnr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, ssim_vals, width, color='darkorange', edgecolor='white')
    ax2.set_title("SSIM — Higher is better (max=1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, 1.05)
    for bar, val in zip(bars2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Saved metric bars → {save_path}")
    plt.close()
