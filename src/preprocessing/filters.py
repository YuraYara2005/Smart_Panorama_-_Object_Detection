import cv2
import numpy as np
from scipy.ndimage import generic_filter


#  GAUSSIAN FILTER

def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    k = kernel_size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_filter_manual(image: np.ndarray,
                            kernel_size: int = 5,
                            sigma: float = 1.0) -> np.ndarray:

    kernel = _gaussian_kernel(kernel_size, sigma)
    original_dtype = image.dtype

    # Work in float64 for precision
    img_f = image.astype(np.float64)

    if img_f.ndim == 2:
        result = _convolve2d(img_f, kernel)
    else:
        channels = [_convolve2d(img_f[:, :, c], kernel)
                    for c in range(img_f.shape[2])]
        result = np.stack(channels, axis=2)

    result = np.clip(result, 0, 255)
    return result.astype(original_dtype)


def gaussian_filter_cv(image: np.ndarray,
                        kernel_size: int = 5,
                        sigma: float = 1.0) -> np.ndarray:
    
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


#  MEDIAN FILTER

def median_filter_manual(image: np.ndarray,
                          kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    original_dtype = image.dtype
    img_f = image.astype(np.float64)

    def _channel_median(channel: np.ndarray) -> np.ndarray:
        return generic_filter(channel, np.median, size=kernel_size)

    if img_f.ndim == 2:
        result = _channel_median(img_f)
    else:
        channels = [_channel_median(img_f[:, :, c])
                    for c in range(img_f.shape[2])]
        result = np.stack(channels, axis=2)

    result = np.clip(result, 0, 255)
    return result.astype(original_dtype)


def median_filter_cv(image: np.ndarray,
                      kernel_size: int = 5) -> np.ndarray:
    
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    return cv2.medianBlur(image, kernel_size)


#  INTERNAL HELPERS

def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Reflect padding avoids dark borders better than zero-padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def add_noise(image: np.ndarray,
              noise_type: str = 'gaussian',
              intensity: float = 25.0) -> np.ndarray:
    
    noisy = image.astype(np.float64)

    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, image.shape)
        noisy = noisy + noise

    elif noise_type == 'salt_pepper':
        prob = intensity / 255.0
        salt_mask = np.random.rand(*image.shape[:2]) < prob / 2
        pepper_mask = np.random.rand(*image.shape[:2]) < prob / 2
        if image.ndim == 3:
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
        else:
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Use 'gaussian' or 'salt_pepper'.")

    return np.clip(noisy, 0, 255).astype(np.uint8)
