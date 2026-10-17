"""Appearance-invariant dense features for cross-domain (drone vs satellite) matching.

Raw intensities do not survive a change of season, sensor or sun angle, but
the *geometry* of the scene does: field boundaries, roads, building outlines
keep their position and orientation. We therefore match on a dense, coherent
orientation field (the structure tensor in complex form):

    z = |g| exp(2i theta),   g = (dI/dx, dI/dy)
    f = G_s * z / (G_n * |g| + floor)

* Doubling the gradient angle makes ``z`` invariant to contrast polarity (a
  field darker than its neighbour in summer may be brighter in winter).
* Averaging ``z`` over a window (``G_s``) keeps edges, whose orientation is
  consistent, and cancels isotropic texture such as tree canopies, whose
  appearance changes completely with sun angle and season.
* Dividing by the local gradient magnitude removes illumination, haze and
  contrast differences, so ``|f|`` is roughly the local orientation coherence.
* Subtracting the regional mean orientation removes texture whose direction
  is set by the sun (canopy relief shading) or by the season (crop rows),
  keeping boundaries, which are what persist between acquisitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    pre_blur: float = 1.0  # Gaussian sigma (px) before differentiation
    smooth: float = 2.5  # orientation averaging sigma (px); suppresses texture
    norm_sigma: float = 6.0  # window (px) for local magnitude normalization
    norm_floor: float = 0.2  # fraction of the mean magnitude added to the normalizer
    highpass: float = 8.0  # sigma (px) of the regional mean orientation to remove; 0 = off


def orientation_field(
    gray: np.ndarray, mask: np.ndarray | None = None, cfg: FeatureConfig | None = None
) -> np.ndarray:
    """Complex64 orientation field of a gray image. Zero where ``mask`` is False."""
    cfg = cfg or FeatureConfig()
    img = gray.astype(np.float32)
    if cfg.pre_blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), cfg.pre_blur)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    blur = _MaskedBlur(None) if mask is None else _masked_blur_for(mask)
    if blur.mask is not None:
        mag *= blur.mask

    # |g| e^{2i theta} = (gx + i gy)^2 / |g|, as a two-channel (re, im) image
    inv = 1.0 / (mag + 1e-6)
    z = blur(cv2.merge([(gx * gx - gy * gy) * inv, (2.0 * gx * gy) * inv]), cfg.smooth)

    valid = blur.mask.sum() if blur.mask is not None else mag.size
    mean_mag = float(mag.sum() / max(valid, 1.0))
    z *= (1.0 / (blur(mag, cfg.norm_sigma) + cfg.norm_floor * mean_mag + 1e-6))[..., None]
    if cfg.highpass > 0:
        z -= blur(z, cfg.highpass)
    out = (z[..., 0] + 1j * z[..., 1]).astype(np.complex64)
    if blur.mask is not None:
        out[blur.mask == 0] = 0
    return out


class _MaskedBlur:
    """Gaussian blur, normalized inside a mask; caches the mask's own blur per sigma."""

    def __init__(self, mask: np.ndarray | None):
        self.mask = mask
        self._inv_den: dict[float, np.ndarray] = {}

    def __call__(self, img: np.ndarray, sigma: float) -> np.ndarray:
        if self.mask is None:
            return cv2.GaussianBlur(img, (0, 0), sigma)
        if sigma not in self._inv_den:
            den = cv2.GaussianBlur(self.mask, (0, 0), sigma)
            self._inv_den[sigma] = 1.0 / np.maximum(den, 1e-3)
        m, inv = self.mask, self._inv_den[sigma]
        if img.ndim == 3:
            m, inv = m[..., None], inv[..., None]
        return cv2.GaussianBlur(img * m, (0, 0), sigma).reshape(img.shape) * inv


def _masked_blur_for(mask: np.ndarray) -> _MaskedBlur:
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    return _cached_blur(m.tobytes(), m.shape)


@lru_cache(maxsize=16)
def _cached_blur(key: bytes, shape: tuple[int, int]) -> _MaskedBlur:
    mask = np.frombuffer(key, np.uint8).reshape(shape)
    # Gradients near the mask border see the artificial edge of the warp.
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(np.float32)
    return _MaskedBlur(eroded)
