"""Masked normalized cross-correlation of complex feature images via FFT."""

from __future__ import annotations

import copy

import cv2
import numpy as np
from scipy import fft


class Correlator:
    """Slides one masked template over a fixed search image.

    The image-side spectra and the template-mask normalization are computed
    once, so scoring many template variants (headings, scales) that share the
    same mask costs one forward and one inverse FFT each.

    Scores are cosine similarities in [-1, 1] for every offset ``(dy, dx)`` at
    which the template's top-left corner sits inside the image with the
    template fully inside it. Offsets where less than ``min_overlap`` of the
    template mask lands on valid image pixels score -1.
    """

    def __init__(
        self,
        image: np.ndarray,
        image_mask: np.ndarray,
        template_mask: np.ndarray,
        min_overlap: float = 0.75,
    ):
        H, W = image.shape
        h, w = template_mask.shape
        if h > H or w > W:
            raise ValueError("template larger than search image")
        self._shape = (fft.next_fast_len(H), fft.next_fast_len(W))
        self._out = (H - h + 1, W - w + 1)
        self._tshape = (h, w)
        self._min_overlap = min_overlap

        img = np.where(image_mask, image, 0).astype(np.complex64)
        self._F_img = fft.fft2(img, s=self._shape, workers=-1)
        energy = (img.real**2 + img.imag**2).astype(np.float32)
        self._F_energy = fft.rfft2(energy, s=self._shape, workers=-1)
        self._F_valid = fft.rfft2(image_mask.astype(np.float32), s=self._shape, workers=-1)
        self._normalize(template_mask)

    def with_mask(self, template_mask: np.ndarray) -> Correlator:
        """The same search image with another template mask of the same size."""
        if template_mask.shape != self._tshape:
            raise ValueError("template mask size differs")
        other = copy.copy(self)
        other._normalize(template_mask)
        return other

    def _normalize(self, template_mask: np.ndarray) -> None:
        self._mask_t = template_mask.astype(np.float32)
        F_mask_t = np.conj(fft.rfft2(self._mask_t, s=self._shape, workers=-1))
        energy = self._rcorr(F_mask_t, self._F_energy)
        overlap = self._rcorr(F_mask_t, self._F_valid)
        self._invalid = overlap < self._min_overlap * self._mask_t.sum()
        self._inv_energy = 1.0 / np.sqrt(np.maximum(energy, 1e-9))

    def _rcorr(self, F_a_conj: np.ndarray, F_b: np.ndarray) -> np.ndarray:
        out = fft.irfft2(F_a_conj * F_b, s=self._shape, workers=-1)
        return out[: self._out[0], : self._out[1]]

    def score(self, template: np.ndarray) -> np.ndarray:
        """Normalized correlation map for a complex template (zero outside its mask)."""
        t = np.where(self._mask_t > 0, template, 0).astype(np.complex64)
        t_norm = float(np.sqrt((t.real**2 + t.imag**2).sum()))
        if t_norm < 1e-9:
            return np.full(self._out, -1.0, np.float32)
        F_t = np.conj(fft.fft2(t, s=self._shape, workers=-1))
        num = fft.ifft2(F_t * self._F_img, workers=-1).real[: self._out[0], : self._out[1]]
        s = (num * self._inv_energy / t_norm).astype(np.float32)
        s[self._invalid] = -1.0
        return s


def find_peaks(score: np.ndarray, k: int, min_distance: int) -> list[tuple[int, int, float]]:
    """Up to ``k`` local maxima ``(row, col, value)``, at least ``min_distance`` apart."""
    size = 2 * max(min_distance, 1) + 1
    dil = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    rows, cols = np.nonzero((score >= dil) & (score > -1))
    order = np.argsort(score[rows, cols])[::-1]
    peaks: list[tuple[int, int, float]] = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        if all((r - pr) ** 2 + (c - pc) ** 2 >= min_distance**2 for pr, pc, _ in peaks):
            peaks.append((r, c, float(score[r, c])))
            if len(peaks) == k:
                break
    return peaks


def peak_excess(score: np.ndarray, r: int, c: int, rho: float) -> float:
    """How far the peak at ``(r, c)`` stands above the noise of its own search window.

    The peak's robust z-score against all offsets outside its main lobe, minus
    the z expected for the best of ``N`` independent noise samples (``N`` = valid
    area / correlation area, ``rho`` the correlation length in pixels). Unlike
    the raw score this does not depend on how much texture the scene has or on
    how large the window is, so the same threshold works in simulation and on
    real imagery. Above ~1.5 a peak is unlikely to be noise.
    """
    noise = _noise(score, r, c, rho)
    if noise is None:
        return 0.0
    mu, sigma, n = noise
    n_eff = max(n / (np.pi * rho**2), 2.0)
    return float((score[r, c] - mu) / sigma - np.sqrt(2 * np.log(n_eff)))


def peak_covariance(
    score: np.ndarray, r: int, c: int, rho: float, open_variance: float = 0.0
) -> np.ndarray:
    """Spread (2x2, pixels squared, row/column) of the offsets that score as well as the peak.

    Offsets within noise of the peak (one robust sigma, or a tenth of the
    peak's height above the median) are about as plausible as the peak itself.
    Their second moment about the peak is small for a sharp peak but long
    along a ridge (a straight edge seen along its length, which fixes position
    only across it) and between repeats of a repeated pattern. A ridge that
    runs out of the window is longer than measured: its variance along it is
    raised to ``open_variance``.
    """
    noise = _noise(score, r, c, rho)
    if noise is None:
        return np.zeros((2, 2))
    mu, sigma, _ = noise
    peak = score[r, c]
    rows, cols = np.nonzero(score >= peak - max(sigma, 0.1 * (peak - mu)))
    d = np.column_stack([rows - r, cols - c]).astype(float)
    cov = d.T @ d / len(d)
    h, w = score.shape
    edge = rho  # within a correlation length of the window's edge
    if min(rows.min(), cols.min()) < edge or rows.max() >= h - edge or cols.max() >= w - edge:
        ev, vec = np.linalg.eigh(cov)
        if ev[1] > 4 * max(ev[0], 1e-9) and ev[1] < open_variance:  # elongated
            cov = vec @ np.diag([ev[0], open_variance]) @ vec.T
    return cov


def _noise(score: np.ndarray, r: int, c: int, rho: float) -> tuple[float, float, int] | None:
    """Robust median and sigma of the scores outside the peak's main lobe, and their count."""
    yy, xx = np.ogrid[: score.shape[0], : score.shape[1]]
    far = ((yy - r) ** 2 + (xx - c) ** 2 > (3 * rho) ** 2) & (score > -1)
    n = int(far.sum())
    if n < 20:
        return None
    vals = score[far]
    mu = float(np.median(vals))
    return mu, 1.4826 * float(np.median(np.abs(vals - mu))) + 1e-6, n


def subpixel(score: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Refine an integer peak with a separable parabola fit."""
    dr = _parabola(score, (r - 1, c), (r, c), (r + 1, c))
    dc = _parabola(score, (r, c - 1), (r, c), (r, c + 1))
    return r + dr, c + dc


def parabola_offset(a: float, b: float, c: float) -> float:
    """Vertex offset in [-0.5, 0.5] of a parabola through (-1, a), (0, b), (1, c)."""
    den = a - 2 * b + c
    if den >= 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def _parabola(score: np.ndarray, p0, p1, p2) -> float:
    h, w = score.shape
    if not all(0 <= p[0] < h and 0 <= p[1] < w for p in (p0, p2)):
        return 0.0
    return parabola_offset(score[p0], score[p1], score[p2])
