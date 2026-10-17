import cv2
import numpy as np

from pharos.vision.correlate import (
    Correlator,
    find_peaks,
    parabola_offset,
    peak_covariance,
    peak_excess,
    subpixel,
)
from pharos.vision.features import orientation_field
from pharos.vision.rectify import disk_mask


def texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def test_finds_known_offset_with_subpixel_peak():
    img = texture((200, 260))
    feat = orientation_field(img)
    mask = disk_mask(30)
    corr = Correlator(feat, np.ones(feat.shape, bool), mask)
    score = corr.score(feat[70:131, 110:171])
    r, c, s = find_peaks(score, 1, 10)[0]
    assert (r, c) == (70, 110)
    assert s > 0.99
    rr, cc = subpixel(score, r, c)
    assert abs(rr - 70) < 0.1 and abs(cc - 110) < 0.1


def test_contrast_inversion_still_matches():
    img = texture((150, 150), seed=1)
    corr = Correlator(orientation_field(img), np.ones(img.shape, bool), disk_mask(25))
    inverted = orientation_field(255 - img)[40:91, 50:101]
    score = corr.score(inverted)
    assert np.unravel_index(score.argmax(), score.shape) == (40, 50)


def test_low_overlap_offsets_are_invalid():
    img = texture((100, 100), seed=2)
    valid = np.ones(img.shape, bool)
    valid[:, 50:] = False
    corr = Correlator(orientation_field(img), valid, disk_mask(10), min_overlap=0.75)
    score = corr.score(orientation_field(img)[0:21, 0:21])
    assert np.all(score[:, 60:] == -1.0)


def test_peaks_respect_min_distance():
    score = np.zeros((50, 50), np.float32)
    score[10, 10], score[12, 12], score[40, 40] = 1.0, 0.9, 0.8
    peaks = find_peaks(score, 2, 5)
    assert [(r, c) for r, c, _ in peaks] == [(10, 10), (40, 40)]


def test_parabola_offset():
    assert parabola_offset(1.0, 2.0, 1.0) == 0.0
    assert 0 < parabola_offset(0.0, 1.0, 0.9) <= 0.5


def test_peak_excess_separates_signal_from_window_noise():
    rng = np.random.default_rng(4)
    noise = cv2.GaussianBlur(rng.standard_normal((120, 120)).astype(np.float32), (0, 0), 2)
    noise = 0.25 + 0.05 * noise / noise.std()
    r, c = np.unravel_index(noise.argmax(), noise.shape)
    assert peak_excess(noise, r, c, rho=5) < 1.0  # the best noise peak is unremarkable
    signal = noise.copy()
    signal[60, 60] = 0.6
    assert peak_excess(signal, 60, 60, rho=5) > 3.0


def test_peak_covariance_is_long_along_a_ridge():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, (101, 101))
    yy, xx = np.mgrid[:101, :101]
    sharp = noise + 0.8 * np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / 8)
    ridge = noise + 0.8 * np.exp(-((yy - 50) ** 2) / 8)  # a horizontal line: rows fixed, cols free
    ridge[50, 50] += 0.1
    sharp_cov = peak_covariance(sharp, 50, 50, rho=5)
    ridge_cov = peak_covariance(ridge, 50, 50, rho=5)
    assert np.sqrt(np.linalg.eigvalsh(sharp_cov).max()) < 2
    assert ridge_cov[1, 1] > 100 * ridge_cov[0, 0]  # spread along columns only
    assert np.sqrt(ridge_cov[1, 1]) > 15
