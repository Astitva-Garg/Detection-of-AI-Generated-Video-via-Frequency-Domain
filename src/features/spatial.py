"""Per-frame spatial frequency-domain features."""
from functools import lru_cache

import cv2
import numpy as np
from scipy.fft import dctn

from src.config import N_BINS

EPS = 1e-8


def to_luma(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


@lru_cache(maxsize=8)
def _geometry(h: int, w: int, n_bins: int):
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5            # 1.0 = Nyquist along an axis
    ang = np.mod(np.arctan2(fy, fx), np.pi)          # orientation in [0, pi)
    bins = np.minimum((r * n_bins).astype(int), n_bins)  # n_bins = corner region (> Nyquist), dropped
    counts = np.bincount(bins.ravel(), minlength=n_bins + 1)[:n_bins]
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return r, ang, bins, counts, win


def power_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centred 2D power spectrum of the Hann-windowed, mean-removed frame."""
    h, w = gray.shape
    win = _geometry(h, w, N_BINS)[4]
    g = (gray - gray.mean()) * win
    return np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2


def _radial_mean(values: np.ndarray, n_bins: int) -> np.ndarray:
    h, w = values.shape
    _, _, bins, counts, _ = _geometry(h, w, n_bins)
    s = np.bincount(bins.ravel(), weights=values.ravel(), minlength=n_bins + 1)[:n_bins]
    return s / np.maximum(counts, 1)


def radial_spectrum(gray: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Azimuthally averaged log power, mean removed (shape, not brightness)."""
    prof = _radial_mean(np.log(power_spectrum(gray) + EPS), n_bins)
    return (prof - prof.mean()).astype(np.float32)


def residual(gray: np.ndarray) -> np.ndarray:
    """High-pass noise residual: frame minus its 3x3 median."""
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return gray - cv2.medianBlur(g8, 3).astype(np.float32) / 255.0


def _zigzag_index(n: int = 8) -> np.ndarray:
    idx = np.zeros((n, n), int)
    order = sorted(((i, j) for i in range(n) for j in range(n)),
                   key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 else p[0]))
    for k, (i, j) in enumerate(order):
        idx[i, j] = k
    return idx


_ZZ = _zigzag_index()


def dct_bands(gray: np.ndarray) -> tuple[float, float, float]:
    h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
    blocks = gray[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    e = (dctn(blocks, axes=(2, 3), norm="ortho") ** 2).sum(axis=(0, 1))
    e[0, 0] = 0.0  # drop DC
    tot = e.sum() + EPS
    return (float(e[_ZZ < 6].sum() / tot), float(e[(_ZZ >= 6) & (_ZZ < 28)].sum() / tot),
            float(e[_ZZ >= 28].sum() / tot))


def frame_features(gray: np.ndarray, n_bins: int = N_BINS) -> dict[str, float]:
    h, w = gray.shape
    r, ang, _, _, _ = _geometry(h, w, n_bins)
    P = power_spectrum(gray)
    logprof = _radial_mean(np.log(P + EPS), n_bins)
    rad = logprof - logprof.mean()
    res_rad = radial_spectrum(residual(gray), n_bins)
    feats = {f"rad_{i:02d}": float(v) for i, v in enumerate(rad)}
    feats.update({f"res_rad_{i:02d}": float(v) for i, v in enumerate(res_rad)})

    # spectral slope: log power vs log radius over bins 2..n-1
    centres = (np.arange(n_bins) + 0.5) / n_bins
    sel = slice(2, n_bins)
    feats["slope"] = float(np.polyfit(np.log(centres[sel]), logprof[sel], 1)[0])

    inside = r <= 1.0
    Pin = P[inside]
    rin = r[inside]
    tot = Pin.sum() + EPS
    for c in (25, 50, 75):
        feats[f"hf_ratio_{c}"] = float(Pin[rin > c / 100].sum() / tot)
    feats["centroid"] = float((Pin * rin).sum() / tot)
    order = np.argsort(rin)
    cum = np.cumsum(Pin[order]) / tot
    feats["rolloff85"] = float(rin[order][min(np.searchsorted(cum, 0.85), len(cum) - 1)])

    feats["dct_low"], feats["dct_mid"], feats["dct_high"] = dct_bands(gray)

    hf = inside & (r > 0.25)
    Phf = P[hf]
    wedge = np.minimum((ang[hf] / np.pi * 8).astype(int), 7)
    e = np.bincount(wedge, weights=Phf, minlength=8)
    e = e / (e.sum() + EPS)
    feats.update({f"ang_{i}": float(v) for i, v in enumerate(e)})
    return feats
