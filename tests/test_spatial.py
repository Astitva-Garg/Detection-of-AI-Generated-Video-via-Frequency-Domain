import cv2
import numpy as np

from src.features.spatial import frame_features, radial_spectrum


def _noise(seed=0, n=256):
    return np.random.default_rng(seed).random((n, n)).astype(np.float32)


def _pink(beta=2.0, n=256, seed=0):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f ** (beta / 2)
    img = np.real(np.fft.ifft2(spec))
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)


def test_white_noise_flat_slope():
    assert -0.3 <= frame_features(_noise())["slope"] <= 0.3


def test_pink_noise_slope():
    assert -2.4 <= frame_features(_pink(2.0))["slope"] <= -1.6


def test_blur_reduces_hf():
    x = _noise()
    xb = cv2.GaussianBlur(x, (0, 0), 2.0)
    assert frame_features(xb)["hf_ratio_50"] < frame_features(x)["hf_ratio_50"]


def test_constant_frame_finite():
    f = frame_features(np.full((256, 256), 0.5, np.float32))
    assert all(np.isfinite(v) for v in f.values())


def test_radial_len():
    assert len(radial_spectrum(_noise())) == 48
    f = frame_features(_noise())
    assert sum(k.startswith("rad_") for k in f) == 48 and sum(k.startswith("ang_") for k in f) == 8
