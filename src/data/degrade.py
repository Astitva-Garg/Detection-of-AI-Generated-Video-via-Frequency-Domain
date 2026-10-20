"""Post-processing degradations for robustness evaluation."""
import cv2
import numpy as np

from src.config import FPS
from src.data.video_io import h264_roundtrip


def _h264(crf):
    return lambda f: h264_roundtrip(f, FPS, crf)


def _down(f, s=0.5):
    h, w = f.shape[1:3]
    small = [cv2.resize(x, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) for x in f]
    return np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in small])


def _blur(f, sigma=1.0):
    return np.stack([cv2.GaussianBlur(x, (0, 0), sigma) for x in f])


def _noise(f, std=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(f.astype(np.float32) + rng.normal(0, std, f.shape), 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "h264_crf23": _h264(23),
    "h264_crf30": _h264(30),
    "h264_crf38": _h264(38),
    "down0.5": _down,
    "blur1.0": _blur,
    "noise5": _noise,
}


def apply_degradation(frames: np.ndarray, name: str) -> np.ndarray:
    out = DEGRADATIONS[name](frames)
    return np.ascontiguousarray(out[: len(frames)]).astype(np.uint8)
