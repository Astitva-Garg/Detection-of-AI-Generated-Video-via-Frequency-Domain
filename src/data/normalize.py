"""Identical normalisation for every clip (real and fake) before feature extraction."""
from pathlib import Path

import numpy as np

from src.config import FPS, N_FRAMES, NORM_MODE, SIZE
from src.data.video_io import h264_roundtrip, probe, read_frames_stride, resize_crop, resize_short, write_video


def centre_crop(frames: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Native-resolution centre crop (no resampling unless the clip is smaller than `size`)."""
    if min(frames.shape[1:3]) < size:
        frames = resize_short(frames, size)
    h, w = frames.shape[1:3]
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(frames[:, y:y + size, x:x + size])


def normalize_frames(frames: np.ndarray, pre_short_side: int | None = None, mode: str = NORM_MODE) -> np.ndarray:
    """Optional pre-downscale + H.264 round trip (real clips only), then resize+crop or native crop to SIZE."""
    if pre_short_side is not None and min(frames.shape[1:3]) > pre_short_side:
        frames = resize_short(frames, pre_short_side)
        frames = h264_roundtrip(frames, FPS, crf=18)
    return centre_crop(frames, SIZE) if mode == "crop" else resize_crop(frames, SIZE)


def normalize_clip(src, dst, pre_short_side: int | None = None, start: float = 0.0) -> dict:
    meta = probe(src)
    frames, _ = read_frames_stride(src, N_FRAMES, FPS, start)
    out = normalize_frames(frames, pre_short_side)
    write_video(Path(dst), out, FPS, crf=18)
    return {"orig_w": meta["w"], "orig_h": meta["h"], "orig_fps": meta["fps"],
            "duration": meta["duration"], "n_src_frames": meta["frames"]}
