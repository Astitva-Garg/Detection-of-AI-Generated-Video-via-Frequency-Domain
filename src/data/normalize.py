"""Identical normalisation for every clip (real and fake) before feature extraction."""
from pathlib import Path

import numpy as np

from src.config import FPS, N_FRAMES, SIZE
from src.data.video_io import h264_roundtrip, probe, read_frames, resize_crop, resize_short, write_video


def normalize_frames(frames: np.ndarray, pre_short_side: int | None = None) -> np.ndarray:
    """Optional pre-downscale + H.264 round trip (real clips only), then resize/crop to SIZE."""
    if pre_short_side is not None and min(frames.shape[1:3]) > pre_short_side:
        frames = resize_short(frames, pre_short_side)
        frames = h264_roundtrip(frames, FPS, crf=18)
    return resize_crop(frames, SIZE)


def normalize_clip(src, dst, pre_short_side: int | None = None, start: float = 0.0) -> dict:
    meta = probe(src)
    frames = read_frames(src, N_FRAMES, FPS, start)
    out = normalize_frames(frames, pre_short_side)
    write_video(Path(dst), out, FPS, crf=18)
    return {"orig_w": meta["w"], "orig_h": meta["h"], "orig_fps": meta["fps"],
            "duration": meta["duration"], "n_src_frames": meta["frames"]}
