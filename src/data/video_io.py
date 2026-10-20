"""Video decoding/encoding helpers (PyAV) shared by every pipeline stage."""
from pathlib import Path

import av
import cv2
import numpy as np


def _decode_all(path, max_seconds: float | None = None) -> tuple[list[np.ndarray], list[float], float | None]:
    frames, times = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        rate = float(s.average_rate) if s.average_rate else None
        tb = float(s.time_base) if s.time_base else None
        for i, f in enumerate(c.decode(s)):
            if f.pts is not None and tb:
                t = f.pts * tb
            else:
                t = i / (rate or 24.0)
            if max_seconds is not None and times and t > max_seconds:
                break
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(t)
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]
    return frames, times, rate


def probe(path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = float(s.average_rate) if s.average_rate else None
        dur = float(c.duration / 1e6) if c.duration else None
        return {"w": s.codec_context.width, "h": s.codec_context.height, "fps": rate,
                "duration": dur, "frames": s.frames or None}


def _sample_stream(path, targets: np.ndarray, max_seconds: float):
    """Decode once, keeping only the frame nearest to each target time (memory ~ len(targets) frames)."""
    best_d = np.full(len(targets), np.inf)
    best: list = [None] * len(targets)
    times = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        rate = float(s.average_rate) if s.average_rate else None
        tb = float(s.time_base) if s.time_base else None
        t0 = None
        for i, f in enumerate(c.decode(s)):
            t = f.pts * tb if (f.pts is not None and tb) else i / (rate or 24.0)
            t0 = t if t0 is None else t0
            t -= t0
            if times and t > max_seconds:
                break
            times.append(t)
            d = np.abs(targets - t)
            upd = d < best_d
            if upd.any():
                arr = f.to_ndarray(format="rgb24")
                for k in np.nonzero(upd)[0]:
                    best_d[k], best[k] = d[k], arr
    return best, np.asarray(times)


def read_frames(path, n_frames: int = 16, fps: int = 8, start: float = 0.0) -> np.ndarray:
    """Sample `n_frames` at times start + k/fps (nearest decoded frame). Loops short videos."""
    targets = start + np.arange(n_frames) / fps
    best, times = _sample_stream(path, targets, targets[-1] + 0.5)
    if not len(times):
        raise ValueError(f"no frames decoded from {path}")
    if start > times[-1]:  # start beyond end: fall back to beginning
        return read_frames(path, n_frames, fps, 0.0)
    dt = times[1] - times[0] if len(times) > 1 else 1 / fps
    if targets[-1] > times[-1] + 0.5 * dt:  # video shorter than the window -> loop over decoded frames
        span = times[-1] + dt
        looped = start + np.mod(targets - start, span)
        best, _ = _sample_stream(path, looped, times[-1] + 0.5)
    return np.stack(best)


def read_all_frames(path, max_seconds: float | None = None) -> tuple[np.ndarray, float | None]:
    frames, _, rate = _decode_all(path, max_seconds)
    return np.stack(frames), rate


def write_video(path, frames: np.ndarray, fps: int, crf: int = 18) -> None:
    """H.264 yuv420p mp4. Pads to even dims."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w = frames.shape[:3]
    h2, w2 = h + h % 2, w + w % 2
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=int(round(fps)))
        s.width, s.height, s.pix_fmt = w2, h2, "yuv420p"
        s.options = {"crf": str(crf), "preset": "medium"}
        for fr in frames:
            if (h2, w2) != (h, w):
                fr = np.pad(fr, ((0, h2 - h), (0, w2 - w), (0, 0)), mode="edge")
            c.mux(s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")))
        c.mux(s.encode())


def resize_short(frames: np.ndarray, short: int) -> np.ndarray:
    h, w = frames.shape[1:3]
    scale = short / min(h, w)
    nh, nw = max(short, round(h * scale)), max(short, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, (nw, nh), interpolation=interp) for f in frames])


def resize_crop(frames: np.ndarray, size: int = 256) -> np.ndarray:
    frames = resize_short(frames, size)
    h, w = frames.shape[1:3]
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(frames[:, y:y + size, x:x + size])


def h264_roundtrip(frames: np.ndarray, fps: int, crf: int, tmp_dir=None) -> np.ndarray:
    """Encode + decode through H.264 in memory (via temp file)."""
    import tempfile
    with tempfile.TemporaryDirectory(dir=tmp_dir) as d:
        p = Path(d) / "rt.mp4"
        write_video(p, frames, fps, crf)
        out, _ = read_all_frames(p)
    h, w = frames.shape[1:3]
    return out[: len(frames), :h, :w]
