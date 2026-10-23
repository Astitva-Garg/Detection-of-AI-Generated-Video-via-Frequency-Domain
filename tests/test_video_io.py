import av
import numpy as np

from src.data.normalize import normalize_clip
from src.data.video_io import read_all_frames, read_frames, write_video


def _clip(n, h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (n, h, w, 3), dtype=np.uint8)


def test_read_frames_shape(tmp_path):
    p = tmp_path / "a.mp4"
    write_video(p, _clip(40, 360, 640), fps=24)
    assert read_frames(p, 16, 8).shape == (16, 360, 640, 3)


def test_short_clip_loops(tmp_path):
    p = tmp_path / "short.mp4"
    write_video(p, _clip(5, 64, 64), fps=24)
    assert read_frames(p, 16, 8).shape == (16, 64, 64, 3)


def test_tiny_portrait_normalised(tmp_path):
    p, q = tmp_path / "tiny.mp4", tmp_path / "norm.mp4"
    write_video(p, _clip(20, 100, 60), fps=10)
    meta = normalize_clip(p, q)
    out, _ = read_all_frames(q)
    assert out.shape == (14, 256, 256, 3)
    assert meta["orig_w"] == 60 and meta["orig_h"] == 100


def test_missing_fps_metadata(tmp_path):
    # mpeg4 in .avi with no explicit rate info beyond the codec default
    p = tmp_path / "odd.avi"
    with av.open(str(p), "w") as c:
        s = c.add_stream("mpeg4")
        s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
        for fr in _clip(12, 64, 64):
            c.mux(s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")))
        c.mux(s.encode())
    assert read_frames(p, 16, 8).shape == (16, 64, 64, 3)


def test_pre_downscale(tmp_path):
    p, q = tmp_path / "big.mp4", tmp_path / "n.mp4"
    write_video(p, _clip(24, 720, 1280), fps=24)
    normalize_clip(p, q, pre_short_side=320)
    assert read_all_frames(q)[0].shape == (14, 256, 256, 3)


def _numbered(n, h=32, w=32):
    """Each frame filled with its own index, so sampled indices can be read back."""
    return np.stack([np.full((h, w, 3), i * 5, np.uint8) for i in range(n)])  # n <= 51


def _ids(frames):
    return [int(round(f.mean() / 5)) for f in frames]


def test_stride_svd_like_no_repeats(tmp_path):
    from src.data.video_io import read_frames_stride
    p = tmp_path / "svd.mp4"
    write_video(p, _numbered(14), fps=7, crf=0)
    fr, eff = read_frames_stride(p, 14, 8)
    assert _ids(fr) == list(range(14)) and abs(eff - 7) < 1e-6


def test_stride_24fps_every_third(tmp_path):
    from src.data.video_io import read_frames_stride
    p = tmp_path / "ltx.mp4"
    write_video(p, _numbered(49), fps=24, crf=0)
    fr, eff = read_frames_stride(p, 14, 8)
    assert _ids(fr) == [3 * i for i in range(14)] and eff == 8


def test_stride_start_offset_and_short_loop(tmp_path):
    from src.data.video_io import read_frames_stride
    p = tmp_path / "r.mp4"
    write_video(p, _numbered(50), fps=30, crf=0)
    fr, _ = read_frames_stride(p, 14, 8, start=0.5)          # k=4, first frame 15
    assert _ids(fr)[:3] == [15, 19, 23]
    q = tmp_path / "short.mp4"
    write_video(q, _numbered(5), fps=24, crf=0)
    assert read_frames_stride(q, 14, 8)[0].shape[0] == 14
