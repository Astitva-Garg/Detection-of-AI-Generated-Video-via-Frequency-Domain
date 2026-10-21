import numpy as np

from src.features.pixel_baseline import clip_pixel_features


def test_pixel_features_finite_and_prefixed():
    rng = np.random.default_rng(0)
    f = clip_pixel_features(rng.integers(0, 255, (16, 64, 64, 3), dtype=np.uint8))
    assert all(k.startswith("px_") for k in f) and all(np.isfinite(v) for v in f.values())


def test_static_clip_zero_motion():
    frames = np.repeat(np.random.default_rng(1).integers(0, 255, (1, 64, 64, 3), dtype=np.uint8), 16, 0)
    f = clip_pixel_features(frames)
    assert f["px_framediff_mean"] == 0.0
    assert abs(sum(f[f"px_hist_h{i}_mean"] for i in range(8)) - 1.0) < 1e-6
