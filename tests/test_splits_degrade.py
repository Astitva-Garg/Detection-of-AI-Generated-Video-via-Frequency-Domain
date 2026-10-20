import numpy as np

from src.data.degrade import DEGRADATIONS, apply_degradation
from src.data.splits import assign_split


def test_assign_split_deterministic_and_proportions():
    assert assign_split("prompt_001") == assign_split("prompt_001")
    s = [assign_split(f"id_{i}") for i in range(10_000)]
    for name, p in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(s.count(name) / len(s) - p) < 0.02


def test_degradations_shape_and_effect():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (16, 64, 64, 3), dtype=np.uint8)
    for name in DEGRADATIONS:
        out = apply_degradation(f, name)
        assert out.shape == f.shape and out.dtype == np.uint8, name
    assert np.abs(apply_degradation(f, "h264_crf38").astype(int) - f).mean() > 0
    const = np.full((16, 64, 64, 3), 100, np.uint8)
    assert (apply_degradation(const, "blur1.0") == const).all()
    assert np.isfinite(apply_degradation(const, "noise5")).all()
