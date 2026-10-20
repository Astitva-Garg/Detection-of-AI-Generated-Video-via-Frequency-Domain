import numpy as np

from src.features.temporal import clip_features, drift_features, group_columns, residual_temporal


def test_identical_frames_zero_drift():
    spec = np.tile(np.linspace(1, -1, 48), (16, 1))
    f = drift_features(spec, {"slope": np.ones(16)})
    drift = {k: v for k, v in f.items()}
    assert all(np.isfinite(v) for v in drift.values())
    assert f["drift_l2_mean"] == 0 and f["drift_l2_ac1"] == 0 and f["drift_slope_mean"] == 0


def test_residual_temporal_independent_vs_static():
    rng = np.random.default_rng(0)
    indep = rng.random((16, 64, 64)).astype(np.float32)
    static = np.repeat(rng.random((1, 64, 64)).astype(np.float32), 16, 0)
    static = static + rng.normal(0, 1e-3, static.shape).astype(np.float32)  # tiny jitter
    assert abs(residual_temporal(indep)["rt_lag1_corr"]) < 0.1
    # static frames: mean-removed residual is just jitter -> uncorrelated, but energy far lower
    assert residual_temporal(static)["rt_energy"] < residual_temporal(indep)["rt_energy"]


def test_residual_temporal_slow_drift_correlated():
    rng = np.random.default_rng(1)
    base = rng.random((64, 64)).astype(np.float32)
    ramp = np.stack([base * (0.5 + 0.03 * t) for t in range(16)])
    assert residual_temporal(ramp)["rt_lag1_corr"] > 0.7  # linear ramp over 16 frames: ~0.81 after mean removal


def test_clip_features_shapes_and_finite():
    frames = np.random.default_rng(0).integers(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    feats, seq = clip_features(frames)
    assert seq.shape == (2, 16, 48)
    assert all(np.isfinite(v) for v in feats.values()) and np.isfinite(seq).all()
    cols = list(feats)
    sp, tm = group_columns(cols, "spatial"), group_columns(cols, "temporal")
    assert sp and tm and set(sp).isdisjoint(tm) and len(group_columns(cols, "all")) == len(sp) + len(tm)


def test_constant_clip_finite():
    frames = np.full((16, 256, 256, 3), 128, np.uint8)
    feats, seq = clip_features(frames)
    assert all(np.isfinite(v) for v in feats.values()) and np.isfinite(seq).all()
