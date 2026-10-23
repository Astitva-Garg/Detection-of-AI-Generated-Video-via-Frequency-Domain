import time

import joblib
import numpy as np
import pytest

from src.data.video_io import write_video
from src.features.temporal import clip_features, group_columns
from src.models.classic import make_model
from src.predict import Predictor


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        feats, _ = clip_features(rng.integers(0, 255, (16, 64, 64, 3), dtype=np.uint8))
        rows.append(feats)
    cols = group_columns(list(rows[0]), "all")
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float32)
    y = np.array([0, 1] * 6)
    m = make_model("lr").fit(X, y)
    joblib.dump({"model": m, "cols": cols}, d / "lr_all.joblib")
    return d


def _video(path, n, h, w, fps=24):
    write_video(path, np.random.default_rng(1).integers(0, 255, (n, h, w, 3), dtype=np.uint8), fps)


def test_predict_normal_clip(tmp_path, model_dir):
    p = tmp_path / "a.mp4"
    _video(p, 60, 360, 640)
    t = time.time()
    r = Predictor(model_dir).predict(p, model="lr_all", check_content=False)
    assert 0 <= r.p_fake <= 1 and time.time() - t < 10
    assert r.frames.shape == (14, 256, 256, 3) and r.top_features


def test_predict_short_clip_note(tmp_path, model_dir):
    p = tmp_path / "short.mp4"
    _video(p, 12, 256, 256)            # 0.5 s
    r = Predictor(model_dir).predict(p, model="lr_all", check_content=False)
    assert 0 <= r.p_fake <= 1 and any("looped" in n for n in r.notes)


def test_predict_tiny_clip(tmp_path, model_dir):
    p = tmp_path / "tiny.mp4"
    _video(p, 30, 60, 100)
    r = Predictor(model_dir).predict(p, model="lr_all", check_content=False, degradation="h264_crf38")
    assert 0 <= r.p_fake <= 1 and any("Low resolution" in n for n in r.notes)
