import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.classic import fit_model
from src.models.cnn import SpecTempCNN, cnn_predict, train_cnn


def _data(n=300, d=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += 2.0 * y
    X[:, 1] -= 1.5 * y
    return X, y


@pytest.mark.parametrize("name", ["lr", "svm", "lgbm", "mlp"])
def test_classic_models_learn(name):
    X, y = _data()
    Xv, yv = _data(seed=1)
    m = fit_model(name, X, y, Xv, yv)
    assert roc_auc_score(yv, m.predict_proba(Xv)[:, 1]) > 0.9


def test_cnn_shapes_and_learns(tmp_path):
    import torch
    m = SpecTempCNN()
    assert m(torch.zeros(4, 2, 16, 48)).shape == (4,)
    assert sum(p.numel() for p in m.parameters()) < 200_000
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    X = rng.normal(size=(200, 2, 16, 48)).astype(np.float32)
    X[:, 0, :, 30:] -= 1.0 * y[:, None, None]       # high-frequency deficit for "fake"
    model, stats, hist = train_cnn(X[:150], y[:150], X[150:], y[150:], epochs=15, device="cpu")
    p = cnn_predict(model, stats, X[150:])
    assert roc_auc_score(y[150:], p) > 0.9
    torch.save(model.state_dict(), tmp_path / "m.pt")
    m2 = SpecTempCNN()
    m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    assert np.allclose(cnn_predict(m2, stats, X[150:]), p, atol=1e-6)
