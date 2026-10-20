"""Feature-vector classifiers: logistic regression, linear SVM, LightGBM, MLP."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import SEED


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """2-hidden-layer MLP. BCEWithLogits loss, AdamW, cosine LR, early stopping on val AUROC."""

    def __init__(self, lr=1e-3, weight_decay=1e-4, epochs=200, patience=20, batch_size=64):
        self.lr, self.weight_decay, self.epochs, self.patience, self.batch_size = (
            lr, weight_decay, epochs, patience, batch_size)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(SEED)
        self.classes_ = np.array([0, 1])
        self.model_ = MLP(X.shape[1])
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        pos = max(y.mean(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        best, best_state, bad = -1.0, None, 0
        self.history_ = []
        g = torch.Generator().manual_seed(SEED)
        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            if X_val is None:
                continue
            auc = _auc(y_val, self.predict_proba(X_val)[:, 1])
            self.history_.append({"epoch": ep, "train_loss": tot / len(Xt), "val_auc": auc})
            if auc > best + 1e-4:
                best, best_state, bad = auc, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(name: str, C: float = 1.0):
    if name == "lr":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))])
    if name == "svm":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced",
                                                                  max_iter=20000), cv=3))])
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              class_weight="balanced", random_state=SEED, verbose=-1)
    if name == "mlp":
        return Pipeline([("sc", StandardScaler()), ("clf", TorchMLPClassifier())])
    raise ValueError(name)


def fit_model(name: str, Xtr, ytr, Xva, yva):
    """Fit with validation-based tuning: C grid for linear models, early stopping for lgbm/mlp."""
    if name in ("lr", "svm"):
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            m = make_model(name, C).fit(Xtr, ytr)
            auc = _auc(yva, m.predict_proba(Xva)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, m)
        return best[1]
    m = make_model(name)
    if name == "lgbm":
        from lightgbm import early_stopping
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[early_stopping(50, verbose=False)])
        return m
    sc = m.named_steps["sc"].fit(Xtr)
    m.named_steps["clf"].fit(sc.transform(Xtr), ytr, sc.transform(Xva), yva)
    return m
