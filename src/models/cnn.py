"""Spectro-temporal 1D-CNN over the per-frame radial-spectrum sequence.

Input (B, 2, T, F): channel 0 = radial log-spectrum of each frame, channel 1 = radial spectrum of the
frame difference. A shared frequency encoder (Conv1d over F) turns each frame into a vector; a temporal
Conv1d stack models how the spectrum drifts across frames.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import N_BINS, SEED


class SpecTempCNN(nn.Module):
    def __init__(self, in_ch: int = 2, n_bins: int = N_BINS):
        super().__init__()
        self.freq = nn.Sequential(
            nn.Conv1d(in_ch, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2, stride=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        d = 32 * 8
        self.time = nn.Sequential(
            nn.Conv1d(d, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 * 2, 1))

    def forward(self, x):                          # x: (B, C, T, F)
        b, c, t, f = x.shape
        z = self.freq(x.permute(0, 2, 1, 3).reshape(b * t, c, f))   # (B*T, 32, 8)
        z = z.reshape(b, t, -1).transpose(1, 2)                       # (B, 256, T)
        z = self.time(z)                                              # (B, 64, T)
        z = torch.cat([z.mean(-1), z.amax(-1)], dim=1)
        return self.head(z).squeeze(-1)


def standardise_stats(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = seq.mean(axis=(0, 2), keepdims=True)
    sd = seq.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu[0], sd[0]


def train_cnn(Xtr, ytr, Xva, yva, epochs=60, patience=10, lr=1e-3, wd=1e-4, batch=32, device=None):
    """Returns (model, (mu, sd), history). BCEWithLogits loss, AdamW, cosine LR, early stop on val AUROC."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    mu, sd = standardise_stats(Xtr)
    norm = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    Xt, Xv = norm(Xtr), norm(Xva).to(device)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = SpecTempCNN(Xtr.shape[1], Xtr.shape[3]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = max(ytr.mean(), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    best, best_state, bad, hist = -1.0, None, 0, []
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            if torch.rand(1, generator=g).item() < 0.5:   # time-reversal augmentation
                xb = xb.flip(2)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        hist.append({"epoch": ep, "train_loss": tot / len(Xt), "val_auc": float(auc)})
        if auc > best + 1e-4:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), (mu, sd), hist


@torch.no_grad()
def cnn_predict(model, stats, X) -> np.ndarray:
    mu, sd = stats
    model.eval()
    return torch.sigmoid(model(torch.tensor((X - mu) / sd, dtype=torch.float32))).numpy()
