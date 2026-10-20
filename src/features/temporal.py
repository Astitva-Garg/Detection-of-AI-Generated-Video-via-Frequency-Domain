"""Temporal frequency-drift features and full per-clip feature extraction."""
import numpy as np

from src.config import N_BINS
from src.features.spatial import EPS, frame_features, radial_spectrum, residual, to_luma

DRIFT_SCALARS = ["slope", "hf_ratio_50", "centroid", "dct_high"]
FEATURE_GROUPS = {"spatial": ["sp_"], "temporal": ["drift_", "rt_"]}


def _stats(x: np.ndarray, name: str) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        x = np.zeros(1)
    sd = float(x.std())
    ac1 = 0.0
    if x.size > 2 and sd > 1e-12:
        a, b = x[:-1] - x.mean(), x[1:] - x.mean()
        ac1 = float((a * b).mean() / (x.var() + 1e-12))
    return {f"{name}_mean": float(x.mean()), f"{name}_std": sd, f"{name}_max": float(x.max()),
            f"{name}_ac1": ac1}


def drift_features(spectra: np.ndarray, scalars: dict[str, np.ndarray]) -> dict[str, float]:
    """spectra: (T, n_bins) per-frame radial log-spectra; scalars: per-frame series."""
    spectra = np.asarray(spectra, dtype=np.float64)
    d = np.diff(spectra, axis=0)
    l2 = np.linalg.norm(d, axis=1)
    na = np.linalg.norm(spectra, axis=1)
    cos = 1.0 - (spectra[1:] * spectra[:-1]).sum(1) / (na[1:] * na[:-1] + EPS)
    cos = np.where((na[1:] < 1e-9) | (na[:-1] < 1e-9), 0.0, cos)
    out = {}
    out.update(_stats(l2, "drift_l2"))
    out.update(_stats(cos, "drift_cos"))
    acc = np.abs(np.diff(l2))
    out["drift_l2_acc_mean"] = float(acc.mean()) if acc.size else 0.0
    out["drift_l2_acc_std"] = float(acc.std()) if acc.size else 0.0
    n = spectra.shape[1]
    for name, sl in (("low", slice(0, n // 3)), ("mid", slice(n // 3, 2 * n // 3)), ("high", slice(2 * n // 3, n))):
        b = np.linalg.norm(d[:, sl], axis=1)
        out[f"drift_band_{name}_mean"] = float(b.mean())
        out[f"drift_band_{name}_std"] = float(b.std())
    for k, s in scalars.items():
        s = np.asarray(s, dtype=np.float64)
        ds = np.abs(np.diff(s))
        out[f"drift_{k}_mean"] = float(ds.mean())
        out[f"drift_{k}_std"] = float(ds.std())
        out[f"drift_{k}_max"] = float(ds.max())
        dd = np.abs(np.diff(np.diff(s)))
        out[f"drift_{k}_acc"] = float(dd.mean()) if dd.size else 0.0
    return out


def residual_temporal(gray: np.ndarray) -> dict[str, float]:
    """gray: (T,H,W). Temporal behaviour of the per-pixel noise residual.

    Real sensor noise is (nearly) independent frame to frame; generated video tends to have
    temporally correlated / smoothed fine detail.
    """
    res = np.stack([residual(g) for g in gray]).astype(np.float64)   # (T,H,W)
    res = res - res.mean(axis=0, keepdims=True)
    a, b = res[:-1], res[1:]
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-12
    corr = num / den
    valid = (a * a).sum(axis=0) > 1e-10
    lag1 = float(corr[valid].mean()) if valid.any() else 0.0
    spec = np.abs(np.fft.rfft(res, axis=0)) ** 2      # (T//2+1, H, W)
    spec = spec[1:].sum(axis=(1, 2))                   # drop temporal DC
    half = len(spec) // 2
    hi = float(spec[half:].sum() / (spec.sum() + EPS)) if spec.sum() > 0 else 0.0
    energy = float((res ** 2).mean())
    # spatial-temporal: drift of residual radial spectrum
    rr = np.stack([radial_spectrum(r.astype(np.float32)) for r in res])
    rdrift = np.linalg.norm(np.diff(rr, axis=0), axis=1)
    return {"rt_lag1_corr": lag1, "rt_hi_temporal_frac": hi, "rt_energy": float(np.log(energy + 1e-12)),
            "rt_res_drift_mean": float(rdrift.mean()), "rt_res_drift_std": float(rdrift.std())}


def clip_features(frames_rgb: np.ndarray, n_bins: int = N_BINS) -> tuple[dict[str, float], np.ndarray]:
    """frames_rgb: (T,H,W,3) uint8 normalised clip -> (feature dict, seq (2,T,n_bins))."""
    gray = np.stack([to_luma(f) for f in frames_rgb])
    per = [frame_features(g, n_bins) for g in gray]
    keys = list(per[0])
    M = np.array([[p[k] for k in keys] for p in per])          # (T, F)
    feats = {}
    for j, k in enumerate(keys):
        feats[f"sp_{k}_mean"] = float(M[:, j].mean())
        feats[f"sp_{k}_std"] = float(M[:, j].std())
    rad = M[:, [keys.index(f"rad_{i:02d}") for i in range(n_bins)]]
    feats.update(drift_features(rad, {k: M[:, keys.index(k)] for k in DRIFT_SCALARS}))
    feats.update(residual_temporal(gray))
    diff_spec = np.zeros_like(rad)
    for t in range(1, len(gray)):
        diff_spec[t] = radial_spectrum(gray[t] - gray[t - 1], n_bins)
    seq = np.stack([rad, diff_spec]).astype(np.float32)
    return feats, seq


def group_columns(columns, group: str) -> list[str]:
    if group == "all":
        prefixes = sum(FEATURE_GROUPS.values(), [])
    else:
        prefixes = FEATURE_GROUPS[group]
    return [c for c in columns if any(c.startswith(p) for p in prefixes)]
