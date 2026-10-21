"""Pixel-domain baseline features (no frequency analysis) for comparison.

Per frame: colour histograms (HSV, 8 bins each), luminance mean/std/percentiles, gradient-magnitude
statistics (Sobel), Laplacian variance (sharpness), edge density (Canny), saturation stats.
Temporal: mean absolute frame difference (pixel-level motion) mean/std.
Prefix `px_`. Answers: do frequency features beat plain pixel statistics?
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.config import SUF, CLIPS, DATA_ROOT, FEATURES, FPS, N_FRAMES
from src.data.video_io import read_frames


def frame_pixel_features(rgb: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    f = {}
    for c, name, rng in ((0, "h", 180), (1, "s", 256), (2, "v", 256)):
        h = np.histogram(hsv[..., c], bins=8, range=(0, rng))[0].astype(np.float64)
        h /= h.sum() + 1e-9
        f.update({f"hist_{name}{i}": float(v) for i, v in enumerate(h)})
    f["lum_mean"], f["lum_std"] = float(gray.mean()), float(gray.std())
    for q in (5, 50, 95):
        f[f"lum_p{q}"] = float(np.percentile(gray, q))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx ** 2 + gy ** 2)
    f["grad_mean"], f["grad_std"], f["grad_p95"] = float(g.mean()), float(g.std()), float(np.percentile(g, 95))
    f["lap_var"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    f["edge_density"] = float((cv2.Canny((gray * 255).astype(np.uint8), 100, 200) > 0).mean())
    f["sat_mean"], f["sat_std"] = float(hsv[..., 1].mean() / 255), float(hsv[..., 1].std() / 255)
    return f


def clip_pixel_features(frames: np.ndarray) -> dict[str, float]:
    per = [frame_pixel_features(fr) for fr in frames]
    out = {}
    for k in per[0]:
        v = np.array([p[k] for p in per])
        out[f"px_{k}_mean"], out[f"px_{k}_std"] = float(v.mean()), float(v.std())
    d = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3)) / 255.0
    out["px_framediff_mean"], out["px_framediff_std"] = float(d.mean()), float(d.std())
    return out


def _one(args):
    key, path = args
    try:
        return {"key": key, **clip_pixel_features(read_frames(path, N_FRAMES, FPS))}
    except Exception as e:
        print("pixel fail", path, e, flush=True)
        return None


def main(workers: int) -> None:
    man = pd.read_csv(DATA_ROOT / f"manifest{SUF}.csv")
    out = FEATURES / "pixel.parquet"
    old = pd.read_parquet(out) if out.exists() else None
    have = set(old.key) if old is not None else set()
    items = [(c, str(CLIPS / f"{c}.mp4")) for c in man.clip_id if c not in have]
    deg_p = DATA_ROOT / f"manifest_degraded{SUF}.csv"
    if deg_p.exists():
        deg = pd.read_csv(deg_p)
        items += [(f"{r.clip_id}|{r.degradation}", r.path) for r in deg.itertuples()
                  if f"{r.clip_id}|{r.degradation}" not in have]
    print(f"pixel features: {len(have)} cached, {len(items)} to extract", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_one, items, chunksize=8) if r is not None]
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
