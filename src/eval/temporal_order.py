"""Does the detector use real temporal structure, or only per-frame spectra?

The held-out test clips are fed to the shipped models three ways, features recomputed from the frames each time:
  original   the clip as it is (sanity check: must equal the stored scores);
  reversed   frames in reverse order (time-asymmetric cues change, per-frame cues do not);
  shuffled   frames in random order (destroys temporal coherence; drift features become those of unrelated frames).

Spatial features are means and standard deviations over frames, so they cannot change under reversal or shuffling:
that is a built-in check. Any change in AUROC for the temporal or combined models is therefore due to the
frame-to-frame (drift) features alone.

    python -m src.eval.temporal_order --workers 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLIPS, FPS, MODELS, N_FRAMES, RESULTS, SEED, UNSEEN
from src.data.video_io import read_frames
from src.eval.resize_ablation import paired_ci
from src.features.temporal import clip_features
from src.models import data as D

MODEL_NAMES = ["mlp_all", "mlp_spatial", "mlp_temporal", "lgbm_all", "lgbm_spatial", "lgbm_temporal"]
VARIANTS = ["original", "reversed", "shuffled"]


def _one(args):
    cid, variant, seed = args
    fr = read_frames(CLIPS / f"{cid}.mp4", N_FRAMES, FPS)
    if variant == "reversed":
        fr = fr[::-1]
    elif variant == "shuffled":
        fr = fr[np.random.default_rng(seed).permutation(len(fr))]
    feats, _ = clip_features(np.ascontiguousarray(fr))
    return cid, variant, feats


def main(workers: int) -> pd.DataFrame:
    df = D.clean_table()
    te = df[(df.split == "test") & ~df.aug.astype(bool)].reset_index(drop=True)
    jobs = [(c, v, SEED + i) for i, c in enumerate(te.clip_id) for v in VARIANTS]
    feats = {v: {} for v in VARIANTS}
    with ProcessPoolExecutor(workers) as ex:
        for n, (cid, v, f) in enumerate(ex.map(_one, jobs, chunksize=6), 1):
            feats[v][cid] = f
            if n % 150 == 0:
                print(f"  {n}/{len(jobs)} feature sets", flush=True)
    real = te.generator == "real"
    sets = {"seen": ~te.generator.isin(UNSEEN), "ModelScope (unseen)": real | (te.generator == "modelscope"),
            "Wan2.1 (unseen)": real | (te.generator == "wan")}
    y = te.label.to_numpy().astype(int)
    rows, rng = [], np.random.default_rng(SEED)
    for name in MODEL_NAMES:
        entry = joblib.load(MODELS / f"{name}.joblib")
        model, cols = entry["model"], entry["cols"]
        p = {}
        for v in VARIANTS:
            X = pd.DataFrame([feats[v][c] for c in te.clip_id])[cols].to_numpy(np.float32)
            p[v] = model.predict_proba(np.nan_to_num(X))[:, 1]
        for sname, mask in sets.items():
            mk = mask.to_numpy()
            base = roc_auc_score(y[mk], p["original"][mk])
            for v in VARIANTS[1:]:
                a = roc_auc_score(y[mk], p[v][mk])
                lo, hi = paired_ci(y[mk], p[v][mk], p["original"][mk], rng)
                rows.append({"model": name, "set": sname, "variant": v, "auroc_original": base, "auroc_variant": a,
                             "change": a - base, "ci_low": lo, "ci_high": hi, "n": int(mk.sum())})
    res = pd.DataFrame(rows)
    res.round(3).to_csv(RESULTS / "temporal_order.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    main(ap.parse_args().workers)
