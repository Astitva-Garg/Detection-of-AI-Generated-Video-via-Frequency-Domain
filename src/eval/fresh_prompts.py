"""Prompt-disjoint test of an unseen generator.

The main unseen test sets (ModelScope, Wan2.1) reuse the 60 prompts that the seen generators were trained on, so
those clips are content-matched to training fakes. Here ModelScope is run again on 36 *new* prompts
(`configs/fresh_prompts.yaml`, two seeds each; `python -m src.generate.modelscope --fresh`), and the shipped models
score those clips with no retraining and no tuning. Real comparison: the same 65 held-out real test clips as in every
other test set. The old-prompt ModelScope test clips are scored in the same run, so the two are directly comparable.

Clips go through the same normalisation as the main dataset (14 frames, 256 native crop, H.264 CRF 18) into
data/fresh/clips/, which is not part of the main dataset or of any training or selection.

    python -m src.eval.fresh_prompts [--workers 3]
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLIPS, DATA_ROOT, FPS, MODELS, N_FRAMES, RAW_FAKE, RESULTS, SEED
from src.data.normalize import normalize_clip
from src.data.video_io import read_frames
from src.features.pixel_baseline import clip_pixel_features
from src.features.temporal import clip_features
from src.eval.resize_ablation import paired_ci
from src.models import data as D

GEN = "modelscope_fresh"
OUT = DATA_ROOT / "fresh"
MODEL_NAMES = ["lr_pixel", "lgbm_pixel", "mlp_pixel", "lr_all", "lgbm_all", "mlp_all", "lr_fusion", "lgbm_fusion",
               "mlp_fusion"]


def _prepare(job):
    cid, src = job
    dst = OUT / "clips" / f"{cid}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        normalize_clip(src, dst, None, 0.0)
    fr = read_frames(dst, N_FRAMES, FPS)
    f, _ = clip_features(fr)
    f.update(clip_pixel_features(fr))
    return cid, f


def fresh_table(workers: int) -> pd.DataFrame:
    """Features of the prompt-disjoint clips. From the raw clips if present (normalises them and writes
    data/fresh/manifest.csv); otherwise from the normalised clips of the data bundle."""
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    raw = RAW_FAKE / GEN / "manifest.jsonl"
    man = OUT / "manifest.csv"
    if raw.exists():
        recs = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        recs = [r for r in recs if r["status"] == "ok"]
        jobs = [(r["id"], str(RAW_FAKE / GEN / f"{r['id']}.mp4")) for r in recs]
        pd.DataFrame([{"clip_id": r["id"], "prompt": r["prompt"], "scene": r["scene"], "seed": r["seed"]}
                      for r in recs]).to_csv(man, index=False)
    else:
        jobs = [(c, "") for c in pd.read_csv(man).clip_id]
    with ProcessPoolExecutor(workers) as ex:
        rows = [{"clip_id": c, **f} for c, f in ex.map(_prepare, jobs, chunksize=2)]
    out = pd.DataFrame(rows)
    out["label"] = 1
    out["generator"] = GEN
    return out


def main(workers: int) -> pd.DataFrame:
    fresh = fresh_table(workers)
    df = D.clean_table()
    te = df[(df.split == "test") & ~df.aug.astype(bool)]
    real, old = te[te.generator == "real"], te[te.generator == "modelscope"]
    print(f"fresh-prompt ModelScope clips: {len(fresh)};  real test clips: {len(real)};  old-prompt ModelScope: {len(old)}")
    rows, rng = [], np.random.default_rng(SEED)
    scored = {}
    for name in MODEL_NAMES:
        entry = joblib.load(MODELS / f"{name}.joblib")
        model, cols = entry["model"], entry["cols"]

        def score(t):
            return model.predict_proba(np.nan_to_num(t[cols].to_numpy(np.float32)))[:, 1]
        p_real, p_old, p_new = score(real), score(old), score(fresh)
        scored[name] = (np.r_[p_real, p_old], np.r_[p_real, p_new])
        y_old = np.r_[np.zeros(len(real)), np.ones(len(old))]
        y_new = np.r_[np.zeros(len(real)), np.ones(len(fresh))]
        a_old, a_new = roc_auc_score(y_old, np.r_[p_real, p_old]), roc_auc_score(y_new, np.r_[p_real, p_new])
        rows.append({"model": name, "auroc_old_prompts": a_old, "auroc_new_prompts": a_new, "change": a_new - a_old,
                     "detect_old@0.5": float((p_old >= 0.5).mean()), "detect_new@0.5": float((p_new >= 0.5).mean()),
                     "real_rejected@0.5": float((p_real < 0.5).mean()),
                     "n_real": len(real), "n_old": len(old), "n_new": len(fresh)})
    try:                                   # the fine-tuned ResNet-18 on the same clips (needs a CUDA GPU)
        import torch
        from src.models import pixel_cnn as P
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        _, frames, cidx, *_ = P.load_frames()
        net = P.resnet18_ft().cuda()
        net.load_state_dict(torch.load(MODELS / "resnet18.pt", map_location="cuda"))
        pr = P.predict_clips(net, frames, [cidx[c] for c in real.clip_id])
        po = P.predict_clips(net, frames, [cidx[c] for c in old.clip_id])
        fr = np.stack([read_frames(OUT / "clips" / f"{c}.mp4", N_FRAMES, FPS)[P.FRAME_IDX] for c in fresh.clip_id])
        pn = P.predict_clips(net, fr, list(range(len(fr))))
        y_old = np.r_[np.zeros(len(real)), np.ones(len(old))]
        y_new = np.r_[np.zeros(len(real)), np.ones(len(fresh))]
        a_old, a_new = roc_auc_score(y_old, np.r_[pr, po]), roc_auc_score(y_new, np.r_[pr, pn])
        scored["resnet18"] = (np.r_[pr, po], np.r_[pr, pn])
        rows.append({"model": "resnet18", "auroc_old_prompts": a_old, "auroc_new_prompts": a_new, "change": a_new - a_old,
                     "detect_old@0.5": float((po >= 0.5).mean()), "detect_new@0.5": float((pn >= 0.5).mean()),
                     "real_rejected@0.5": float((pr < 0.5).mean()), "n_real": len(real), "n_old": len(old), "n_new": len(fresh)})
    except Exception as e:
        print("ResNet-18 not scored:", e)
    res = pd.DataFrame(rows)
    res.round(3).to_csv(RESULTS / "fresh_prompts.csv", index=False)
    print(res.round(3).to_string(index=False))
    diffs = []
    for a, b in [("mlp_all", "mlp_pixel"), ("mlp_fusion", "mlp_pixel"), ("mlp_fusion", "mlp_all"),
                 ("lr_all", "lr_pixel"), ("lgbm_all", "lgbm_pixel"), ("resnet18", "mlp_all"), ("resnet18", "mlp_fusion")]:
        if a not in scored:
            continue
        for which, y, k in [("old prompts", y_old, 0), ("new prompts", y_new, 1)]:
            pa, pb = scored[a][k], scored[b][k]
            lo, hi = paired_ci(y.astype(int), pa, pb, rng)
            diffs.append({"set": which, "a": a, "b": b, "auroc_a": roc_auc_score(y, pa), "auroc_b": roc_auc_score(y, pb),
                          "a_minus_b": roc_auc_score(y, pa) - roc_auc_score(y, pb), "ci_low": lo, "ci_high": hi})
    dd = pd.DataFrame(diffs)
    dd.round(3).to_csv(RESULTS / "fresh_prompts_diffs.csv", index=False)
    print(); print(dd.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    main(ap.parse_args().workers)
