"""Out-of-domain stress test on in-the-wild generated clips (Sora 2, Veo 3, Kling, HunyuanVideo, Wan2).

Every clip — the wild fakes and the real test clips used for comparison — goes through the deployment
path of the app (src.predict: first 2 s at 8 fps, large videos downscaled to 512 px short side, native
256 crop, H.264 round trip). Reports AUROC vs. real test clips and the detection rate at 0.5, for all
clips and for the scenery-only subset that passes the YOLO content filter.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import FPS, MODELS, N_FRAMES, RAW_REAL, RESULTS
from src.data.normalize import normalize_frames
from src.data.video_io import h264_roundtrip, read_frames, read_frames_stride
from src.data.wild import WILD
from src.eval.evaluate import metrics
from src.features.pixel_baseline import clip_pixel_features
from src.features.temporal import clip_features
from src.models import data as D
from src.models.cnn import SpecTempCNN, cnn_predict

MODEL_NAMES = ["mlp_fusion", "mlp_all", "lgbm_all", "mlp_pixel"]


def _features(args):
    key, path = args
    try:
        raw, _ = read_frames_stride(path, N_FRAMES, FPS)
        pre = 512 if min(raw.shape[1:3]) > 576 else None
        frames = h264_roundtrip(normalize_frames(raw, pre), FPS, crf=18)
        f, seq = clip_features(frames)
        f.update(clip_pixel_features(frames))
        from src.data.content_filter import max_blocked_area
        return key, f, seq, raw
    except Exception as e:
        print("fail", path, e, flush=True)
        return key, None, None, None


def main(workers: int = 3) -> None:
    man = D.manifest()
    real = man[(man.generator == "real") & (man.split == "test") & ~man.aug.astype(bool)]
    items = [(f"real|{c}", str(RAW_REAL / f"{c.replace('real_', '')}.mp4")) for c in real.clip_id]
    for d in sorted(p for p in WILD.iterdir() if p.is_dir()):
        items += [(f"{d.name}|{p.stem}", str(p)) for p in sorted(d.glob("*.mp4"))]
    print(len(items), "clips", flush=True)
    feats, seqs, firsts = {}, {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for key, f, seq, raw in ex.map(_features, items):
            if f is not None:
                feats[key], seqs[key] = f, seq
                firsts[key] = raw[np.linspace(0, len(raw) - 1, 4).round().astype(int)]
    from ultralytics import YOLO
    from src.data.content_filter import max_blocked_area
    yolo = YOLO("yolov8n.pt")
    df = pd.DataFrame.from_dict(feats, orient="index")
    df["source"] = [k.split("|")[0] for k in df.index]
    df["label"] = (df.source != "real").astype(int)
    df["scenery"] = [max_blocked_area(firsts[k], yolo)[0] <= 0.05 for k in df.index]
    # cache features so other analyses (seed stability) can reuse the deployment-path features
    from src.config import FEATURES
    df.reset_index(names="key").to_parquet(FEATURES / "wild.parquet", index=False)
    np.savez_compressed(FEATURES / "seq_wild.npz", **{k: seqs[k] for k in df.index})
    for name in MODEL_NAMES:
        ent = joblib.load(MODELS / f"{name}.joblib")
        X = np.nan_to_num(df[ent["cols"]].to_numpy(np.float32))
        df[name] = ent["model"].predict_proba(X)[:, 1]
    ck = torch.load(MODELS / "cnn.pt", weights_only=False)
    m = SpecTempCNN(ck["in_ch"], ck["n_bins"])
    m.load_state_dict(ck["state"])
    df["cnn"] = cnn_predict(m, (ck["mu"], ck["sd"]), np.stack([seqs[k] for k in df.index]))
    models = MODEL_NAMES + ["cnn"]
    rows = []
    realdf = df[df.source == "real"]
    for src in sorted(df.source.unique()):
        if src == "real":
            continue
        for subset in ("all", "scenery"):
            fk = df[(df.source == src) & (df.scenery if subset == "scenery" else True)]
            if len(fk) < 3:
                continue
            sub = pd.concat([fk, realdf])
            for mn in models:
                r = metrics(sub.label, sub[mn])
                rows.append({"source": src, "subset": subset, "model": mn, "n_fake": len(fk),
                             "auroc": r["auroc"], "detected_at_0.5": r["tpr_fake"]})
    for mn in models:
        rows.append({"source": "real (test, deployment path)", "subset": "all", "model": mn,
                     "n_fake": 0, "auroc": np.nan, "detected_at_0.5": float((realdf[mn] >= 0.5).mean())})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(RESULTS / "wild_eval.csv", index=False)
    df[["source", "label", "scenery"] + models].to_csv(RESULTS / "wild_predictions.csv")
    piv = res[res.subset == "all"].pivot(index="source", columns="model", values="auroc")
    print(piv.to_string())
    print(res[res.subset == "all"].pivot(index="source", columns="model", values="detected_at_0.5").to_string())

    fig, ax = plt.subplots(figsize=(10, 4.2))
    srcs = [s for s in piv.index if not s.startswith("real")]
    w = 0.8 / len(models)
    for i, mn in enumerate(models):
        ax.bar(np.arange(len(srcs)) + (i - (len(models) - 1) / 2) * w, piv.loc[srcs, mn], w, label=mn)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(len(srcs)), srcs)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC vs. real test clips")
    ax.set_title("Out-of-domain stress test: public clips from commercial / newer generators")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "wild_stress_test.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    main(ap.parse_args().workers)
