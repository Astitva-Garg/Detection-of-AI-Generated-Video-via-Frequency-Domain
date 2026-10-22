"""Seed stability: retrain the main models with several random seeds and report mean ± std AUROC.

Single training runs of neural models (MLP, CNN) vary with initialisation and early stopping; this
quantifies that variance on the seen test set, each unseen generator and the in-the-wild clips.
"""
import argparse

import numpy as np
import pandas as pd
import torch

import src.models.classic as C
import src.models.cnn as K
from src.config import FEATURES, RESULTS, UNSEEN
from src.eval.evaluate import metrics
from src.models import data as D
from src.models.cnn import cnn_predict, train_cnn


def _sets(df):
    te = df[(df.split == "test") & ~df.aug.astype(bool)]
    real = te[te.generator == "real"]
    out = {"seen": te[~te.generator.isin(UNSEEN)]}
    for g in UNSEEN:
        out[g] = pd.concat([real, te[te.generator == g]])
    return out


def main(seeds: list[int]) -> pd.DataFrame:
    df = D.clean_table()
    tr, va = D.select(df, "train", include_aug=True), D.select(df, "val")
    sets = _sets(df)
    wild = pd.read_parquet(FEATURES / "wild.parquet").set_index("key")
    wseq = np.load(FEATURES / "seq_wild.npz")
    wreal = wild[wild.source == "real"]
    rows = []
    for seed in seeds:
        C.SEED = K.SEED = seed          # both modules read SEED at call time
        torch.manual_seed(seed)
        models = {}
        for name, group in (("mlp", "fusion"), ("mlp", "all"), ("mlp", "pixel")):
            cols = D.feature_cols(df, group)
            m = C.fit_model(name, *D.xy(tr, cols), *D.xy(va, cols))
            models[f"{name}_{group}"] = ("feat", m, cols)
        Xtr, Xva = D.load_seq("clean", tr.clip_id), D.load_seq("clean", va.clip_id)
        cm, st, _ = train_cnn(Xtr, tr.label.to_numpy(), Xva, va.label.to_numpy())
        models["cnn"] = ("cnn", cm, st)

        def score(kind, m, extra, frame, keys=None, seqname="clean"):
            if kind == "cnn":
                X = (np.stack([wseq[k] for k in keys]) if seqname == "wild"
                     else D.load_seq(seqname, frame.clip_id))
                return cnn_predict(m, extra, X)
            X = np.nan_to_num(frame[extra].to_numpy(np.float32))
            return m.predict_proba(X)[:, 1]

        for mn, (kind, m, extra) in models.items():
            r = {"seed": seed, "model": mn}
            for sname, sub in sets.items():
                r[sname] = metrics(sub.label, score(kind, m, extra, sub))["auroc"]
            for src in sorted(s for s in wild.source.unique() if s != "real"):
                sub = pd.concat([wild[wild.source == src], wreal])
                p = score(kind, m, extra, sub, keys=list(sub.index), seqname="wild")
                r[f"wild_{src}"] = metrics(sub.label, p)["auroc"]
            rows.append(r)
            print(seed, mn, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    res = pd.DataFrame(rows)
    wcols = [c for c in res.columns if c.startswith("wild_")]
    res["wild_mean"] = res[wcols].mean(axis=1)
    res.to_csv(RESULTS / "seed_stability_runs.csv", index=False)
    summ = res.drop(columns="seed").groupby("model").agg(["mean", "std"]).round(3)
    summ.to_csv(RESULTS / "seed_stability.csv")
    print(summ.to_string())
    return summ


def plot() -> None:
    """Bar chart of mean ± std AUROC over seeds (unseen generators and in-the-wild sources)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = pd.read_csv(RESULTS / "seed_stability_runs.csv")
    cols = ["seen", "modelscope", "wan", "wild_sora2", "wild_veo3", "wild_hunyuan", "wild_kling", "wild_wan2"]
    labels = ["seen\ntest", "ModelScope\n(unseen)", "Wan2.1\n(unseen)", "Sora 2", "Veo 3", "Hunyuan", "Kling", "Wan2"]
    order = [("mlp_fusion", "MLP freq+pixel"), ("mlp_all", "MLP frequency"), ("cnn", "Spectro-temporal CNN"),
             ("mlp_pixel", "MLP pixel")]
    g = r.groupby("model")[cols]
    m, s = g.mean(), g.std()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    w = 0.2
    for i, (k, lab) in enumerate(order):
        ax.bar(np.arange(len(cols)) + (i - 1.5) * w, m.loc[k], w, yerr=s.loc[k], capsize=2, label=lab)
    ax.axvline(2.5, color="k", lw=0.8)
    ax.text(5.5, 1.005, "in the wild (never seen, different content)", ha="center", fontsize=9)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(len(cols)), labels)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("AUROC (mean ± std, 5 seeds)")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "seed_stability.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[643, 1, 2, 3, 4])
    main(ap.parse_args().seeds)
    plot()
