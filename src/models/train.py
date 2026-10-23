"""Train all models x feature groups, save them, report validation metrics.

Models: lr, svm, lgbm, mlp on {spatial, temporal, all}; spectro-temporal CNN on the spectrum sequence;
lgbm_all_robust trained on clean + degraded train clips (degradation-aware variant).
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import MODELS, RESULTS
from src.models import data as D
from src.models.classic import fit_model
from src.models.cnn import cnn_predict, train_cnn

CLASSIC = ["lr", "svm", "lgbm", "mlp"]
GROUPS = ["spatial", "temporal", "all"]


def _metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "acc": float(accuracy_score(y, p >= 0.5))}


def train_classic(df, name, group, train_df=None):
    cols = D.feature_cols(df, group)
    tr = train_df if train_df is not None else D.select(df, "train", include_aug=True)
    va = D.select(df, "val")
    m = fit_model(name, *D.xy(tr, cols), *D.xy(va, cols))
    Xv, yv = D.xy(va, cols)
    return m, cols, _metrics(yv, m.predict_proba(Xv)[:, 1])


def main(skip_cnn: bool = False, generators: list[str] | None = None, tag: str = "") -> dict:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df = D.clean_table()
    if generators:
        df = df[df.generator.isin(generators + ["real"])]
    rows = []
    groups = GROUPS + (["pixel", "fusion"] if any(c.startswith("px_") for c in df.columns) else [])
    for name in CLASSIC:
        for group in groups:
            m, cols, met = train_classic(df, name, group)
            joblib.dump({"model": m, "cols": cols}, MODELS / f"{name}_{group}{tag}.joblib")
            rows.append({"model": f"{name}_{group}{tag}", **met})
            print(rows[-1], flush=True)

    # degradation-aware variant
    try:
        deg = D.degraded_table()
        deg_tr = deg[(deg.split == "train") & ~deg.generator.isin(D.UNSEEN)]
        if generators:
            deg_tr = deg_tr[deg_tr.generator.isin(generators + ["real"])]
        tr = pd.concat([D.select(df, "train", include_aug=True), deg_tr], ignore_index=True)
        m, cols, met = train_classic(df, "lgbm", "all", train_df=tr)
        joblib.dump({"model": m, "cols": cols}, MODELS / f"lgbm_all_robust{tag}.joblib")
        rows.append({"model": f"lgbm_all_robust{tag}", **met})
        print(rows[-1], flush=True)
    except FileNotFoundError:
        print("no degraded features yet; skipping robust model")

    if not skip_cnn:
        tr, va = D.select(df, "train", include_aug=True), D.select(df, "val")
        Xtr, Xva = D.load_seq("clean", tr.clip_id), D.load_seq("clean", va.clip_id)
        model, stats, hist = train_cnn(Xtr, tr.label.to_numpy(), Xva, va.label.to_numpy())
        torch.save({"state": model.state_dict(), "mu": stats[0], "sd": stats[1],
                    "in_ch": Xtr.shape[1], "n_bins": Xtr.shape[3]}, MODELS / f"cnn{tag}.pt")
        pd.DataFrame(hist).to_csv(RESULTS / f"cnn_history{tag}.csv", index=False)
        rows.append({"model": f"cnn{tag}", **_metrics(va.label.to_numpy(), cnn_predict(model, stats, Xva))})
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"val_metrics{tag}.csv", index=False)
    return {"val": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()
    main(a.skip_cnn)
