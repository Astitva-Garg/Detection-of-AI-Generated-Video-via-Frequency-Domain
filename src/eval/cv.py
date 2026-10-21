"""Grouped k-fold cross-validation over all seen-generator data (more stable than one small test split).

Folds are formed by group_id (source video / prompt), so no group appears in both train and test.
Inside each training fold, 15 % of groups are held out as the validation set for model selection.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import RESULTS, SEED, UNSEEN
from src.data.splits import assign_split
from src.eval.evaluate import metrics
from src.models import data as D
from src.models.classic import fit_model

MODELS = ["lr", "lgbm", "mlp"]
GROUPS = ["pixel", "spatial", "temporal", "all"]


def run(k: int = 5) -> pd.DataFrame:
    df = D.clean_table()
    df = df[~df.generator.isin(UNSEEN)].reset_index(drop=True)
    groups = df.group_id.to_numpy()
    rows = []
    gkf = GroupKFold(n_splits=k)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, df.label, groups)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        te = te[~te.aug.astype(bool)]
        inner = tr.group_id.map(lambda g: assign_split(g, seed=SEED + fold) == "test")  # ~15 % of groups
        fit, val = tr[~inner], tr[inner & ~tr.aug.astype(bool)]
        for group in GROUPS:
            cols = D.feature_cols(df, group)
            if not cols:
                continue
            for name in MODELS:
                m = fit_model(name, *D.xy(fit, cols), *D.xy(val, cols))
                p = m.predict_proba(D.xy(te, cols)[0])[:, 1]
                r = {"fold": fold, "model": name, "group": group, **metrics(te.label, p)}
                for gen in te.generator.unique():
                    if gen == "real":
                        continue
                    sub = te.generator.isin([gen, "real"]).to_numpy()
                    r[f"auroc_{gen}"] = metrics(te.label[sub], p[sub])["auroc"]
                rows.append(r)
                print(fold, name, group, round(r["auroc"], 3), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "cv_folds.csv", index=False)
    agg_cols = ["auroc", "acc", "f1"] + [c for c in res.columns if c.startswith("auroc_")]
    summary = res.groupby(["model", "group"])[agg_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(RESULTS / "cv_summary.csv")
    print(summary.to_string())
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    run(ap.parse_args().k)
