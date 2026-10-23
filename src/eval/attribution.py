"""Generator attribution: which source produced a clip (real or one of the five generators)?

6-class grouped 5-fold CV over ALL clips (unseen generators included here, since this asks whether each
generator leaves a distinguishable fingerprint, not whether detection transfers). Folds by group_id.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import GENERATORS, RESULTS, SEED
from src.models import data as D

CLASSES = ["real"] + GENERATORS


def _model(name):
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5,
                              subsample=0.8, subsample_freq=1, class_weight="balanced", random_state=SEED,
                              verbose=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


def run(k: int = 5) -> pd.DataFrame:
    df = D.clean_table()
    df = df[~df.aug.astype(bool)].reset_index(drop=True)
    y = df.generator.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    rows, cms = [], {}
    for group in ["pixel", "spatial", "temporal", "all", "fusion"]:
        cols = D.feature_cols(df, group)
        X = D.xy(df, cols)[0]
        for name in ["lr", "lgbm"]:
            pred = np.empty_like(y)
            for tr, te in GroupKFold(k).split(X, y, df.group_id):
                pred[te] = _model(name).fit(X[tr], y[tr]).predict(X[te])
            cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
            per = cm.diagonal() / cm.sum(1)
            rows.append({"features": group, "model": name, "balanced_acc": balanced_accuracy_score(y, pred),
                         **{f"recall_{c}": r for c, r in zip(CLASSES, per)}})
            cms[(group, name)] = cm
            print(rows[-1]["features"], name, round(rows[-1]["balanced_acc"], 3), flush=True)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(RESULTS / "attribution.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, key in zip(ax, [("all", "lr"), ("fusion", "lr")]):
        cm = cms[key]
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmn):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.5 else "black")
        a.set_xticks(range(len(CLASSES)), CLASSES, rotation=35)
        a.set_yticks(range(len(CLASSES)), CLASSES)
        a.set_xlabel("predicted source")
        ba = res[(res.features == key[0]) & (res.model == key[1])].balanced_acc.iloc[0]
        a.set_title(f"{'frequency features' if key[0] == 'all' else 'frequency + pixel'} — balanced acc {ba:.2f}")
    ax[0].set_ylabel("true source")
    fig.suptitle("Generator attribution (6 classes, grouped 5-fold CV, logistic regression; chance = 0.17)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "attribution_confusion.png", dpi=200, bbox_inches="tight")
    print(res.to_string())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    run(ap.parse_args().k)
