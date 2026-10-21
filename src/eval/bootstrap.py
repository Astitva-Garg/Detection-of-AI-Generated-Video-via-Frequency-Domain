"""Bootstrap 95 % confidence intervals for test AUROC and paired model differences.

Uses clip-level predictions from results/predictions.csv. Each resample draws real and fake clips with
replacement separately (stratified), so every resample has both classes. Paired differences reuse the
same resample for both models.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RESULTS, SEED, UNSEEN

MODELS = ["mlp_fusion", "mlp_all", "cnn", "lgbm_all", "mlp_pixel", "lgbm_pixel"]
PAIRS = [("mlp_fusion", "mlp_all"), ("mlp_fusion", "mlp_pixel"), ("mlp_all", "mlp_pixel"),
         ("mlp_all", "cnn")]


def _sets(pr: pd.DataFrame) -> dict[str, pd.Series]:
    first = pr[pr.model == pr.model.iloc[0]]
    gen = first.set_index("clip_id").generator
    real = gen.index[gen == "real"]
    out = {"seen": gen.index[~gen.isin(UNSEEN)]}
    for g in UNSEEN + ["svd"]:
        out[g] = real.append(gen.index[gen == g])
    out["unseen_pooled"] = real.append(gen.index[gen.isin(UNSEEN)])
    return out


def bootstrap_auc(y: np.ndarray, scores: dict[str, np.ndarray], n: int, rng) -> dict[str, np.ndarray]:
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    out = {k: np.empty(n) for k in scores}
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k][b] = roc_auc_score(y[idx], s[idx])
    return out


def main(n: int = 2000) -> pd.DataFrame:
    pr = pd.read_csv(RESULTS / "predictions.csv")
    wide = pr.pivot_table(index="clip_id", columns="model", values="p_fake")
    lab = pr.drop_duplicates("clip_id").set_index("clip_id").label
    rng = np.random.default_rng(SEED)
    rows, diffs = [], []
    for name, ids in _sets(pr).items():
        ids = [i for i in ids if i in wide.index]
        y = lab.loc[ids].to_numpy()
        sc = {m: wide.loc[ids, m].to_numpy() for m in MODELS if m in wide.columns}
        bs = bootstrap_auc(y, sc, n, rng)
        for m, s in sc.items():
            lo, hi = np.percentile(bs[m], [2.5, 97.5])
            rows.append({"set": name, "model": m, "n_fake": int(y.sum()), "n_real": int((1 - y).sum()),
                         "auroc": roc_auc_score(y, s), "ci_low": lo, "ci_high": hi})
        for a, b in PAIRS:
            if a in bs and b in bs:
                d = bs[a] - bs[b]
                diffs.append({"set": name, "a": a, "b": b, "diff": roc_auc_score(y, sc[a]) - roc_auc_score(y, sc[b]),
                              "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                              "p_a_not_better": float((d <= 0).mean())})
    res, dif = pd.DataFrame(rows).round(3), pd.DataFrame(diffs).round(3)
    res.to_csv(RESULTS / "bootstrap_auroc.csv", index=False)
    dif.to_csv(RESULTS / "bootstrap_diffs.csv", index=False)
    print(res.to_string())
    print(dif.to_string())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    main(ap.parse_args().n)
