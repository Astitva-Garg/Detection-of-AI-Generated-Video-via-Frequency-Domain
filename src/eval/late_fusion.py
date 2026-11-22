"""Do the frequency features add anything to the strongest model (the fine-tuned ResNet-18)?

Late fusion: a 2-parameter logistic regression stacks the ResNet-18 score and the frequency-MLP score. The stacker is
fitted on the *validation* clips of the seen generators only (the test clips and the unseen generators play no part in
fitting anything), then applied to the held-out test sets. Compared on the same clips, with a paired bootstrap:

    fused  vs  ResNet-18 alone      (does frequency add to the pixel network?)
    fused  vs  frequency MLP alone  (does the pixel network add to frequency?)

ResNet-18 here is the seed-643 checkpoint stored by `src.models.pixel_cnn`; the frequency model is `mlp_all`.

    python -m src.eval.late_fusion
"""
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import FEATURES, MODELS, RESULTS, SEED, UNSEEN
from src.eval.resize_ablation import paired_ci
from src.models import data as D
from src.models import pixel_cnn as P


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> pd.DataFrame:
    df, frames, cidx, wild, wframes, widx = P.load_frames()
    cols = D.feature_cols(df, "all")
    entry = joblib.load(MODELS / "mlp_all.joblib")
    mlp, cols = entry["model"], entry.get("cols", cols)
    net = P.resnet18_ft().cuda()
    net.load_state_dict(torch.load(MODELS / "resnet18.pt", map_location="cuda"))

    def scores(sub: pd.DataFrame):
        a = P.predict_clips(net, frames, [cidx[c] for c in sub.clip_id])
        b = mlp.predict_proba(D.xy(sub, cols)[0])[:, 1]
        return logit(a), logit(b)

    va = D.select(df, "val")
    ra, fa = scores(va)
    stack = LogisticRegression(C=1.0).fit(np.c_[ra, fa], va.label.to_numpy())
    print(f"stacker weights (resnet, frequency): {stack.coef_[0].round(3)}, bias {stack.intercept_.round(3)}", flush=True)

    te = df[(df.split == "test") & ~df.aug.astype(bool)]
    real = te[te.generator == "real"]
    sets = {"seen": te[~te.generator.isin(UNSEEN)]}
    for g in UNSEEN:
        sets[g] = pd.concat([real, te[te.generator == g]])
    rows, rng = [], np.random.default_rng(SEED)

    def report(name, y, r, f):
        fused = stack.decision_function(np.c_[r, f])
        y = np.asarray(y).astype(int)
        a_r, a_f, a_u = roc_auc_score(y, r), roc_auc_score(y, f), roc_auc_score(y, fused)
        lo1, hi1 = paired_ci(y, fused, r, rng)
        lo2, hi2 = paired_ci(y, fused, f, rng)
        rows.append({"set": name, "n": len(y), "resnet18": a_r, "frequency_mlp": a_f, "fused": a_u,
                     "fused_minus_resnet": a_u - a_r, "ci_low_r": lo1, "ci_high_r": hi1,
                     "fused_minus_frequency": a_u - a_f, "ci_low_f": lo2, "ci_high_f": hi2})

    for name, sub in sets.items():
        r, f = scores(sub)
        report(name, sub.label, r, f)

    # the 250 public clips: the same models through the deployment path, versus the public real clips
    wf = pd.read_parquet(FEATURES / "wild.parquet").set_index("key")
    wf = wf.loc[list(wild.index)]
    r_all = logit(P.predict_clips(net, wframes, [widx[k] for k in wild.index]))
    f_all = logit(mlp.predict_proba(np.nan_to_num(wf[cols].to_numpy(np.float32)))[:, 1])
    is_real = (wild.source == "real").to_numpy()
    for src in sorted(s for s in wild.source.unique() if s != "real"):
        m = (wild.source == src).to_numpy() | is_real
        report(f"public: {src}", wild.label.to_numpy()[m], r_all[m], f_all[m])

    res = pd.DataFrame(rows)
    res.round(3).to_csv(RESULTS / "late_fusion.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
