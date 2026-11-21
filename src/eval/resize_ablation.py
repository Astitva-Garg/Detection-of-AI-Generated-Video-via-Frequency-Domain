"""Resize versus native crop, on identical clips and identical splits.

The main pipeline cuts a 256 x 256 square at native resolution ("crop"). The ablation shrinks every frame to a
short side of 256 first ("resize"; run the pipeline with IE643_NORM=resize). Resizing is a low-pass filter, so it
should remove the detail the frequency features measure. This script checks that with the same clips, the same
train / validation / test split (the `split` column of the crop manifest) and the same models:

  * AUROC on the held-out clips for pixel, frequency and fusion features, in both variants;
  * a paired bootstrap on the same test clips for the difference resize minus crop;
  * the grouped 5-fold cross-validation of both variants side by side (results/cv_summary.csv vs
    results_crop/cv_summary.csv), if the resize cross-validation has been run.

    python -m src.eval.resize_ablation       # needs data/features (resize) and data/features_crop (crop)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_ROOT, PROJECT_ROOT, SEED, UNSEEN
from src.features.temporal import group_columns
from src.models.classic import fit_model
from src.models.data import xy

OUT = PROJECT_ROOT / "results_crop"
MODELS = ["lr", "lgbm", "mlp"]
GROUPS = {"pixel": "pixel", "frequency": "all", "fusion": "fusion"}
N_BOOT = 1000


def load(manifest: str, feats: str) -> pd.DataFrame:
    f = pd.read_parquet(DATA_ROOT / feats / "clean.parquet")
    px = DATA_ROOT / feats / "pixel.parquet"
    if px.exists():
        f = f.merge(pd.read_parquet(px), on="key", how="left")
    f = f.rename(columns={"key": "clip_id"})
    return pd.read_csv(DATA_ROOT / manifest).merge(f, on="clip_id", how="inner")


def paired_ci(y, a, b, rng):
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    return np.percentile(d, [2.5, 97.5])


def main() -> pd.DataFrame:
    crop = load("manifest_crop.csv", "features_crop")
    resize = load("manifest.csv", "features")
    ids = sorted(set(crop.clip_id) & set(resize.clip_id))
    print(f"clips in both variants: {len(ids)} (crop {len(crop)}, resize {len(resize)})", flush=True)
    crop = crop[crop.clip_id.isin(ids)].sort_values("clip_id").reset_index(drop=True)
    resize = resize[resize.clip_id.isin(ids)].sort_values("clip_id").reset_index(drop=True)
    split = crop.set_index("clip_id")[["split", "aug", "generator", "label"]]
    resize = resize.drop(columns=["split", "aug", "generator", "label"]).merge(split, left_on="clip_id", right_index=True)
    variants = {"crop": crop, "resize": resize}
    seen = ~split.generator.isin(UNSEEN)
    tr_ids = split[(split.split == "train") & seen].index
    va_ids = split[(split.split == "val") & seen & ~split.aug.astype(bool)].index
    te = split[(split.split == "test") & ~split.aug.astype(bool)]
    real = te.generator == "real"
    sets = {"seen": ~te.generator.isin(UNSEEN),
            "ModelScope (unseen)": real | (te.generator == "modelscope"),
            "Wan2.1 (unseen)": real | (te.generator == "wan")}
    scores, rows = {}, []
    for vname, df in variants.items():
        d = df.set_index("clip_id")
        for gname, g in GROUPS.items():
            cols = group_columns(df.columns, g)
            for m in MODELS:
                model = fit_model(m, *xy(d.loc[tr_ids], cols), *xy(d.loc[va_ids], cols))
                p = model.predict_proba(xy(d.loc[te.index], cols)[0])[:, 1]
                scores[(vname, gname, m)] = p
                print(f"  {vname:<7s}{gname:<10s}{m:<5s}{len(cols):>4d} features", flush=True)
    y = te.label.to_numpy().astype(int)
    rng = np.random.default_rng(SEED)
    for sname, mask in sets.items():
        mk = mask.to_numpy()
        for gname in GROUPS:
            for m in MODELS:
                a, b = scores[("resize", gname, m)][mk], scores[("crop", gname, m)][mk]
                lo, hi = paired_ci(y[mk], a, b, rng)
                ra, rb = roc_auc_score(y[mk], a), roc_auc_score(y[mk], b)
                rows.append({"set": sname, "features": gname, "model": m, "n_test": int(mk.sum()),
                             "auroc_crop": rb, "auroc_resize": ra, "resize_minus_crop": ra - rb,
                             "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(rows)
    res.round(3).to_csv(OUT / "resize_ablation.csv", index=False)
    print("\nAUROC on identical held-out clips, resize minus crop (95 % paired bootstrap interval)")
    for sname in sets:
        print(f"\n  {sname}")
        print(f"  {'features':<11s}{'model':<6s}{'crop':>7s}{'resize':>8s}{'diff':>8s}   interval")
        for r in res[res.set == sname].itertuples():
            print(f"  {r.features:<11s}{r.model:<6s}{r.auroc_crop:7.3f}{r.auroc_resize:8.3f}{r.resize_minus_crop:+8.3f}   "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
    cv_a, cv_b = PROJECT_ROOT / "results" / "cv_summary.csv", PROJECT_ROOT / "results_crop" / "cv_summary.csv"
    if cv_a.exists():
        def table(p):
            d = pd.read_csv(p, header=[0, 1], index_col=[0, 1])
            return d[("auroc", "mean")].rename("auroc")
        a, b = table(cv_a), table(cv_b)
        cv = pd.DataFrame({"crop": b, "resize": a}).dropna()
        cv["resize_minus_crop"] = cv.resize - cv.crop
        cv.index.names = ["model", "features"]
        cv.round(3).to_csv(OUT / "resize_ablation_cv.csv")
        print("\nGrouped 5-fold CV AUROC (all clips of the seen generators), crop vs resize")
        print(cv.round(3).to_string())
    return res


if __name__ == "__main__":
    main()
