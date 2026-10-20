"""Figures for report/slides: spectra, drift, importance, saliency, ROC, confusion, robustness, errors."""
import argparse
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

from src.config import CLIPS, FPS, MODELS, N_BINS, N_FRAMES, RESULTS, SEEN, UNSEEN
from src.data.video_io import read_frames
from src.features.spatial import power_spectrum, to_luma
from src.models import data as D

FIG = RESULTS / "figures"
ORDER = ["real"] + SEEN + UNSEEN
COLORS = {"real": "#2a6f97", "ltx": "#e07a5f", "animatediff": "#f2a541", "svd": "#81b29a",
          "wan": "#9d4edd", "modelscope": "#c9184a"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def _gens(df):
    return [g for g in ORDER if (df.generator == g).any()]


def mean_spectrum_2d(df, n_per=40):
    gens = _gens(df)
    means = {}
    for g in gens:
        ids = df[df.generator == g].clip_id.sample(min(n_per, (df.generator == g).sum()), random_state=0)
        acc = None
        for c in ids:
            fr = read_frames(CLIPS / f"{c}.mp4", N_FRAMES, FPS)
            s = np.mean([np.log(power_spectrum(to_luma(f)) + 1e-8) for f in fr[::4]], axis=0)
            acc = s if acc is None else acc + s
        means[g] = acc / len(ids)
    fig, ax = plt.subplots(2, len(gens), figsize=(2.6 * len(gens), 5.2))
    ax = np.atleast_2d(ax)
    for j, g in enumerate(gens):
        ax[0, j].imshow(means[g], cmap="magma")
        ax[0, j].set_title(g)
        if g == "real":
            ax[1, j].text(0.5, 0.5, "reference\n(differences\nare vs. real)", ha="center", va="center",
                          transform=ax[1, j].transAxes)
        else:
            d = means[g] - means["real"]
            v = np.percentile(np.abs(d), 99)
            ax[1, j].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
            ax[1, j].set_title(f"{g} − real")
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle("Mean log power spectrum (top) and difference from real (bottom; red = more energy)")
    _save(fig, "mean_spectrum_2d.png")


def radial_profiles(df):
    cols = [f"sp_rad_{i:02d}_mean" for i in range(N_BINS)]
    rcols = [f"sp_res_rad_{i:02d}_mean" for i in range(N_BINS)]
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for g in _gens(df):
        for a, cc in zip(ax, (cols, rcols)):
            v = df[df.generator == g][cc].to_numpy()
            m, s = v.mean(0), v.std(0)
            a.plot(x, m, color=COLORS[g], label=g, lw=2 if g == "real" else 1.4)
            a.fill_between(x, m - s / 2, m + s / 2, color=COLORS[g], alpha=0.12)
    ax[0].set_title("Radial log power spectrum of frames")
    ax[1].set_title("Radial spectrum of noise residual")
    for a in ax:
        a.set_xlabel("spatial frequency (fraction of Nyquist)")
        a.set_ylabel("log power (mean removed)")
    ax[0].legend()
    _save(fig, "radial_profiles.png")


def radial_hf_zoom(df):
    cols = [f"sp_rad_{i:02d}_mean" for i in range(N_BINS)]
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    real = df[df.generator == "real"][cols].to_numpy().mean(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in _gens(df):
        if g == "real":
            continue
        ax.plot(x, df[df.generator == g][cols].to_numpy().mean(0) - real, color=COLORS[g], label=g)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("spatial frequency (fraction of Nyquist)")
    ax.set_ylabel("Δ log power vs real")
    ax.set_title("Spectral deviation of each generator from real video")
    ax.legend()
    _save(fig, "radial_deviation.png")


def drift_examples(df, k=3):
    seq_ids = []
    for lab, g in ((0, "real"), (1, None)):
        sub = df[df.label == lab]
        seq_ids += list(sub.clip_id.sample(k, random_state=1))
    seqs = D.load_seq("clean", seq_ids)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, c in enumerate(seq_ids):
        s = seqs[i, 0]
        l2 = np.linalg.norm(np.diff(s, axis=0), axis=1)
        hi = s[:, 2 * N_BINS // 3:].mean(1)
        g = df.set_index("clip_id").loc[c, "generator"]
        style = dict(color=COLORS[g], ls="-" if g == "real" else "--", label=f"{g}")
        ax[0].plot(np.arange(1, N_FRAMES), l2, **style)
        ax[1].plot(np.arange(N_FRAMES), hi - hi.mean(), **style)
    ax[0].set_title("Frame-to-frame spectral drift (L2)")
    ax[0].set_xlabel("frame")
    ax[1].set_title("High-frequency band log power over time (centred)")
    ax[1].set_xlabel("frame")
    h, l = ax[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax[0].legend(uniq.values(), uniq.keys(), fontsize=8)
    _save(fig, "drift_examples.png")


def drift_distributions(df):
    feats = ["drift_l2_mean", "drift_l2_ac1", "drift_band_high_mean", "rt_lag1_corr", "rt_hi_temporal_frac",
             "drift_hf_ratio_50_mean"]
    feats = [f for f in feats if f in df.columns]
    gens = _gens(df)
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, f in zip(ax.ravel(), feats):
        data = [df[df.generator == g][f].dropna().to_numpy() for g in gens]
        lo, hi = np.percentile(np.concatenate(data), [1, 99])
        data = [np.clip(d, lo, hi) for d in data]
        vp = a.violinplot(data, showmedians=True)
        for b, g in zip(vp["bodies"], gens):
            b.set_facecolor(COLORS[g])
            b.set_alpha(0.6)
        a.set_xticks(range(1, len(gens) + 1), gens, rotation=30)
        a.set_title(f)
    _save(fig, "drift_distributions.png")


GROUP_RULES = [("radial low (frame)", r"^sp_rad_(0\d|1[0-5])_"), ("radial mid (frame)", r"^sp_rad_(1[6-9]|2\d|3[01])_"),
               ("radial high (frame)", r"^sp_rad_(3[2-9]|4\d)_"), ("residual spectrum", r"^sp_res_rad_"),
               ("spectral scalars", r"^sp_(slope|hf_ratio|centroid|rolloff)"), ("DCT bands", r"^sp_dct_"),
               ("angular", r"^sp_ang_"), ("spectral drift", r"^drift_(l2|cos|band)"),
               ("scalar drift", r"^drift_(slope|hf_ratio|centroid|dct)"), ("residual temporal", r"^rt_")]


def _group_of(col):
    for name, rx in GROUP_RULES:
        if re.search(rx, col):
            return name
    return "other"


def feature_importance(df):
    ent = joblib.load(MODELS / "lgbm_all.joblib")
    m, cols = ent["model"], ent["cols"]
    te = D.select(df, "test")
    X, _ = D.xy(te, cols)
    contrib = np.abs(m.predict(X, pred_contrib=True)[:, :-1]).mean(0)
    gain = m.booster_.feature_importance("gain")
    s = pd.DataFrame({"col": cols, "shap": contrib, "gain": gain / gain.sum()})
    s["group"] = s.col.map(_group_of)
    g = s.groupby("group")[["shap", "gain"]].sum().sort_values("shap")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(g.index, g.shap, color="#2a6f97")
    ax[0].set_title("Mean |SHAP| by feature group (LightGBM, test set)")
    top = s.sort_values("shap").tail(15)
    ax[1].barh(top.col, top.shap, color=["#e07a5f" if c.startswith(("drift_", "rt_")) else "#2a6f97" for c in top.col])
    ax[1].set_title("Top-15 features (orange = temporal, blue = spatial)")
    _save(fig, "feature_importance_groups.png")
    s.to_csv(RESULTS / "feature_importance.csv", index=False)

    # importance over frequency
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for prefix, lab, col in (("sp_rad_", "frame spectrum", "#2a6f97"), ("sp_res_rad_", "residual spectrum", "#e07a5f")):
        v = [s[s.col.str.match(rf"^{prefix}{i:02d}_")].shap.sum() for i in range(N_BINS)]
        ax.plot(x, v, color=col, label=lab, marker="o", ms=3)
    ax.set_xlabel("spatial frequency (fraction of Nyquist)")
    ax.set_ylabel("mean |SHAP| (summed mean+std)")
    ax.set_title("Which spatial frequencies drive the decision")
    ax.legend()
    _save(fig, "importance_over_frequency.png")


def cnn_saliency(df, n=200):
    ck = torch.load(MODELS / "cnn.pt", weights_only=False)
    from src.models.cnn import SpecTempCNN
    m = SpecTempCNN(ck["in_ch"], ck["n_bins"])
    m.load_state_dict(ck["state"])
    m.eval()
    te = D.select(df, "test")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, (lab, name) in enumerate(((0, "real"), (1, "AI-generated"))):
        ids = te[te.label == lab].clip_id
        ids = ids.sample(min(n, len(ids)), random_state=0)
        X = torch.tensor((D.load_seq("clean", ids) - ck["mu"]) / ck["sd"], dtype=torch.float32, requires_grad=True)
        m(X).sum().backward()
        sal = (X.grad * X).abs().mean(0).detach().numpy()   # (C, T, F)
        im = ax[j].imshow(sal[0].T, origin="lower", aspect="auto", cmap="viridis",
                          extent=[0, N_FRAMES, 0, 1])
        ax[j].set_title(f"CNN |grad×input| — {name} clips (frame spectra)")
        ax[j].set_xlabel("frame")
        ax[j].set_ylabel("frequency (frac. Nyquist)")
        fig.colorbar(im, ax=ax[j])
    _save(fig, "cnn_saliency.png")


def roc_and_confusion(df):
    pr = pd.read_csv(RESULTS / "predictions.csv")
    models = [m for m in ["lr_all", "svm_all", "mlp_all", "lgbm_all", "cnn", "lgbm_all_robust"] if m in pr.model.unique()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    seen = pr[~pr.generator.isin(UNSEEN)]
    for mname in models:
        s = seen[seen.model == mname]
        f, t, _ = roc_curve(s.label, s.p_fake)
        ax[0].plot(f, t, label=mname)
    ax[0].set_title("ROC — seen-generator test set")
    s = pr[pr.model == "lgbm_all"]
    for g in _gens(df):
        if g == "real":
            continue
        sub = pd.concat([s[s.generator == g], s[s.generator == "real"]])
        f, t, _ = roc_curve(sub.label, sub.p_fake)
        ax[1].plot(f, t, color=COLORS[g], label=f"{g}{' (unseen)' if g in UNSEEN else ''}")
    ax[1].set_title("ROC per generator — lgbm_all")
    for a in ax:
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        a.set_xlabel("false positive rate")
        a.set_ylabel("true positive rate")
        a.legend(fontsize=8)
    _save(fig, "roc_curves.png")

    fig, ax = plt.subplots(1, len(models), figsize=(2.9 * len(models), 3))
    for a, mname in zip(np.atleast_1d(ax), models):
        s = seen[seen.model == mname]
        cm = confusion_matrix(s.label, s.p_fake >= 0.5)
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["real", "fake"])
        a.set_yticks([0, 1], ["real", "fake"])
        a.set_xlabel("predicted")
        a.set_title(mname)
    np.atleast_1d(ax)[0].set_ylabel("true")
    _save(fig, "confusion_matrices.png")


def ablation_bars():
    t = pd.read_csv(RESULTS / "test_metrics.csv")
    rows = []
    for m in ["lr", "svm", "mlp", "lgbm"]:
        for g in ["spatial", "temporal", "all"]:
            for st in ["seen_test", "unseen_test"]:
                r = t[(t.model == f"{m}_{g}") & (t.set == st)]
                if len(r):
                    rows.append({"model": m, "group": g, "set": st, "auroc": r.auroc.iloc[0]})
    a = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for k, st in enumerate(["seen_test", "unseen_test"]):
        sub = a[a.set == st]
        w = 0.25
        for i, g in enumerate(["spatial", "temporal", "all"]):
            v = sub[sub.group == g].set_index("model").reindex(["lr", "svm", "mlp", "lgbm"]).auroc
            ax[k].bar(np.arange(4) + (i - 1) * w, v, w, label=g)
        ax[k].set_xticks(range(4), ["LR", "Lin-SVM", "MLP", "LightGBM"])
        ax[k].set_title(f"AUROC — {st.replace('_', ' ')}")
        ax[k].set_ylim(0.4, 1.0)
    ax[0].legend(title="features")
    _save(fig, "ablation_bars.png")
    a.to_csv(RESULTS / "ablation.csv", index=False)


def degradation_bars():
    d = pd.read_csv(RESULTS / "degraded.csv")
    t = pd.read_csv(RESULTS / "test_metrics.csv")
    d = d[d.subset == "seen"]
    models = [m for m in ["lgbm_all", "lgbm_all_robust", "cnn", "lr_all"] if m in d.model.unique()]
    degs = ["clean"] + sorted(d.degradation.unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        clean = t[(t.model == m) & (t.set == "seen_test")].auroc.iloc[0]
        vals = [clean] + [d[(d.model == m) & (d.degradation == g)].auroc.iloc[0] for g in degs[1:]]
        ax.bar(np.arange(len(degs)) + (i - (len(models) - 1) / 2) * w, vals, w, label=m)
    ax.set_xticks(range(len(degs)), degs, rotation=20)
    ax.set_ylabel("AUROC (seen-generator test)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_title("Robustness to post-processing")
    ax.legend(fontsize=8)
    _save(fig, "degradation_bars.png")


def tsne(df):
    cols = D.feature_cols(df, "all")
    base = df[~df.aug.astype(bool)]
    sub = pd.concat([g.sample(min(len(g), 150), random_state=0) for _, g in base.groupby("generator")])
    X = StandardScaler().fit_transform(D.xy(sub, cols)[0])
    Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for g in _gens(sub):
        k = (sub.generator == g).to_numpy()
        ax.scatter(Z[k, 0], Z[k, 1], s=10, color=COLORS[g], label=g, alpha=0.7,
                   marker="x" if g in UNSEEN else "o")
    ax.legend(markerscale=2)
    ax.set_title("t-SNE of frequency + drift features (x = unseen generators)")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, "tsne.png")


def error_gallery(df, k=8):
    pr = pd.read_csv(RESULTS / "predictions.csv")
    s = pr[pr.model == "lgbm_all"].copy()
    s["err"] = np.abs(s.label - s.p_fake)
    s = s[s.err > 0.5].sort_values("err", ascending=False).head(k)
    if s.empty:
        return
    cols = [f"sp_rad_{i:02d}_mean" for i in range(N_BINS)]
    feat = df.set_index("clip_id")
    real_m = df[df.label == 0][cols].to_numpy().mean(0)
    fake_m = df[df.label == 1][cols].to_numpy().mean(0)
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, ax = plt.subplots(2, len(s), figsize=(2.6 * len(s), 5))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, r in enumerate(s.itertuples()):
        fr = read_frames(CLIPS / f"{r.clip_id}.mp4", 1, FPS)[0]
        ax[0, j].imshow(fr)
        ax[0, j].axis("off")
        ax[0, j].set_title(f"{r.generator}\nP(fake)={r.p_fake:.2f}", fontsize=8)
        ax[1, j].plot(x, real_m, color=COLORS["real"], lw=1, label="real mean")
        ax[1, j].plot(x, fake_m, color="#e07a5f", lw=1, label="fake mean")
        ax[1, j].plot(x, feat.loc[r.clip_id, cols].to_numpy(float), color="k", lw=1.4, label="this clip")
        ax[1, j].set_xticks([])
    ax[1, 0].legend(fontsize=6)
    fig.suptitle("Most confident errors (lgbm_all): first frame and radial spectrum vs class means")
    _save(fig, "error_gallery.png")


def training_curves():
    p = RESULTS / "cnn_history.csv"
    if not p.exists():
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(h.epoch, h.train_loss)
    ax[0].set_title("CNN training loss (BCE)")
    ax[1].plot(h.epoch, h.val_auc, color="#e07a5f")
    ax[1].set_title("CNN validation AUROC")
    for a in ax:
        a.set_xlabel("epoch")
    _save(fig, "cnn_training_curves.png")


def dataset_examples(df):
    gens = _gens(df)
    fig, ax = plt.subplots(len(gens), 4, figsize=(9, 2.3 * len(gens)))
    for i, g in enumerate(gens):
        cid = df[df.generator == g].clip_id.sample(1, random_state=3).iloc[0]
        fr = read_frames(CLIPS / f"{cid}.mp4", 4, 2)
        for j in range(4):
            ax[i, j].imshow(fr[j])
            ax[i, j].axis("off")
        ax[i, 0].set_title(g, loc="left", fontsize=9)
    fig.suptitle("Dataset examples (normalised 256×256 clips, frames 0.5 s apart)")
    _save(fig, "dataset_examples.png")


def main(only: list[str] | None = None):
    df = D.clean_table()
    steps = {"examples": lambda: dataset_examples(df), "spectrum2d": lambda: mean_spectrum_2d(df),
             "radial": lambda: (radial_profiles(df), radial_hf_zoom(df)), "drift": lambda: (drift_examples(df), drift_distributions(df)),
             "importance": lambda: feature_importance(df), "saliency": lambda: cnn_saliency(df),
             "roc": lambda: roc_and_confusion(df), "ablation": ablation_bars, "degradation": degradation_bars,
             "tsne": lambda: tsne(df), "errors": lambda: error_gallery(df), "curves": training_curves}
    for k, f in steps.items():
        if only and k not in only:
            continue
        try:
            f()
        except Exception as e:
            print(f"figure {k} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    main(ap.parse_args().only)
