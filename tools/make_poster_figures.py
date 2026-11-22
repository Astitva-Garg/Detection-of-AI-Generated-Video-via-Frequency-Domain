"""Figures for the A1 poster, drawn from the result files at their printed size (figsize in inches = size on paper,
so a font size of 18 means 18 pt on the poster). Every figure reads a file in results_crop/ or demo_output/.

    python -m tools.make_poster_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RESULTS

OUT = PROJECT_ROOT / "poster" / "figs"
COL, WIDE = 7.0, 14.4          # one column / two columns, inches (A1 poster, 3 columns)
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#2a6f97", "#e07a5f", "#3d9970", "#c9184a", "#7b5ea7", "#8a8f98"
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 19,
                     "ytick.labelsize": 19, "legend.fontsize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "axes.linewidth": 1.6})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def table(path):
    d = pd.read_csv(path, header=[0, 1], index_col=0)
    return d[d.index.notna() & (d.index != "model")]


def compare():
    """Pixel vs frequency vs fusion, MLP, same held-out clips, 95 % bootstrap intervals."""
    c = pd.read_csv(RESULTS / "live_comparison.csv")   # written by `python -m src.demo_live`
    c = c[(c.model == "mlp") & c.set.str.contains("seen|ModelScope|Wan")]
    sets = [("seen", "Seen"), ("ModelScope", "ModelScope\n(unseen)"), ("Wan2.1", "Wan2.1\n(unseen)")]
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    w = 0.26
    for j, (feat, col, lab) in enumerate([("pixel", GREY, "Pixel only"), ("frequency", BLUE, "Frequency only"),
                                          ("fusion", RED, "Fusion")]):
        v, lo, hi = [], [], []
        for key, _ in sets:
            r = c[c.set.str.startswith(key) & (c.features == feat)].iloc[0]
            v.append(r.auroc); lo.append(r.auroc - r.ci_low); hi.append(r.ci_high - r.auroc)
        x = np.arange(3) + (j - 1) * w
        ax.bar(x, v, w, color=col, label=lab, yerr=[lo, hi], capsize=5, error_kw={"lw": 2})
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in sets])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC (held-out clips)")
    ax.axhline(0.5, color="k", lw=1.2, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, ncol=3, fontsize=16, columnspacing=1.0, handlelength=1.0)
    save(fig, "fig_compare.png")


def models():
    """Main comparison, mean over 5 seeds, all models."""
    s, p = table(RESULTS / "seed_stability.csv"), table(RESULTS / "pixel_deep.csv")
    cols = [("seen", "Seen"), ("modelscope", "ModelScope\n(unseen)"), ("wan", "Wan2.1\n(unseen)"),
            ("wild_mean", "250 public\nclips")]
    rows = [("mlp_fusion", s, "MLP fusion", RED), ("mlp_all", s, "MLP frequency", BLUE), ("cnn", s, "Spectro-temporal CNN", GREEN),
            ("mlp_pixel", s, "MLP pixel stats", GREY), ("pixel_cnn", p, "Pixel CNN (scratch)", PURPLE),
            ("resnet18", p, "ResNet-18 (fine-tuned)", ORANGE)]
    fig, ax = plt.subplots(figsize=(COL, 3.6))
    w = 0.13
    for j, (key, tab, lab, col) in enumerate(rows):
        m = [float(tab.loc[key, (c, "mean")]) for c, _ in cols]
        sd = [float(tab.loc[key, (c, "std")]) for c, _ in cols]
        ax.bar(np.arange(len(cols)) + (j - 2.5) * w, m, w, color=col, label=lab, yerr=sd, capsize=3, error_kw={"lw": 1.6})
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=15)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (mean of 5 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False, fontsize=16, columnspacing=1.0)
    save(fig, "fig_models.png")


def resize():
    d = pd.read_csv(RESULTS / "resize_ablation_cv.csv").set_index(["model", "features"])
    items = [("lr", "LR"), ("lgbm", "LightGBM"), ("mlp", "MLP")]
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    w = 0.2
    spec = [("all", "crop", BLUE, "Frequency, native crop"), ("all", "resize", "#9dc4dd", "Frequency, resized"),
            ("pixel", "crop", GREY, "Pixel, native crop"), ("pixel", "resize", "#c9ccd1", "Pixel, resized")]
    for j, (feat, var, col, lab) in enumerate(spec):
        v = [d.loc[(m, feat), var] for m, _ in items]
        ax.bar(np.arange(3) + (j - 1.5) * w, v, w, color=col, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in items])
    ax.set_ylim(0.7, 0.95); ax.set_ylabel("AUROC (grouped 5-fold CV)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_resize.png")


def degradation():
    d = pd.read_csv(RESULTS / "degraded.csv")
    d = d[d.subset == "seen"]
    names = [("h264_crf23", "H.264\nCRF 23"), ("h264_crf30", "H.264\nCRF 30"), ("h264_crf38", "H.264\nCRF 38"),
             ("blur1.0", "Blur"), ("down0.5", "Down-\nscale"), ("noise5", "Noise")]
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    w = 0.27
    for j, (m, col, lab) in enumerate([("mlp_pixel", GREY, "Pixel"), ("mlp_all", BLUE, "Frequency"), ("mlp_fusion", RED, "Fusion")]):
        v = [float(d[(d.model == m) & (d.degradation == k)].auroc.iloc[0]) for k, _ in names]
        ax.bar(np.arange(len(names)) + (j - 1) * w, v, w, color=col, label=lab)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([l for _, l in names], fontsize=15)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC (seen test)"); ax.axhline(0.5, color="k", lw=1.2, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=17)
    save(fig, "fig_degrade.png")


def spectra():
    """Mean 2D log power spectrum of real clips and each generator's difference from real (40 clips each)."""
    from src.config import CLIPS, FPS, N_FRAMES
    from src.data.video_io import read_frames
    from src.features.spatial import power_spectrum, to_luma
    from src.models import data as D
    df = D.clean_table()
    df = df[~df.aug.astype(bool)]
    means = {}
    for g in ["real", "ltx", "animatediff", "svd", "modelscope", "wan"]:
        ids = df[df.generator == g].clip_id.sample(min(40, (df.generator == g).sum()), random_state=0)
        acc = 0
        for c in ids:
            fr = read_frames(CLIPS / f"{c}.mp4", N_FRAMES, FPS)
            acc = acc + np.mean([np.log(power_spectrum(to_luma(f)) + 1e-8) for f in fr[::4]], axis=0)
        means[g] = acc / len(ids)
    names = {"ltx": "LTX", "animatediff": "AnimateDiff", "svd": "SVD", "modelscope": "ModelScope*", "wan": "Wan2.1*"}
    fig, axs = plt.subplots(2, 3, figsize=(COL, 4.4))
    axs = axs.ravel()
    axs[0].imshow(means["real"], cmap="magma"); axs[0].set_title("Real", fontsize=17)
    axs[0].annotate("coarse\nstructure", xy=(128, 128), xytext=(128, 200), color="white", ha="center", fontsize=12,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
    axs[0].annotate("fine\ndetail", xy=(252, 128), xytext=(212, 50), color="white", ha="center", fontsize=12,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
    allv = np.concatenate([(means[g] - means["real"]).ravel() for g in names])
    v = np.percentile(np.abs(allv), 99)
    for ax, g in zip(axs[1:], names):
        im = ax.imshow(means[g] - means["real"], cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(names[g], fontsize=17)
    for a in axs:
        a.axis("off")
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, "fig_spectra.png")


def radial():
    from src.viz import figures as F
    from src.models import data as D
    F.FIG = OUT
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    F.radial_profiles(D.clean_table())
    (OUT / "radial_profiles.png").replace(OUT / "fig_radial.png")


def temporal_order():
    p = RESULTS / "temporal_order.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.variant == "shuffled"]
    sets = [("seen", "Seen"), ("ModelScope (unseen)", "ModelScope"), ("Wan2.1 (unseen)", "Wan2.1")]
    models_ = [("mlp_spatial", GREY, "Spatial only (0 by design)"), ("mlp_temporal", GREEN, "Temporal only"), ("mlp_all", BLUE, "Spatial + temporal")]
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    w = 0.26
    for j, (m, col, lab) in enumerate(models_):
        r = [d[(d.model == m) & (d.set == s)].iloc[0] for s, _ in sets]
        v = [x.change for x in r]
        err = [[x.change - x.ci_low for x in r], [x.ci_high - x.change for x in r]]
        ax.bar(np.arange(3) + (j - 1) * w, v, w, color=col, label=lab, yerr=err, capsize=4, error_kw={"lw": 1.8})
    ax.axhline(0, color="k", lw=1.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in sets])
    ax.set_ylabel("Change in AUROC\nwhen frames are shuffled")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False, fontsize=15)
    save(fig, "fig_temporal.png")


REPO_URL = "https://github.com/Astitva-Garg/Detection-of-AI-Generated-Video-via-Frequency-Domain"


def qr():
    """QR code of the repository link (needs the optional package `segno`)."""
    try:
        import segno
    except ImportError:
        print("segno not installed: keeping the existing poster/figs/qr_repo.svg (pip install segno to regenerate)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    segno.make(REPO_URL, error="m").save(OUT / "qr_repo.svg", scale=10, border=1, dark="#000000")
    print("saved qr_repo.svg")


if __name__ == "__main__":
    for f in (compare, models, resize, degradation, spectra, temporal_order, qr):
        f()
