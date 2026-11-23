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
NB_BLUE, NB_RED = "#2a6f97", "#c9184a"          # the colours of novelty.ipynb
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
    """Test 1 (as in novelty.ipynb): native crop vs resized frames, MLP, grouped 5-fold CV."""
    d = pd.read_csv(RESULTS / "resize_ablation_cv.csv")
    d = d[d.model == "mlp"].set_index("features").loc[["pixel", "all", "fusion"]]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.36
    for j, (col, color, lab) in enumerate([("crop", NB_BLUE, "Native crop"), ("resize", NB_RED, "Resized")]):
        ax.bar(np.arange(3) + (j - 0.5) * w, d[col], w, color=color, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Pixel", "Frequency", "Combined"])
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUROC (MLP, 5-fold CV)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_resize.png")


def prompts():
    """Test 3 (as in novelty.ipynb): ModelScope clips from the training prompts vs 36 new prompts."""
    d = pd.read_csv(RESULTS / "fresh_prompts.csv").set_index("model")
    names = [("mlp_pixel", "Pixel"), ("mlp_all", "Frequency"), ("mlp_fusion", "Combined"), ("resnet18", "ResNet-18")]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.36
    for j, (col, color, lab) in enumerate([("auroc_old_prompts", NB_BLUE, "Old prompts"), ("auroc_new_prompts", NB_RED, "New prompts")]):
        ax.bar(np.arange(4) + (j - 0.5) * w, [d.loc[m, col] for m, _ in names], w, color=color, label=lab)
    ax.axhline(0.5, color="k", lw=1.2, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in names], fontsize=17)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC, ModelScope vs real")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_prompts.png")


def fusion():
    """Test 4 (as in novelty.ipynb): ResNet-18 alone, frequency MLP alone, and the two fused."""
    d = pd.read_csv(RESULTS / "late_fusion.csv").set_index("set").loc[["seen", "modelscope", "wan"]]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.26
    for j, (col, color, lab) in enumerate([("resnet18", "#999999", "ResNet-18"), ("frequency_mlp", NB_BLUE, "Frequency MLP"), ("fused", "#f2b705", "Both fused")]):
        ax.bar(np.arange(3) + (j - 1) * w, d[col], w, color=color, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Seen", "ModelScope\n(unseen)", "Wan2.1\n(unseen)"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (held-out clips)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, ncol=3, fontsize=16, columnspacing=1.0, handlelength=1.0)
    save(fig, "fig_fusion.png")


def temporal_order():
    """Test 2 (as in novelty.ipynb): AUROC on seen generators with the original frame order vs shuffled frames, MLP."""
    d = pd.read_csv(RESULTS / "temporal_order.csv")
    d = d[(d.set == "seen") & (d.variant == "shuffled")].set_index("model").loc[["mlp_spatial", "mlp_temporal", "mlp_all"]]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.36
    for j, (col, color, lab) in enumerate([("auroc_original", NB_BLUE, "Original order"), ("auroc_variant", NB_RED, "Frames shuffled")]):
        ax.bar(np.arange(3) + (j - 0.5) * w, d[col], w, color=color, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Spatial only", "Temporal only", "Both"], fontsize=17)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC (seen generators)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_temporal.png")


REPO_URL = "https://github.com/Astitva-Garg/Detection-of-AI-Generated-Video-via-Frequency-Domain"
DEMO_URL = "https://drive.google.com/file/d/1j2pN5h0CsaIPw2K7vSTBoML7otCmXgHS/view?usp=drive_link"


def qr():
    """QR codes of the repository and demo-video links (need the optional package `segno`)."""
    try:
        import segno
    except ImportError:
        print("segno not installed: keeping the existing poster/figs/qr_*.svg (pip install segno to regenerate)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in (("qr_repo.svg", REPO_URL), ("qr_demo.svg", DEMO_URL)):
        segno.make(url, error="m").save(OUT / name, scale=10, border=1, dark="#000000")
        print("saved", name)


if __name__ == "__main__":
    for f in (compare, models, resize, degradation, spectra, temporal_order, qr):
        f()
