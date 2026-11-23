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
    groups = [("pixel", "Pixel", GREY), ("all", "Frequency", BLUE), ("fusion", "Combined", RED)]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.34
    for i, (feat, _, col) in enumerate(groups):
        ax.bar(i - w / 2, d.loc[("mlp", feat), "crop"], w, color=col, edgecolor="white")
        ax.bar(i + w / 2, d.loc[("mlp", feat), "resize"], w, color=col, hatch="///", edgecolor="white", alpha=0.65)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in groups])
    ax.set_ylim(0.7, 0.95); ax.set_ylabel("AUROC (MLP, grouped 5-fold CV)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#555555", label="Native crop"), Patch(facecolor="#555555", hatch="///", alpha=0.65, label="Resized")],
              loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_resize.png")


def prompts():
    """Test 3: ModelScope clips from the training prompts vs 36 new prompts (stored result, MLP and ResNet-18)."""
    d = pd.read_csv(RESULTS / "fresh_prompts.csv").set_index("model")
    groups = [("mlp_pixel", "Pixel", GREY), ("mlp_all", "Detail", BLUE), ("mlp_fusion", "Combined", RED), ("resnet18", "ResNet-18", PURPLE)]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.36
    for i, (m, _, col) in enumerate(groups):
        ax.bar(i - w / 2, d.loc[m, "auroc_old_prompts"], w, color=col, edgecolor="white")
        ax.bar(i + w / 2, d.loc[m, "auroc_new_prompts"], w, color=col, hatch="///", edgecolor="white", alpha=0.65)
    ax.axhline(0.5, color="k", lw=1.2, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in groups], fontsize=17)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC, ModelScope vs real")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#555555", label="Old prompts"), Patch(facecolor="#555555", hatch="///", alpha=0.65, label="New prompts")],
              loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=16)
    save(fig, "fig_prompts.png")


def fusion():
    """Test 4: ResNet-18 alone, our detail MLP alone, and the two fused (stored result)."""
    d = pd.read_csv(RESULTS / "late_fusion.csv").set_index("set")
    sets = [("seen", "Seen"), ("modelscope", "ModelScope\n(unseen)"), ("wan", "Wan2.1\n(unseen)")]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    w = 0.26
    for j, (col_, color, lab) in enumerate([("resnet18", PURPLE, "ResNet-18"), ("frequency_mlp", BLUE, "Detail MLP"), ("fused", ORANGE, "Both fused")]):
        ax.bar(np.arange(3) + (j - 1) * w, [d.loc[s, col_] for s, _ in sets], w, color=color, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in sets])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (held-out clips)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, ncol=3, fontsize=16, columnspacing=1.0, handlelength=1.0)
    save(fig, "fig_fusion.png")


def temporal_order():
    p = RESULTS / "temporal_order.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.variant == "shuffled"]
    sets = [("seen", "Seen"), ("ModelScope (unseen)", "ModelScope"), ("Wan2.1 (unseen)", "Wan2.1")]
    models_ = [("mlp_spatial", GREY, "Spatial only (0 by design)"), ("mlp_temporal", GREEN, "Temporal only"), ("mlp_all", BLUE, "Spatial + temporal")]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
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
