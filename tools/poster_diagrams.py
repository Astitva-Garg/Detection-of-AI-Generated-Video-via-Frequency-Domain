"""The two diagrams of the poster (workflow and architecture), drawn with matplotlib at printed size.
The architecture diagram contains the real curves of one held-out clip and the shipped model's verdict for it.

    python -m tools.poster_diagrams
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

from src.config import CLIPS, FPS, N_FRAMES, PROJECT_ROOT, RESULTS
from src.data.video_io import read_frames
from src.features.spatial import power_spectrum, radial_spectrum, to_luma
from src.features.temporal import clip_features

OUT = PROJECT_ROOT / "poster" / "figs"
NAVY, RED, BLUE, GREEN = "#00008b", "#c9184a", "#2a6f97", "#3d9970"
CLIP = "ltx_0001"          # the clip shown in the architecture diagram (a held-out test clip)


def _style():
    plt.rcParams.update({"font.family": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
                         "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def _box(ax, x, y, w, h, fc="#ffffff", ec=NAVY, lw=2.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec=ec, lw=lw))


def _arrow(ax, p0, p1, color=NAVY, lw=3.2):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=24,
                                                      shrinkA=0, shrinkB=0))


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


def workflow():
    """Figure 2: the seven steps."""
    _style()
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    fig.subplots_adjust(0.003, 0.003, 0.997, 0.997)
    ax.set_xlim(0, 104); ax.set_ylim(0, 57); ax.axis("off")
    steps = [
        ("1", "COLLECT", ["432 real (Pexels stock,", "YOLO-filtered); 552 AI:", "LTX, AnimateDiff, SVD,", "ModelScope, Wan2.1;", "60 prompts, 18 scenes"]),
        ("2", "NORMALISE", ["14 frames per clip,", "256 x 256 native crop,", "H.264 CRF 18; split by", "source group 70/15/15;", "ModelScope, Wan test-only"]),
        ("3", "EXTRACT", ["grey frame, Hann window,", "2-D FFT: radial + noise-", "residual spectra (48 bins)", "and drift over frames:", "263 frequency, 74 pixel"]),
        ("4", "TRAIN", ["MLP (263-128-64-1) on", "frequency, pixel or both;", "ResNet-18 on raw frames;", "weighted BCE, AdamW,", "cosine; best val. AUROC"]),
        ("5", "EVALUATE", ["AUROC on 267 held-out", "clips (65 real), 2 unseen", "generators, 250 public", "clips; paired bootstrap,", "5 seeds, grouped 5-fold CV"]),
        ("6", "EXTRA CHECKS", ["Resize vs native crop; frame", "reversal and shuffle; 72 clips", "from 36 unseen prompts;", "ResNet-18 + frequency", "late fusion (stacker on val.)"]),
        ("7", "REPRODUCE", ["Every score and figure is", "rebuilt from the shared", "data and code, and a", "clean re-run gives the", "same results (GPU noise aside)"]),
    ]
    w1, g1 = 22.9, 3.73
    w2, g2 = 31.0, 5.0
    h = 25.5
    pos = [(0.5 + i * (w1 + g1), 30.5, w1) for i in range(4)] + [(0.5 + i * (w2 + g2), 1.0, w2) for i in range(3)]
    for (n, title, lines), (x, y, w) in zip(steps, pos):
        hot = title == "EXTRA CHECKS"
        col = RED if hot else NAVY
        _box(ax, x, y, w, h, fc="#fdecef" if hot else "#ffffff", ec=col)
        ax.add_patch(Circle((x + 3.0, y + h - 3.5), 2.4, color=col))
        ax.text(x + 3.0, y + h - 3.5, n, color="white", ha="center", va="center", fontsize=18, fontweight="bold")
        ax.text(x + 6.6, y + h - 3.5, title, ha="left", va="center", fontsize=16.5, fontweight="bold", color=col)
        for k, line in enumerate(lines):
            ax.text(x + 1.3, y + h - 8.6 - k * 3.3, line, ha="left", va="center", fontsize=12.6)
    for i in range(3):
        x0, _, w = pos[i]
        _arrow(ax, (x0 + w + 0.2, 43), (pos[i + 1][0] - 0.2, 43))
    for i in range(4, 6):
        x0, _, w = pos[i]
        _arrow(ax, (x0 + w + 0.2, 13.8), (pos[i + 1][0] - 0.2, 13.8))
    x3, _, w3 = pos[3]
    cx = pos[4][0] + pos[4][2] / 2
    ax.plot([x3 + w3 / 2, x3 + w3 / 2, cx], [30.5, 28.8, 28.8], color=NAVY, lw=3.2, solid_capstyle="butt")
    _arrow(ax, (cx, 28.8), (cx, 26.7))
    _save(fig, "fig_workflow.png")


def architecture():
    """Figure 3: from one clip to a verdict, with the actual curves of that clip."""
    _style()
    fr = read_frames(CLIPS / f"{CLIP}.mp4", N_FRAMES, FPS)
    gray = [to_luma(f) for f in fr]
    spec = np.log(power_spectrum(gray[0]) + 1e-8)
    curve = radial_spectrum(gray[0])
    _, seq = clip_features(fr)
    drift = np.linalg.norm(np.diff(seq[0], axis=0), axis=1)
    pr = pd.read_csv(RESULTS / "predictions.csv")
    p_ai = float(pr[(pr.clip_id == CLIP) & (pr.model == "mlp_fusion")].p_fake.iloc[0])

    W, H = 10.4, 5.1
    X, Y = 104, 51
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.axis("off")

    def inset(x, y, w, h):               # place an axes by the diagram's own coordinates
        return fig.add_axes([x / X, y / Y, w / X, h / Y])

    # input clip: three of its 14 frames (square thumbnails; 1 unit is 2.54 mm in both directions)
    _box(ax, 0.8, 7, 17.4, 40, fc="#f4f4f4", ec="#555")
    ax.text(9.5, 44.2, "A video clip", ha="center", fontsize=14.5, fontweight="bold")
    for k, i in enumerate([0, 6, 13]):
        a = inset(4.3, 32.0 - k * 10.6, 10.4, 10.4)
        a.imshow(fr[i]); a.axis("off")
    ax.text(9.5, 9.0, "14 frames", ha="center", fontsize=14)

    # lane A and B
    lanes = [("A", "Fine detail", "how much detail at each scale", "226 numbers", 30.5, 18.0, BLUE),
             ("B", "Change over time", "does the detail flicker between frames?", "37 numbers", 12.0, 17.0, GREEN)]
    for tag, title, sub, nums, y, h, col in lanes:
        _box(ax, 21.5, y, 40, h, ec=col)
        ax.text(23, y + h - 2.6, f"{tag}  {title}", fontsize=16.5, fontweight="bold", color=col, va="center")
        ax.text(60.3, y + h - 2.6, nums, fontsize=14, fontweight="bold", color=col, ha="right", va="center")
        ax.text(23, y + h - 6.0, sub, fontsize=13.2, va="center")
    a = inset(23.2, 32.2, 9.4, 9.4); a.imshow(spec, cmap="magma"); a.axis("off")
    a = inset(36.5, 32.4, 22.5, 8.0); a.plot(curve, color=BLUE, lw=2.4); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("fine  →  coarse detail", fontsize=11.5, labelpad=1)
    a = inset(25.0, 14.0, 33.5, 8.2); a.plot(range(1, len(drift) + 1), drift, "o-", color=GREEN, lw=2.2, ms=4.5)
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("14 frames, left to right", fontsize=11.5, labelpad=1)

    _box(ax, 21.5, 1.0, 40, 8.6, fc="#f4f4f4", ec="#777")
    ax.text(23, 6.8, "Control: plain pixel statistics", fontsize=14.8, fontweight="bold", color="#555", va="center")
    ax.text(23, 3.4, "brightness, contrast, colour: 74 numbers", fontsize=13, va="center")

    # network and verdict
    _box(ax, 65.5, 20, 20, 22, fc="#e8eef9")
    ax.text(75.5, 38.0, "Small neural", ha="center", fontsize=15.5, fontweight="bold")
    ax.text(75.5, 34.8, "network", ha="center", fontsize=15.5, fontweight="bold")
    ax.text(75.5, 30.4, "3 layers,", ha="center", fontsize=14)
    ax.text(75.5, 27.4, "42 000 weights", ha="center", fontsize=14)
    ax.text(75.5, 22.8, "263 → 128 → 64 → 1", ha="center", fontsize=11.5)
    _box(ax, 89.2, 20, 14.2, 22, fc="#fdecef", ec=RED)
    ax.text(96.3, 38.6, "Chance the clip", ha="center", fontsize=12.6)
    ax.text(96.3, 35.9, "is AI-made", ha="center", fontsize=12.6)
    ax.text(96.3, 29.0, f"{p_ai:.2f}", ha="center", fontsize=30, fontweight="bold", color=RED)
    ax.text(96.3, 24.4, "1 = sure", ha="center", fontsize=11.5)
    ax.text(96.3, 22.0, "0.5 = unsure", ha="center", fontsize=11.5)

    _arrow(ax, (18.4, 38), (21.2, 40), lw=2.6)
    _arrow(ax, (18.4, 27), (21.2, 22), lw=2.6)
    _arrow(ax, (18.4, 16), (21.2, 6), lw=2.6, color="#777")
    _arrow(ax, (61.8, 40), (65.2, 34), lw=3)
    _arrow(ax, (61.8, 22), (65.2, 28), lw=3)
    _arrow(ax, (61.8, 5.5), (65.2, 20.5), lw=2.6, color="#777")
    _arrow(ax, (85.8, 31), (88.9, 31), lw=3.4)
    ax.text(66.5, 15.0, "Frequency model = A + B", ha="left", fontsize=13.5, color=BLUE, fontweight="bold")
    ax.text(66.5, 11.6, "Fusion model = A + B + control", ha="left", fontsize=13.5, color=RED, fontweight="bold")
    ax.text(66.5, 7.2, f"Shown: held-out LTX-Video clip {CLIP}", ha="left", fontsize=11.5, color="#444")
    ax.text(66.5, 4.6, "and the fusion model's answer for it.", ha="left", fontsize=11.5, color="#444")
    _save(fig, "fig_architecture.png")


if __name__ == "__main__":
    workflow()
    architecture()
