"""Poster Figure 1: how the spectrum maps are made (top row) and each generator's fingerprint (bottom row).

Every panel is computed from the clips of the dataset: one frame of a real clip, its 2-D log power spectrum, the mean
spectrum over 40 real clips, and, for each generator, the mean spectrum over 40 of its clips minus the real mean.

    python -m tools.poster_spectra
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from src.config import CLIPS, FPS, N_FRAMES, PROJECT_ROOT
from src.data.video_io import read_frames
from src.features.spatial import power_spectrum, to_luma
from src.models import data as D

OUT = PROJECT_ROOT / "poster" / "figs"
EXAMPLE = "real_11072713"      # a held-out real clip, used for panels 1 and 2
GENS = [("ltx", "LTX-Video", "dot grid, less detail"), ("animatediff", "AnimateDiff", "slightly less detail"),
        ("svd", "SVD", "less fine detail"), ("modelscope", "ModelScope*", "more energy"),
        ("wan", "Wan2.1*", "less, cross-shaped")]


def _logspec(frame):
    return np.log(power_spectrum(to_luma(frame)) + 1e-8)


def _mean_spectrum(df, gen, n=40):
    ids = df[df.generator == gen].clip_id.sample(min(n, (df.generator == gen).sum()), random_state=0)
    acc = 0
    for c in ids:
        fr = read_frames(CLIPS / f"{c}.mp4", N_FRAMES, FPS)
        acc = acc + np.mean([_logspec(f) for f in fr[::4]], axis=0)       # frames 0, 4, 8, 12
    return acc / len(ids)


def spectra():
    plt.rcParams.update({"font.family": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"], "savefig.dpi": 200})
    df = D.clean_table()
    df = df[~df.aug.astype(bool)]
    frame = read_frames(CLIPS / f"{EXAMPLE}.mp4", N_FRAMES, FPS)[0]
    one = _logspec(frame)
    real = _mean_spectrum(df, "real")
    diffs = {g: _mean_spectrum(df, g) - real for g, _, _ in GENS}
    v = np.percentile(np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 99)
    norm = Normalize(-v, v)

    W, H = 10.4, 6.0
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 104); bg.set_ylim(0, 60); bg.axis("off")

    def panel(x, y, w, h):
        a = fig.add_axes([x / 104, y / 60, w / 104, h / 60]); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(1.2)
        return a

    # top row: how a map is made
    top = [("1  One frame", "of a real clip", frame, None), ("2  Its spectrum", "log power of the 2-D FFT", one, "magma"),
           ("3  Average spectrum", "of 40 real clips", real, "magma"), ("4  LTX minus real", "average of 40 LTX clips", diffs["ltx"], "RdBu_r")]
    xs = [1.5, 29.3, 57.1, 84.9]
    for (title, sub, img, cmap), x in zip(top, xs):
        a = panel(x, 33.0, 17.4, 17.4)
        if cmap is None:
            a.imshow(img)
        elif cmap == "RdBu_r":
            a.imshow(img, cmap=cmap, norm=norm)
        else:
            a.imshow(img, cmap=cmap)
        bg.text(x + 8.7, 57.6, title, ha="center", fontsize=15, fontweight="bold")
        bg.text(x + 8.7, 54.5, sub, ha="center", fontsize=12.5)
        if cmap == "magma" and title.startswith("2"):
            a.annotate("coarse\nstructure", xy=(128, 128), xytext=(128, 205), color="white", ha="center", fontsize=11,
                       arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
            a.annotate("fine\ndetail", xy=(252, 128), xytext=(205, 45), color="white", ha="center", fontsize=11,
                       arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    for x in (19.3, 47.1, 74.9):
        bg.annotate("", xy=(x + 8.2, 41.7), xytext=(x + 0.4, 41.7),
                    arrowprops=dict(arrowstyle="-|>", color="#00008b", lw=2.6, mutation_scale=20))
    bg.text(52, 30.9, "Same recipe for every generator: its average spectrum minus the real average", ha="center",
            fontsize=13.5, fontweight="bold", color="#00008b")

    # bottom row: the five generators
    xs2 = [1.5, 21.7, 41.9, 62.1, 82.3]
    for (g, name, note), x in zip(GENS, xs2):
        a = panel(x, 10.0, 17.4, 17.4)
        a.imshow(diffs[g], cmap="RdBu_r", norm=norm)
        bg.text(x + 8.7, 28.0, name, ha="center", fontsize=14, fontweight="bold")
        bg.text(x + 8.7, 7.5, note, ha="center", fontsize=11.5)
    cax = fig.add_axes([22 / 104, 3.6 / 60, 60 / 104, 1.3 / 60])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks([-v, 0, v]); cb.set_ticklabels(["less energy than real", "same", "more energy than real"], fontsize=11)
    cb.outline.set_linewidth(0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_spectra.png", facecolor="white")
    plt.close(fig)
    print("saved fig_spectra.png")


if __name__ == "__main__":
    spectra()
