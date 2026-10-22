"""Dataset audit: near-duplicates across splits, frozen/looped frames, scene cuts, borders, static clips.

Runs on the normalised clips the models actually see (CLIPS) and writes DATA_ROOT/audit.csv plus a
summary to stdout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.config import CLIPS, DATA_ROOT, FPS, N_FRAMES, SUF
from src.data.video_io import read_frames


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def clip_stats(clip_id: str) -> dict:
    f = read_frames(CLIPS / f"{clip_id}.mp4", N_FRAMES, FPS).astype(np.float32)
    g = f.mean(axis=3)
    diffs = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))          # mean abs change per frame step (0-255)
    med = float(np.median(diffs))
    rows = g.mean(axis=2)                                          # (T, H) row means
    row_std = g.std(axis=2)
    top_flat = int(((row_std[:, :16] < 2.0) & (rows[:, :16] < 20)).all(axis=0).sum())
    bot_flat = int(((row_std[:, -16:] < 2.0) & (rows[:, -16:] < 20)).all(axis=0).sum())
    return {
        "clip_id": clip_id,
        "hash_first": dhash(f[0].astype(np.uint8)), "hash_mid": dhash(f[N_FRAMES // 2].astype(np.uint8)),
        "frozen_steps": int((diffs < 0.3).sum()),                  # consecutive near-identical frames
        "motion_median": med,
        "max_jump_ratio": float(diffs.max() / (med + 1e-3)),      # scene-cut indicator
        "max_jump": float(diffs.max()),
        "mean_luma": float(g.mean()), "spatial_std": float(g.std(axis=(1, 2)).mean()),
        "black_border_rows": top_flat + bot_flat,
        "dark_frames": int((g.mean(axis=(1, 2)) < 8).sum()),
    }


def main(workers: int) -> None:
    man = pd.read_csv(DATA_ROOT / f"manifest{SUF}.csv")
    man = man[~man.aug.astype(bool)]
    with ProcessPoolExecutor(workers) as ex:
        stats = list(ex.map(clip_stats, man.clip_id, chunksize=8))
    df = man.merge(pd.DataFrame(stats), on="clip_id")
    df.to_csv(DATA_ROOT / "audit.csv", index=False)

    print("== per-generator summary")
    print(df.groupby("generator")[["frozen_steps", "motion_median", "max_jump_ratio", "mean_luma",
                                   "spatial_std", "black_border_rows", "dark_frames"]].median().round(2).to_string())
    print("\n== clips with >=3 frozen steps (repeated / looped frames):")
    print(df[df.frozen_steps >= 3].groupby("generator").size().to_string())
    print("\n== near-static clips (median motion < 0.5):", df[df.motion_median < 0.5].groupby("generator").size().to_dict())
    print("== likely scene cuts (one jump > 8x median and > 15):",
          df[(df.max_jump_ratio > 8) & (df.max_jump > 15)].groupby("generator").size().to_dict())
    print("== black border rows > 0:", df[df.black_border_rows > 0].groupby("generator").size().to_dict())
    print("== clips with dark frames:", df[df.dark_frames > 0].groupby("generator").size().to_dict())
    print("== originals shorter than 2 s:", df[df.duration < 2.0].groupby("generator").size().to_dict())

    # near-duplicates: Hamming distance <= 6 on either hash, across different groups
    dups = []
    H = np.array([[r.hash_first, r.hash_mid] for r in df.itertuples()], dtype=np.uint64)
    for i in range(len(df)):
        x = np.bitwise_xor(H[i + 1:], H[i])
        ham = np.array([[bin(int(v)).count("1") for v in row] for row in x]) if len(x) else np.zeros((0, 2))
        for j in np.nonzero((ham.min(axis=1) <= 6))[0]:
            a, b = df.iloc[i], df.iloc[i + 1 + j]
            if a.group_id != b.group_id:
                dups.append({"a": a.clip_id, "b": b.clip_id, "gen_a": a.generator, "gen_b": b.generator,
                             "split_a": a.split, "split_b": b.split, "ham": int(ham[j].min()),
                             "title_a": str(a.prompt)[:50], "title_b": str(b.prompt)[:50]})
    dd = pd.DataFrame(dups)
    dd.to_csv(DATA_ROOT / "audit_duplicates.csv", index=False)
    print(f"\n== near-duplicate pairs across different groups: {len(dd)}")
    if len(dd):
        cross = dd[dd.split_a != dd.split_b]
        print(f"   of which in different splits (leakage risk): {len(cross)}")
        print(dd.groupby(["gen_a", "gen_b"]).size().to_string())
        print(dd.sort_values("ham").head(25).to_string())

    print("\n== scene balance (share of clips per scene, real vs AI seen):")
    seen = df[df.split != "test"]
    tab = pd.crosstab(seen.scene, seen.label, normalize="columns").round(3)
    tab.columns = ["real", "ai"]
    print(tab.assign(ratio=(tab.ai / tab.real).round(2)).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
