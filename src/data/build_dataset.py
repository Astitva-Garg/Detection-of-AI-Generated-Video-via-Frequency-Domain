"""Normalise all allowed raw clips, write the labelled manifest, splits and degraded copies.

Incremental: rerunning only processes clips not yet in the manifest.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.config import (CLIPS, DATA_ROOT, SUF, DEGRADED, FPS, N_FRAMES, RAW_FAKE, RAW_REAL, REAL_PRE_SIDES,
                        SEEN, UNSEEN)
from src.data.degrade import DEGRADATIONS, apply_degradation
from src.data.normalize import normalize_clip
from src.data.pexels import _word_re, load_scenes
from src.data.splits import assign_split, stable_choice
from src.data.video_io import read_frames, write_video

MANIFEST = DATA_ROOT / f"manifest{SUF}.csv"
MANIFEST_DEG = DATA_ROOT / f"manifest_degraded{SUF}.csv"


def _content() -> dict[str, tuple[bool, float]]:
    p = DATA_ROOT / "content_filter.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {r.path: (bool(r.allowed), float(r.frac)) for r in df.itertuples()}


def candidate_rows() -> list[dict]:
    content = _content()
    rows = []
    meta = pd.read_csv(RAW_REAL / "meta.csv", dtype={"pexels_id": str}).drop_duplicates("pexels_id")
    excluded = _word_re(load_scenes()["exclude"])  # re-applied so later additions (e.g. CG terms) take effect
    for r in meta.itertuples():
        if excluded.search(r.title.replace("· Free Stock Video", "")):
            continue
        src = RAW_REAL / f"{r.pexels_id}.mp4"
        if not src.exists() or str(src) not in content:
            continue
        ok, frac = content[str(src)]
        if not ok:
            continue
        g = f"real_{r.pexels_id}"
        split = assign_split(g)
        pre = stable_choice(r.pexels_id, REAL_PRE_SIDES)
        base = dict(label=0, generator="real", group_id=g, split=split, scene=r.scene, prompt=r.title,
                    source=r.url, src=str(src), content_frac=frac, pre=pre)
        start = 1.0 if r.duration >= 4 else 0.0
        rows.append({**base, "clip_id": g, "start": start, "aug": False})
        if split == "train" and r.duration >= 7:
            rows.append({**base, "clip_id": g + "_w2", "start": 4.0, "aug": True})
    for gen in SEEN + UNSEEN:
        man = RAW_FAKE / gen / "manifest.jsonl"
        if not man.exists():
            continue
        for line in man.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            src = RAW_FAKE / gen / f"{j['id']}.mp4"
            if j["status"] != "ok" or not src.exists() or str(src) not in content:
                continue
            ok, frac = content[str(src)]
            if not ok:
                continue
            split = "test" if gen in UNSEEN else assign_split(j["group_id"])
            rows.append(dict(clip_id=j["id"], label=1, generator=gen, group_id=j["group_id"], split=split,
                             scene=j["scene"], prompt=j["prompt"], source=j.get("image") or gen,
                             src=str(src), content_frac=frac, pre=None, start=0.0, aug=False))
    return rows


def _normalise(row: dict) -> dict:
    meta = normalize_clip(row["src"], CLIPS / f"{row['clip_id']}.mp4", row["pre"], row["start"])
    return {**row, **meta}


def _degrade(row: dict, names: list[str]) -> list[dict]:
    frames = read_frames(CLIPS / f"{row['clip_id']}.mp4", N_FRAMES, FPS)
    out = []
    for name in names:
        dst = DEGRADED / name / f"{row['clip_id']}.mp4"
        if not dst.exists():
            write_video(dst, apply_degradation(frames, name), FPS, crf=18)
        out.append({"clip_id": row["clip_id"], "degradation": name, "path": str(dst)})
    return out


def main(workers: int) -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=["clip_id"])
    cands = candidate_rows()
    keep = {r["clip_id"] for r in cands}
    dropped = old[~old.clip_id.isin(keep)]
    if len(dropped):
        print(f"dropping {len(dropped)} clips that no longer qualify:", list(dropped.clip_id)[:20], flush=True)
    old = old[old.clip_id.isin(keep)]
    have = set(old.clip_id)
    todo = [r for r in cands if r["clip_id"] not in have]
    print(f"{len(have)} in manifest, {len(todo)} new clips to normalise", flush=True)
    new = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_normalise, r): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                new.append(f.result())
            except Exception as e:
                print("failed", futs[f]["clip_id"], e)
            if i % 50 == 0:
                print(f"normalised {i}/{len(todo)}", flush=True)
    man = pd.concat([old, pd.DataFrame(new)], ignore_index=True) if new else old
    man = man.drop(columns=[c for c in ("src", "pre", "n_src_frames") if c in man.columns])
    man.to_csv(MANIFEST, index=False)
    print(man.groupby(["generator", "split"]).size().unstack(fill_value=0).to_string())

    # degraded copies: every test clip gets all degradations; every non-aug train clip gets one
    # (for degradation-aware training)
    old_d = pd.read_csv(MANIFEST_DEG) if MANIFEST_DEG.exists() else pd.DataFrame(columns=["clip_id", "degradation"])
    old_d = old_d[old_d.clip_id.isin(set(man.clip_id))]
    have_d = set(zip(old_d.clip_id, old_d.degradation))
    jobs = []
    for r in man[~man.aug.astype(bool)].to_dict("records"):
        if r["split"] == "test":
            names = list(DEGRADATIONS)
        elif r["split"] == "train":
            names = [stable_choice(r["clip_id"], list(DEGRADATIONS))]
        else:
            continue
        names = [n for n in names if (r["clip_id"], n) not in have_d]
        if names:
            jobs.append((r, names))
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(as_completed([ex.submit(_degrade, r, n) for r, n in jobs]), 1):
            try:
                rows += res.result()
            except Exception as e:
                print("degrade failed", e)
            if i % 100 == 0:
                print(f"degraded {i}/{len(jobs)}", flush=True)
    deg = pd.concat([old_d, pd.DataFrame(rows)], ignore_index=True) if rows else old_d
    deg.to_csv(MANIFEST_DEG, index=False)
    print(f"degraded rows: {len(deg)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
