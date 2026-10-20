"""Select and download real camera-captured clips from Pexels.

Pexels paused new API keys, so we use the public `jovianzm/Pexels-400k` metadata parquet
(title, duration, direct `pexels.com/download/video/<id>` link) and download the files directly.
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.config import PROJECT_ROOT, RAW_REAL

META_URL = "https://huggingface.co/datasets/jovianzm/Pexels-400k/resolve/main/pexels_400k.parquet"
UA = {"User-Agent": "Mozilla/5.0 (IE643 course project; academic dataset)"}


def load_scenes(path: Path = PROJECT_ROOT / "configs" / "scenes.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _word_re(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


def select_titles(df: pd.DataFrame, scenes: dict, min_dur: int = 5, max_dur: int = 60) -> pd.DataFrame:
    """Keep SFW clips whose title matches a scene keyword and contains no excluded word."""
    df = df[(df["duration"] >= min_dur) & (df["duration"] <= max_dur) & df["sfw"].astype(bool)].copy()
    title = df["title"].str.replace("· Free Stock Video", "", regex=False)
    df = df[~title.str.contains(_word_re(scenes["exclude"])) &
            ~df["title"].str.startswith("Free stock video of")].copy()
    title = df["title"].str.replace("· Free Stock Video", "", regex=False)
    df["scene"] = None
    for scene, spec in scenes["scenes"].items():
        hit = title.str.contains(_word_re(spec["keywords"])) & df["scene"].isna()
        df.loc[hit, "scene"] = scene
    return df[df["scene"].notna()].reset_index(drop=True)


def cap_per_scene(df: pd.DataFrame, cap: int, seed: int = 643) -> pd.DataFrame:
    return (df.sample(frac=1.0, random_state=seed)
              .groupby("scene", group_keys=False).head(cap)
              .reset_index(drop=True))


def pexels_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def download(url: str, dst: Path, timeout: int = 60) -> None:
    tmp = dst.with_suffix(".part")
    with requests.get(url, headers=UA, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dst)


def main(n: int, cap: int) -> None:
    RAW_REAL.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_REAL.parent / "pexels_400k.parquet"
    if not meta_path.exists():
        print("downloading metadata ...")
        download(META_URL, meta_path, timeout=300)
    df = pd.read_parquet(meta_path)
    sel = cap_per_scene(select_titles(df, load_scenes()), cap)
    sel = sel.sample(frac=1.0, random_state=643).head(n)
    print(f"selected {len(sel)} clips over {sel['scene'].nunique()} scenes")
    print(sel["scene"].value_counts().to_string())

    log = RAW_REAL / "meta.csv"
    done = set(pd.read_csv(log)["pexels_id"].astype(str)) if log.exists() else set()
    for i, row in enumerate(sel.itertuples(), 1):
        pid = pexels_id(row.video)
        dst = RAW_REAL / f"{pid}.mp4"
        if pid in done and dst.exists():
            continue
        try:
            download(row.video, dst)
        except Exception as e:  # network hiccup: skip, rerun resumes
            print(f"[{i}] failed {pid}: {e}")
            continue
        pd.DataFrame([{"pexels_id": pid, "title": row.title, "scene": row.scene,
                       "url": row.video, "duration": row.duration}]).to_csv(
            log, mode="a", header=not log.exists(), index=False)
        print(f"[{i}/{len(sel)}] {pid} {row.scene} {dst.stat().st_size / 1e6:.1f} MB", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=520)
    ap.add_argument("--cap", type=int, default=40)
    a = ap.parse_args()
    main(a.n, a.cap)
