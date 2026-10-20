"""Extract and cache features for all normalised clips and degraded copies."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.config import CLIPS, DATA_ROOT, FEATURES, FPS, N_FRAMES
from src.data.video_io import read_frames
from src.features.temporal import clip_features


def _one(args):
    key, path = args
    try:
        feats, seq = clip_features(read_frames(path, N_FRAMES, FPS))
        return key, feats, seq
    except Exception as e:
        print("feature fail", path, e, flush=True)
        return key, None, None


def _run(items: list[tuple[str, str]], name: str, workers: int) -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    pq, npz = FEATURES / f"{name}.parquet", FEATURES / f"seq_{name}.npz"
    old = pd.read_parquet(pq) if pq.exists() else None
    old_seq = dict(np.load(npz)) if npz.exists() else {}
    have = set(old["key"]) if old is not None else set()
    todo = [it for it in items if it[0] not in have]
    print(f"{name}: {len(have)} cached, {len(todo)} to extract", flush=True)
    if not todo:
        return
    rows, seqs = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (key, feats, seq) in enumerate(ex.map(_one, todo, chunksize=4), 1):
            if feats is not None:
                rows.append({"key": key, **feats})
                seqs[key] = seq
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(pq, index=False)
    old_seq.update(seqs)
    np.savez_compressed(npz, **old_seq)


def main(workers: int) -> None:
    man = pd.read_csv(DATA_ROOT / "manifest.csv")
    _run([(c, str(CLIPS / f"{c}.mp4")) for c in man.clip_id], "clean", workers)
    deg_p = DATA_ROOT / "manifest_degraded.csv"
    if deg_p.exists():
        deg = pd.read_csv(deg_p)
        _run([(f"{r.clip_id}|{r.degradation}", r.path) for r in deg.itertuples()], "degraded", workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
