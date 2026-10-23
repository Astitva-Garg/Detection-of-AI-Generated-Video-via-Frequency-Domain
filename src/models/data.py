"""Load cached features joined with the manifest."""
import numpy as np
import pandas as pd

from src.config import SUF, DATA_ROOT, FEATURES, UNSEEN
from src.features.temporal import group_columns


def manifest() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / f"manifest{SUF}.csv")


def _with_pixel(f: pd.DataFrame) -> pd.DataFrame:
    p = FEATURES / "pixel.parquet"
    if p.exists():
        f = f.merge(pd.read_parquet(p), on="key", how="left")
    return f


def clean_table() -> pd.DataFrame:
    f = _with_pixel(pd.read_parquet(FEATURES / "clean.parquet")).rename(columns={"key": "clip_id"})
    return manifest().merge(f, on="clip_id", how="inner")


def degraded_table() -> pd.DataFrame:
    f = _with_pixel(pd.read_parquet(FEATURES / "degraded.parquet"))
    f[["clip_id", "degradation"]] = f["key"].str.split("|", expand=True)
    return manifest().merge(f.drop(columns="key"), on="clip_id", how="inner")


def feature_cols(df: pd.DataFrame, group: str = "all") -> list[str]:
    return group_columns(df.columns, group)


def select(df: pd.DataFrame, split: str, include_aug: bool = False, generators: list[str] | None = None,
           exclude_unseen: bool = True) -> pd.DataFrame:
    d = df[df.split == split]
    if not include_aug:
        d = d[~d.aug.astype(bool)]
    if exclude_unseen:
        d = d[~d.generator.isin(UNSEEN)]
    if generators is not None:
        d = d[d.generator.isin(generators + ["real"])]
    return d


def xy(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[cols].to_numpy(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df.label.to_numpy(np.int64)


def load_seq(name: str, keys) -> np.ndarray:
    z = np.load(FEATURES / f"seq_{name}.npz")
    return np.stack([z[k] for k in keys]).astype(np.float32)
