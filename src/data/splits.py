"""Deterministic group-level train/val/test assignment."""
import hashlib

from src.config import SEED


def assign_split(group_id: str, seed: int = SEED) -> str:
    h = int(hashlib.sha1(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.70 else ("val" if h < 0.85 else "test")


def stable_choice(key: str, options: list, seed: int = SEED):
    h = int(hashlib.sha1(f"{seed}:choice:{key}".encode()).hexdigest()[:8], 16)
    return options[h % len(options)]
