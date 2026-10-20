"""Shared, resumable generation runner and prompt list."""
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.data.pexels import load_scenes
from src.data.video_io import write_video

NEGATIVE = ("people, person, human, face, hands, animal, bird, text, watermark, logo, car, building, "
            "low quality, blurry, distorted")


@dataclass
class Job:
    id: str
    generator: str
    prompt: str
    scene: str
    seed: int
    group_id: str
    image: str | None = None


def prompt_list() -> list[tuple[str, str]]:
    """(scene, prompt) pairs in a fixed order; index = prompt group id."""
    out = []
    for scene, spec in load_scenes()["scenes"].items():
        out += [(scene, p) for p in spec["prompts"]]
    return out


def t2v_jobs(generator: str, n: int, seed0: int = 0) -> list[Job]:
    prompts = prompt_list()
    jobs = []
    for k in range(n):
        i = k % len(prompts)
        scene, p = prompts[i]
        seed = seed0 + k
        jobs.append(Job(f"{generator}_{k:04d}", generator, p, scene, seed, f"prompt_{i:03d}"))
    return jobs


def _done_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    done = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            done.add(r["id"])
    return done


def _failed_counts(manifest: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] == "failed":
                counts[r["id"]] = counts.get(r["id"], 0) + 1
    return counts


def run_jobs(jobs: list[Job], gen_fn: Callable[[Job], tuple[np.ndarray, int]], out_dir: Path,
             max_failures: int = 2) -> None:
    """Generate each job unless already done; append one JSON line per attempt to manifest.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done, failed = _done_ids(manifest), _failed_counts(manifest)
    todo = [j for j in jobs if j.id not in done and failed.get(j.id, 0) < max_failures]
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)
    for n, job in enumerate(todo, 1):
        t0 = time.time()
        rec = asdict(job)
        try:
            frames, fps = gen_fn(job)
            frames = np.asarray(frames)
            write_video(out_dir / f"{job.id}.mp4", frames, fps, crf=12)
            rec.update(status="ok", fps=fps, n_frames=int(frames.shape[0]),
                       h=int(frames.shape[1]), w=int(frames.shape[2]))
        except Exception as e:  # includes torch.cuda.OutOfMemoryError
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        rec["seconds"] = round(time.time() - t0, 1)
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(todo)}] {job.id} {rec['status']} {rec['seconds']}s", flush=True)


def to_uint8(frames) -> np.ndarray:
    """Pipeline output (list of PIL / float array in [0,1]) -> uint8 THWC."""
    if isinstance(frames, list) and hasattr(frames[0], "size"):
        return np.stack([np.asarray(f.convert("RGB")) for f in frames])
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    return a
