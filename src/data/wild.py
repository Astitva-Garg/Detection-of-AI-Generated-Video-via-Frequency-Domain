"""In-the-wild stress test set: random clips from public generated-video collections (34data on
Hugging Face: Sora 2, Veo 3, Kling, HunyuanVideo, Wan2). Content is NOT restricted to scenery and the
exact generation settings are unknown — results are reported separately as an out-of-domain test.
"""
import argparse
import random
from pathlib import Path

from src.config import DATA_ROOT

WILD = DATA_ROOT / "wild"
SOURCES = {"sora2": ("gen-videos-sora2", 60), "veo3": ("gen-videos-veo3", 60),
           "kling": ("gen-videos-kling", 30), "hunyuan": ("gen-videos-hunyuanvideo", 60),
           "wan2": ("gen-videos-wan2", 40)}


def download(seed: int = 643) -> None:
    from remotezip import RemoteZip
    rng = random.Random(seed)
    for name, (repo, n) in SOURCES.items():
        out = WILD / name
        out.mkdir(parents=True, exist_ok=True)
        have = {p.name for p in out.glob("*.mp4")}
        url = f"https://huggingface.co/datasets/34data/{repo}/resolve/main/data_001.zip"
        with RemoteZip(url) as z:
            files = sorted((i for i in z.infolist() if i.filename.endswith(".mp4")), key=lambda i: i.filename)
            pick = rng.sample(files, min(n, len(files)))
            for i in pick:
                fn = Path(i.filename).name
                if fn not in have:
                    (out / fn).write_bytes(z.read(i))
        print(name, len(list(out.glob("*.mp4"))), "clips", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    download()
