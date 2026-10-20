"""Stable Video Diffusion XT image-to-video, conditioned on frame 0 of real Pexels clips.

Content-matched pairs: each SVD fake shares its first frame (and its group_id, hence its split)
with a real clip, so detection cannot rely on scene content.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import RAW_FAKE, RAW_REAL, SEED
from src.data.video_io import read_frames
from src.generate.common import Job, run_jobs, to_uint8

REPO = "stabilityai/stable-video-diffusion-img2vid-xt"
W, H, FRAMES, FPS, STEPS = 1024, 576, 25, 7, 25


def make_jobs(n: int) -> list[Job]:
    meta = pd.read_csv(RAW_REAL / "meta.csv", dtype={"pexels_id": str}).drop_duplicates("pexels_id")
    meta = meta[[(RAW_REAL / f"{p}.mp4").exists() for p in meta.pexels_id]]
    # spread across scenes
    meta = meta.sample(frac=1.0, random_state=SEED).groupby("scene", group_keys=False).head(max(1, n // 10))
    meta = meta.sample(frac=1.0, random_state=SEED).head(n)
    return [Job(f"svd_{r.pexels_id}", "svd", r.title, r.scene, SEED + i, f"real_{r.pexels_id}",
                image=str(RAW_REAL / f"{r.pexels_id}.mp4")) for i, r in enumerate(meta.itertuples())]


def first_frame(video_path: str) -> Image.Image:
    f = read_frames(video_path, n_frames=1, fps=8)[0]
    img = Image.fromarray(f)
    # centre-crop to 16:9 then resize to model resolution
    w, h = img.size
    tw, th = (w, round(w * 9 / 16)) if w / h < 16 / 9 else (round(h * 16 / 9), h)
    l, t = (w - tw) // 2, (h - th) // 2
    return img.crop((l, t, l + tw, t + th)).resize((W, H), Image.LANCZOS)


def main(n: int) -> None:
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()

    def gen(job):
        out = pipe(first_frame(job.image), width=W, height=H, num_frames=FRAMES,
                   num_inference_steps=STEPS, decode_chunk_size=2, motion_bucket_id=127, fps=FPS,
                   generator=torch.Generator("cpu").manual_seed(job.seed), output_type="np").frames[0]
        return to_uint8(out), FPS

    run_jobs(make_jobs(n), gen, RAW_FAKE / "svd")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    main(ap.parse_args().n)
