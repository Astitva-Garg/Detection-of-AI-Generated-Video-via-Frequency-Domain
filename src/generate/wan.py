"""Wan2.1 T2V 1.3B. Unseen generator (test only)."""
import argparse

import torch

from src.config import CACHE, RAW_FAKE
from src.generate.common import NEGATIVE, run_jobs, t2v_jobs, to_uint8
from src.generate.text_embed import cache_embeddings

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
W, H, FRAMES, FPS, STEPS = 832, 480, 33, 16, 30


def main(n: int) -> None:
    jobs = t2v_jobs("wan", n, seed0=90_000)
    emb = cache_embeddings(REPO, "UMT5EncoderModel", [j.prompt for j in jobs] + [NEGATIVE],
                           CACHE / "wan_embeds.pt", max_len=512, style="wan")
    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(REPO, vae=vae, text_encoder=None, tokenizer=None,
                                       torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    neg = emb[NEGATIVE][0]

    def gen(job):
        out = pipe(prompt_embeds=emb[job.prompt][0].cuda(), negative_prompt_embeds=neg.cuda(),
                   width=W, height=H, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0,
                   generator=torch.Generator("cuda").manual_seed(job.seed), output_type="np").frames[0]
        return to_uint8(out), FPS

    run_jobs(jobs, gen, RAW_FAKE / "wan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    main(ap.parse_args().n)
