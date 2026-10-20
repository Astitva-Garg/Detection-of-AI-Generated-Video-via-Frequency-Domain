"""LTX-Video (2B) text-to-video. Seen generator."""
import argparse

import torch

from src.config import CACHE, RAW_FAKE
from src.generate.common import NEGATIVE, run_jobs, t2v_jobs, to_uint8
from src.generate.text_embed import cache_embeddings

REPO = "Lightricks/LTX-Video"
W, H, FRAMES, FPS, STEPS = 704, 480, 49, 24, 40


def main(n: int) -> None:
    jobs = t2v_jobs("ltx", n)
    emb = cache_embeddings(REPO, "T5EncoderModel", [j.prompt for j in jobs] + [NEGATIVE],
                           CACHE / "ltx_embeds.pt", max_len=128, style="ltx")
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(REPO, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    neg_e, neg_m = emb[NEGATIVE]

    def gen(job):
        e, m = emb[job.prompt]
        out = pipe(prompt_embeds=e.cuda(), prompt_attention_mask=m.cuda(),
                   negative_prompt_embeds=neg_e.cuda(), negative_prompt_attention_mask=neg_m.cuda(),
                   width=W, height=H, num_frames=FRAMES, num_inference_steps=STEPS,
                   generator=torch.Generator("cuda").manual_seed(job.seed), output_type="np").frames[0]
        return to_uint8(out), FPS

    run_jobs(jobs, gen, RAW_FAKE / "ltx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    main(ap.parse_args().n)
