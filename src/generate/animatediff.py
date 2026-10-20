"""AnimateDiff (SD1.5 + motion adapter v1-5-2) text-to-video. Seen generator."""
import argparse

import torch

from src.config import RAW_FAKE
from src.generate.common import NEGATIVE, run_jobs, t2v_jobs, to_uint8

ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
BASE = "emilianJR/epiCRealism"
W, H, FRAMES, FPS, STEPS = 512, 512, 16, 8, 25


def main(n: int) -> None:
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
    adapter = MotionAdapter.from_pretrained(ADAPTER, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(BASE, motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_pretrained(
        BASE, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1)
    pipe.to("cuda")
    pipe.enable_vae_slicing()

    def gen(job):
        out = pipe(prompt=job.prompt + ", high quality, photorealistic", negative_prompt=NEGATIVE,
                   width=W, height=H, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=7.5,
                   generator=torch.Generator("cuda").manual_seed(job.seed), output_type="np").frames[0]
        return to_uint8(out), FPS

    run_jobs(t2v_jobs("animatediff", n), gen, RAW_FAKE / "animatediff")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    main(ap.parse_args().n)
