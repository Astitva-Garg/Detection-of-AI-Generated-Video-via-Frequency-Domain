"""ModelScope T2V 1.7B (older UNet3D text-to-video). Unseen generator (test only)."""
import argparse

import torch

from src.config import RAW_FAKE
from src.generate.common import NEGATIVE, run_jobs, t2v_jobs, to_uint8

REPO = "ali-vilab/text-to-video-ms-1.7b"
W, H, FRAMES, FPS, STEPS = 256, 256, 16, 8, 30


def main(n: int) -> None:
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    pipe = DiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    def gen(job):
        out = pipe(prompt=job.prompt, negative_prompt=NEGATIVE, width=W, height=H, num_frames=FRAMES,
                   num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(job.seed),
                   output_type="np").frames[0]
        return to_uint8(out), FPS

    # offset seeds so prompts/seeds differ from the seen generators
    run_jobs(t2v_jobs("modelscope", n, seed0=50_000), gen, RAW_FAKE / "modelscope")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    main(ap.parse_args().n)
