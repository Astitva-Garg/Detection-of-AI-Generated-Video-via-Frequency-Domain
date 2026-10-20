"""Run generators one after another, each in its own process (frees GPU memory fully).

Example: python -m src.generate.run_queue ltx=160 animatediff=160 svd=120 modelscope=100 wan=80
"""
import os
import subprocess
import sys
import time

from src.config import RAW_FAKE


def main(specs: list[str]) -> None:
    RAW_FAKE.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1",
           "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    for spec in specs:
        name, n = spec.split("=")
        t0 = time.time()
        print(f"=== {name} n={n}", flush=True)
        with open(RAW_FAKE / f"{name}.log", "a", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-m", f"src.generate.{name}", "--n", n],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
        print(f"=== {name} exit={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
