"""Project-wide paths and constants."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("IE643_DATA", r"C:\ie643_data"))
# normalisation variant: "crop" (native-resolution 256 crop; main) or "resize" (short side -> 256, then
# crop; ablation — resizing low-pass filters the frame and erased much of the frequency evidence)
NORM_MODE = os.environ.get("IE643_NORM", "crop")
SUF = "" if NORM_MODE == "resize" else f"_{NORM_MODE}"

N_FRAMES = 16          # frames per clip
FPS = 8                # sampling rate -> 2 s clips
SIZE = 256             # spatial size after resize + centre crop
N_BINS = 48            # radial spectrum bins
SEED = 643

SEEN = ["ltx", "animatediff", "svd"]
UNSEEN = ["wan", "modelscope"]
GENERATORS = SEEN + UNSEEN
# generator native short sides; real clips are pre-downscaled to one of these
REAL_PRE_SIDES = [256, 320, 480, 512, 576]


def paths(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root is not None else DATA_ROOT
    return {
        "raw_real": root / "raw" / "real",
        "raw_fake": root / "raw" / "fake",
        "clips": root / f"clips{SUF}",
        "degraded": root / f"degraded{SUF}",
        "features": root / f"features{SUF}",
        "cache": root / "cache",
    }


RAW_REAL = paths()["raw_real"]
RAW_FAKE = paths()["raw_fake"]
CLIPS = paths()["clips"]
DEGRADED = paths()["degraded"]
FEATURES = paths()["features"]
CACHE = paths()["cache"]
MODELS = PROJECT_ROOT / f"models{SUF}"
RESULTS = PROJECT_ROOT / f"results{SUF}"


def ensure_dirs(root: Path | None = None) -> None:
    for p in paths(root).values():
        p.mkdir(parents=True, exist_ok=True)
    if root is None:
        MODELS.mkdir(exist_ok=True)
        (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
