"""Project-wide paths and constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\ie643_data")

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
        "clips": root / "clips",
        "degraded": root / "degraded",
        "features": root / "features",
        "cache": root / "cache",
    }


RAW_REAL = paths()["raw_real"]
RAW_FAKE = paths()["raw_fake"]
CLIPS = paths()["clips"]
DEGRADED = paths()["degraded"]
FEATURES = paths()["features"]
CACHE = paths()["cache"]
MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "results"


def ensure_dirs(root: Path | None = None) -> None:
    for p in paths(root).values():
        p.mkdir(parents=True, exist_ok=True)
    if root is None:
        MODELS.mkdir(exist_ok=True)
        (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
