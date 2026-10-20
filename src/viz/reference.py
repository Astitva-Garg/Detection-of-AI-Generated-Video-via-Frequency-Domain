"""Class-mean reference spectra for the app, and the bundled example clips."""
import json
import shutil

from src.config import CLIPS, MODELS, N_BINS, PROJECT_ROOT, SEEN, UNSEEN
from src.models import data as D


def build_reference() -> None:
    df = D.clean_table()
    tr = D.select(df, "train")
    ref = {}
    for name, lab in (("real", 0), ("fake", 1)):
        s = tr[tr.label == lab]
        ref[name] = {
            "rad": [float(s[f"sp_rad_{i:02d}_mean"].mean()) for i in range(N_BINS)],
            "res_rad": [float(s[f"sp_res_rad_{i:02d}_mean"].mean()) for i in range(N_BINS)],
            "drift_l2_mean": float(s["drift_l2_mean"].mean()),
        }
    (MODELS / "reference_spectra.json").write_text(json.dumps(ref))
    print("wrote reference_spectra.json")


def build_examples(per_gen: int = 1, n_real: int = 3) -> None:
    """Copy a few *test-split* normalised clips into examples/ for the demo."""
    df = D.clean_table()
    te = df[(df.split == "test") & ~df.aug.astype(bool)]
    out = PROJECT_ROOT / "examples"
    out.mkdir(exist_ok=True)
    picks = list(te[te.generator == "real"].clip_id.sample(min(n_real, (te.generator == "real").sum()), random_state=7))
    for g in SEEN + UNSEEN:
        sub = te[te.generator == g]
        if len(sub):
            picks += list(sub.clip_id.sample(min(per_gen, len(sub)), random_state=7))
    gen_of = df.set_index("clip_id").generator
    for c in picks:
        g = gen_of[c]
        tag = "real" if g == "real" else f"ai_{g}"
        shutil.copy(CLIPS / f"{c}.mp4", out / f"{tag}__{c}.mp4")
    print(f"copied {len(picks)} examples")


if __name__ == "__main__":
    build_reference()
    build_examples()
