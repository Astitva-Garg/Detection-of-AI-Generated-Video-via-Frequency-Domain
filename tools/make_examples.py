"""Choose the example clips in examples/: 3 held-out test clips per source (real + 5 generators).

Clips already in examples/ are kept; the rest are drawn at random with the project seed from the held-out test split
(main clips, not the extra training windows), so the choice is not hand-picked. Also writes examples/README.md with
the truth and, if the models are available, every verdict (including the wrong ones).

    python -m tools.make_examples
"""
import shutil

import numpy as np
import pandas as pd

from src.config import DATA_ROOT, PROJECT_ROOT, SEED

EX = PROJECT_ROOT / "examples"
PER = 3


def main():
    man = pd.read_csv(DATA_ROOT / "manifest_crop.csv")
    man = man[(man.split == "test") & ~man.aug.astype(bool)]
    rng = np.random.default_rng(SEED)
    EX.mkdir(exist_ok=True)
    have = {p.name.split("__", 1)[1][:-4]: p for p in EX.glob("*.mp4")}
    for gen in ["real", "ltx", "animatediff", "svd", "modelscope", "wan"]:
        ids = sorted(man[man.generator == gen].clip_id)
        keep = [i for i in ids if i in have]
        need = PER - len(keep)
        pool = [i for i in ids if i not in have]
        for cid in (list(rng.choice(pool, need, replace=False)) if need > 0 else []):
            prefix = "real" if gen == "real" else f"ai_{gen}"
            shutil.copy(DATA_ROOT / "clips_crop" / f"{cid}.mp4", EX / f"{prefix}__{cid}.mp4")
    rows = []
    try:
        from src.predict import Predictor
        P = Predictor()
    except Exception as e:  # models or detector missing
        P = None
        print("no verdicts:", e)
    m = man.set_index("clip_id")
    for p in sorted(EX.glob("*.mp4")):
        cid = p.name.split("__", 1)[1][:-4]
        r = {"file": p.name, "truth": "real" if m.loc[cid, "label"] == 0 else "AI", "source": m.loc[cid, "generator"],
             "scene": m.loc[cid, "scene"]}
        if P is not None:
            res = P.predict(p, "mlp_fusion")
            r["P(AI), MLP fusion"] = round(float(res.p_fake), 3)
            r["verdict"] = "AI" if res.p_fake >= 0.5 else "real"
            r["correct"] = "yes" if r["verdict"] == r["truth"] else "NO"
        rows.append(r)
    df = pd.DataFrame(rows)
    txt = ("# examples/\n\nEighteen clips for the app and for `python -m src.demo_live --video <file>`: three per source "
           "(real, LTX-Video, AnimateDiff, SVD, ModelScope, Wan2.1). All are **held-out test clips** (never used for training or "
           "model selection), already normalised (14 frames, 256 x 256 native crop, H.264 CRF 18). They were drawn at random with "
           "seed 643 by `tools/make_examples.py`, not hand-picked. ModelScope and Wan2.1 were never seen in training.\n\n"
           "Verdicts below are from the shipped fusion MLP; wrong ones are kept in.\n\n"
           + df.to_markdown(index=False) + "\n")
    (EX / "README.md").write_text(txt, encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
