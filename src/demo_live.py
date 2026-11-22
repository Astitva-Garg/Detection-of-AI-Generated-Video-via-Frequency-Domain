"""Live, reproducible demonstration on real held-out clips (no cached numbers are reused for the results).

    python -m src.demo_live                      # full demo (about 1-2 minutes on CPU)
    python -m src.demo_live --video my_clip.mp4  # also trace any video you give it, raw file to verdict
    python -m src.demo_live --retrain            # also retrain the feature models from the training features

Steps, each printed and saved under demo_output/:
  1. show a real and a generated held-out clip
  2. extract their spectra live (2D spectrum, radial curves, residual curves, spectral drift)
  3. trace one clip from frames to features to the verdict of every model family
  4. compare pixel-only, frequency-only and fusion models on the same held-out clips (AUROC, bootstrap CI)
  5. check the live numbers against the stored results (reproducibility)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from src.config import CLIPS, FPS, MODELS, N_BINS, N_FRAMES, RESULTS, SEED, SEEN, UNSEEN
from src.data.video_io import read_frames
from src.eval.evaluate import metrics
from src.features.pixel_baseline import clip_pixel_features
from src.features.spatial import frame_features, power_spectrum, radial_spectrum, residual, to_luma
from src.features.temporal import clip_features
from src.models import data as D
from src.models.cnn import SpecTempCNN, cnn_predict
from src.models.classic import fit_model

OUT = Path("demo_output")
X_FREQ = (np.arange(N_BINS) + 0.5) / N_BINS
FAMILIES = {"pixel": "pixel", "frequency": "all", "fusion": "fusion"}   # label -> feature-group name


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def _extract(path: str):
    """Live feature extraction for one stored clip: exactly what produced the cached features."""
    frames = read_frames(path, N_FRAMES, FPS)
    feats, seq = clip_features(frames)
    feats.update(clip_pixel_features(frames))
    return feats, seq


def load_model(name: str):
    e = joblib.load(MODELS / f"{name}.joblib")
    return e["model"], e["cols"]


def load_cnn():
    import torch
    ck = torch.load(MODELS / "cnn.pt", weights_only=False)
    m = SpecTempCNN(ck["in_ch"], ck["n_bins"])
    m.load_state_dict(ck["state"])
    return m.eval(), (ck["mu"], ck["sd"])


def score(name: str, feats: pd.DataFrame) -> np.ndarray:
    m, cols = load_model(name)
    X = np.nan_to_num(feats[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- 1. examples
def pick_examples(man: pd.DataFrame, real_id=None, fake_id=None):
    te = man[(man.split == "test") & ~man.aug.astype(bool)]
    rng = np.random.default_rng(SEED)
    real = real_id or rng.choice(te[te.generator == "real"].clip_id.to_numpy())
    pool = te[te.generator.isin(SEEN)]
    fake = fake_id or rng.choice(pool[pool.generator == "ltx"].clip_id.to_numpy())
    return real, fake


def show_examples(ids: dict, man: pd.DataFrame) -> dict:
    head("1. HELD-OUT EXAMPLES (test split, never used for training or model selection)")
    frames = {}
    fig, ax = plt.subplots(2, 4, figsize=(11, 6))
    for r, (tag, cid) in enumerate(ids.items()):
        row = man[man.clip_id == cid].iloc[0]
        f = read_frames(CLIPS / f"{cid}.mp4", N_FRAMES, FPS)
        frames[tag] = f
        print(f"  {tag:5s} clip_id={cid}  source={row.generator}  scene={row.scene}  split={row.split}  "
              f"frames={f.shape}  label={'AI' if row.label == 1 else 'real'}")
        for j, i in enumerate((0, 4, 8, 12)):
            ax[r, j].imshow(f[i])
            ax[r, j].set_title(f"{tag} ({row.generator}) frame {i}", fontsize=9)
            ax[r, j].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "01_examples.png", dpi=130)
    plt.close(fig)
    return frames


# ---------------------------------------------------------------- 2. spectra
def show_spectra(frames: dict) -> None:
    head("2. SPECTRA EXTRACTED LIVE FROM THOSE CLIPS")
    cols = {"real": "#2a6f97", "AI": "#c9184a"}
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for r, (tag, f) in enumerate(frames.items()):
        gray = np.stack([to_luma(x) for x in f])
        s2d = np.mean([np.log(power_spectrum(g) + 1e-8) for g in gray[::4]], axis=0)
        rad = np.stack([radial_spectrum(g) for g in gray])
        res = np.stack([radial_spectrum(residual(g)) for g in gray])
        drift = np.linalg.norm(np.diff(rad, axis=0), axis=1)
        ff = frame_features(gray[0])
        print(f"  {tag}: frame 0 -> windowed FFT map {s2d.shape}, radial curve {rad.shape[1]} bins, "
              f"slope {ff['slope']:+.2f}, energy beyond 75% Nyquist {ff['hf_ratio_75']:.4f}, "
              f"mean L2 drift {drift.mean():.3f}")
        ax[r, 0].imshow(s2d, cmap="magma")
        ax[r, 0].set_title(f"{tag}: mean 2D log power spectrum")
        ax[r, 0].axis("off")
        ax[r, 1].plot(X_FREQ, rad.mean(0), color=cols[tag])
        ax[r, 1].set_title("radial curve (frame), mean over frames")
        ax[r, 2].plot(X_FREQ, res.mean(0), color=cols[tag])
        ax[r, 2].set_title("radial curve (noise residual)")
        ax[r, 3].plot(range(1, len(drift) + 1), drift, marker="o", color=cols[tag])
        ax[r, 3].set_title("frame-to-frame spectral drift (L2)")
        for a in ax[r, 1:3]:
            a.set_xlabel("spatial frequency (fraction of Nyquist)")
        ax[r, 3].set_xlabel("frame step")
    fig.tight_layout()
    fig.savefig(OUT / "02_spectra_live.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 3. trace
def trace_stored(cid: str, man: pd.DataFrame, cached: pd.DataFrame) -> None:
    head(f"3. TRACE ONE CLIP FROM FRAMES TO VERDICT: {cid}")
    row = man[man.clip_id == cid].iloc[0]
    path = CLIPS / f"{cid}.mp4"
    frames = read_frames(path, N_FRAMES, FPS)
    gray = np.stack([to_luma(x) for x in frames])
    print(f"  file                      {path.name}  (true label: {'AI' if row.label == 1 else 'real'}, source {row.generator})")
    print(f"  14 frames read            {frames.shape}  uint8")
    print(f"  grey frames               {gray.shape}  range {gray.min():.2f}-{gray.max():.2f}")
    p0 = power_spectrum(gray[0])
    print(f"  frame 0 power spectrum    {p0.shape}  (window, mean removal, 2D FFT, squared magnitude)")
    r0 = radial_spectrum(gray[0])
    print(f"  frame 0 radial curve      {r0.shape}  first 5 bins {np.round(r0[:5], 2)}")
    print(f"  frame 0 residual curve    {radial_spectrum(residual(gray[0])).shape}")
    print(f"  per-frame features        {len(frame_features(gray[0]))} values per frame")
    feats, seq = _extract(str(path))
    fr = {k: v for k, v in feats.items() if k.startswith(("sp_", "drift_", "rt_"))}
    px = {k: v for k, v in feats.items() if k.startswith("px_")}
    n_sp = sum(k.startswith("sp_") for k in fr)
    n_dr = sum(k.startswith("drift_") for k in fr)
    n_rt = sum(k.startswith("rt_") for k in fr)
    print(f"  clip features             {n_sp} spatial + {n_dr} drift + {n_rt} residual-temporal = {len(fr)} frequency; "
          f"{len(px)} pixel; fusion {len(fr) + len(px)}")
    print(f"  CNN input                 {seq.shape}  (2 channels x 14 frames x 48 bins)")
    cached_row = cached[cached.clip_id == cid]
    if len(cached_row):
        keys = list(fr)
        diff = np.abs(cached_row[keys].to_numpy(float)[0] - np.array([fr[k] for k in keys]))
        print(f"  live vs cached features   max absolute difference {np.nanmax(diff):.2e} (same code, same clip)")
    one = pd.DataFrame([feats])
    print("\n  model verdicts on this one clip (P(AI); threshold 0.5)")
    print(f"  {'model':<24s}{'features':<12s}{'P(AI)':>8s}   verdict")
    for fam, group in FAMILIES.items():
        for mdl in ("lr", "lgbm", "mlp"):
            p = score(f"{mdl}_{group}", one)[0]
            print(f"  {mdl + ' (' + fam + ')':<24s}{group:<12s}{p:8.3f}   {'AI' if p >= 0.5 else 'real'}")
    cnn, stats = load_cnn()
    pc = float(cnn_predict(cnn, stats, seq[None])[0])
    print(f"  {'cnn (frequency)':<24s}{'spectra':<12s}{pc:8.3f}   {'AI' if pc >= 0.5 else 'real'}")
    m, cols = load_model("lgbm_all")
    X = np.nan_to_num(one[cols].to_numpy(np.float32))
    contrib = m.predict(X, pred_contrib=True)[0][:-1]
    top = np.argsort(-np.abs(contrib))[:8]
    print("\n  why (LightGBM, frequency features): top feature contributions, + pushes towards AI")
    for i in top:
        print(f"    {cols[i]:<28s}{contrib[i]:+.3f}")


def trace_video(path: str) -> None:
    from src.data.normalize import normalize_frames
    from src.data.video_io import h264_roundtrip, probe, read_frames_stride
    head(f"3b. TRACE YOUR OWN VIDEO FROM THE RAW FILE: {path}")
    meta = probe(path)
    raw, eff = read_frames_stride(path, N_FRAMES, FPS)
    print(f"  raw file                  {meta['w']}x{meta['h']}, {meta['fps']:.1f} fps, {meta['duration']:.1f} s")
    print(f"  stride sampling           14 frames, effective {eff:.2f} fps -> {raw.shape}")
    pre = 512 if min(raw.shape[1:3]) > 576 else None
    frames = h264_roundtrip(normalize_frames(raw, pre), FPS, crf=18)
    print(f"  normalised                {frames.shape}  (pre-shrink {pre}, 256 centre crop, H.264 CRF 18)")
    feats, seq = clip_features(frames)
    feats.update(clip_pixel_features(frames))
    one = pd.DataFrame([feats])
    for fam, group in FAMILIES.items():
        for mdl in ("lr", "lgbm", "mlp"):
            p = score(f"{mdl}_{group}", one)[0]
            print(f"  {mdl + ' (' + fam + ')':<24s}P(AI) {p:.3f}   {'AI' if p >= 0.5 else 'real'}")
    cnn, stats = load_cnn()
    pc = float(cnn_predict(cnn, stats, seq[None])[0])
    print(f"  {'cnn (frequency)':<24s}P(AI) {pc:.3f}   {'AI' if pc >= 0.5 else 'real'}")


# ---------------------------------------------------------------- 4. comparison
def boot(y, scores: dict, n=500):
    rng = np.random.default_rng(SEED)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    out = {k: np.empty(n) for k in scores}
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k][b] = roc_auc_score(y[idx], s[idx])
    return out


def compare(man: pd.DataFrame, workers: int) -> pd.DataFrame:
    head("4. PIXEL-ONLY vs FREQUENCY-ONLY vs FUSION ON THE SAME HELD-OUT CLIPS (features extracted live)")
    te = man[(man.split == "test") & ~man.aug.astype(bool)].reset_index(drop=True)
    paths = [str(CLIPS / f"{c}.mp4") for c in te.clip_id]
    print(f"  extracting features live for {len(te)} held-out test clips ({workers} workers)...", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_extract, paths, chunksize=4))
    feats = pd.DataFrame([r[0] for r in res])
    seqs = np.stack([r[1] for r in res])
    real = (te.generator == "real").to_numpy()
    sets = {"seen (real+LTX+AnimateDiff+SVD)": ~te.generator.isin(UNSEEN).to_numpy(),
            "ModelScope (unseen) vs real": real | (te.generator == "modelscope").to_numpy(),
            "Wan2.1 (unseen) vs real": real | (te.generator == "wan").to_numpy()}
    y = te.label.to_numpy().astype(int)
    scores = {}
    for fam, group in FAMILIES.items():
        for mdl in ("lr", "lgbm", "mlp"):
            scores[f"{mdl}|{fam}"] = score(f"{mdl}_{group}", feats)
    cnn, stats = load_cnn()
    scores["cnn|frequency"] = cnn_predict(cnn, stats, seqs)
    rows = []
    for sname, mask in sets.items():
        ym = y[mask]
        bs = boot(ym, {k: v[mask] for k, v in scores.items()})
        for k, v in scores.items():
            mdl, fam = k.split("|")
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            m = metrics(ym, v[mask])
            rows.append({"set": sname, "model": mdl, "features": fam, "n": len(ym), "auroc": m["auroc"],
                         "ci_low": lo, "ci_high": hi, "acc@0.5": m["acc"], "tpr_fake@0.5": m["tpr_fake"]})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "04_comparison.csv", index=False)
    df.round(3).to_csv(RESULTS / "live_comparison.csv", index=False)   # tracked copy (the poster figure reads it)
    for sname in sets:
        sub = df[df.set == sname]
        n = int(sub.n.iloc[0])
        print(f"\n  {sname}  (n = {n} clips)")
        print(f"  {'model':<8s}{'pixel-only':>22s}{'frequency-only':>22s}{'fusion':>22s}")
        for mdl in ("lr", "lgbm", "mlp"):
            cells = []
            for fam in FAMILIES:
                r = sub[(sub.model == mdl) & (sub.features == fam)].iloc[0]
                cells.append(f"{r.auroc:.3f} [{r.ci_low:.2f},{r.ci_high:.2f}]")
            print(f"  {mdl:<8s}{cells[0]:>22s}{cells[1]:>22s}{cells[2]:>22s}")
        r = sub[(sub.model == "cnn")].iloc[0]
        print(f"  {'cnn':<8s}{'-':>22s}{r.auroc:.3f} [{r.ci_low:.2f},{r.ci_high:.2f}]".ljust(66) + f"{'-':>22s}")
    # paired bootstrap on the main comparison (seen set, MLP)
    mask = sets["seen (real+LTX+AnimateDiff+SVD)"]
    bs = boot(y[mask], {k: v[mask] for k, v in scores.items() if k.startswith("mlp")})
    print("\n  paired bootstrap on the seen set (MLP), difference in AUROC with 95% interval")
    for a, b in (("mlp|frequency", "mlp|pixel"), ("mlp|fusion", "mlp|frequency"), ("mlp|fusion", "mlp|pixel")):
        d = bs[a] - bs[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"    {a.split('|')[1]:<10s} minus {b.split('|')[1]:<10s}: {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for a, sname in zip(ax, sets):
        sub = df[(df.set == sname) & (df.features != "frequency") | ((df.set == sname) & (df.model != "cnn"))]
        sub = df[(df.set == sname) & (df.model != "cnn")]
        for i, mdl in enumerate(("lr", "lgbm", "mlp")):
            for j, fam in enumerate(FAMILIES):
                r = sub[(sub.model == mdl) & (sub.features == fam)].iloc[0]
                a.bar(i + (j - 1) * 0.27, r.auroc, 0.25, yerr=[[r.auroc - r.ci_low], [r.ci_high - r.auroc]],
                      color=["#999999", "#2a6f97", "#f2b705"][j], label=fam if i == 0 else None, capsize=2)
        a.set_xticks(range(3), ["LR", "LightGBM", "MLP"])
        a.set_title(sname, fontsize=9)
        a.set_ylim(0.4, 1.0)
        a.axhline(0.5, color="k", lw=0.5, ls=":")
    ax[0].set_ylabel("AUROC (95% bootstrap CI)")
    ax[0].legend(title="features")
    fig.tight_layout()
    fig.savefig(OUT / "04_comparison.png", dpi=130)
    plt.close(fig)
    return df, feats


def check_reproducible(df: pd.DataFrame) -> None:
    head("5. REPRODUCIBILITY: LIVE NUMBERS vs THE STORED RESULTS FILE")
    stored = pd.read_csv(RESULTS / "test_metrics.csv")
    stored = stored[stored.set == "seen_test"].set_index("model").auroc
    rows = []
    for mdl in ("lr", "lgbm", "mlp"):
        for fam, group in FAMILIES.items():
            live = df[(df.set.str.startswith("seen")) & (df.model == mdl) & (df.features == fam)].auroc.iloc[0]
            rows.append((f"{mdl}_{group}", live, stored[f"{mdl}_{group}"]))
    print(f"  {'model':<14s}{'live':>8s}{'stored':>9s}{'diff':>9s}")
    for n, a, b in rows:
        print(f"  {n:<14s}{a:8.3f}{b:9.3f}{a - b:+9.3f}")
    worst = max(abs(a - b) for _, a, b in rows)
    print(f"\n  largest difference {worst:.4f} -> " + ("matches the stored results." if worst < 0.005 else "CHECK: differs from stored."))


def retrain(man: pd.DataFrame, feats_live: pd.DataFrame) -> None:
    head("6. RETRAIN FROM THE TRAINING FEATURES AND EVALUATE ON THE LIVE HELD-OUT FEATURES")
    df = D.clean_table()
    tr, va = D.select(df, "train", include_aug=True), D.select(df, "val")
    te = man[(man.split == "test") & ~man.aug.astype(bool)].reset_index(drop=True)
    seen = ~te.generator.isin(UNSEEN).to_numpy()
    y = te.label.to_numpy().astype(int)[seen]
    print(f"  train {len(tr)} clips (incl. extra real windows), validation {len(va)}, held-out seen test {seen.sum()}")
    print(f"  {'model':<8s}{'features':<12s}{'retrained AUROC':>17s}")
    for fam, group in FAMILIES.items():
        cols = D.feature_cols(df, group)
        for mdl in ("lr", "lgbm", "mlp"):
            m = fit_model(mdl, *D.xy(tr, cols), *D.xy(va, cols))
            X = np.nan_to_num(feats_live[cols].to_numpy(np.float32))[seen]
            print(f"  {mdl:<8s}{fam:<12s}{roc_auc_score(y, m.predict_proba(X)[:, 1]):17.3f}", flush=True)


def main(args) -> None:
    OUT.mkdir(exist_ok=True)
    man = D.manifest()
    real_id, fake_id = pick_examples(man, args.real, args.fake)
    frames = show_examples({"real": real_id, "AI": fake_id}, man)
    show_spectra(frames)
    cached = D.clean_table()
    trace_stored(args.trace or fake_id, man, cached)
    if args.video:
        trace_video(args.video)
    df, feats = compare(man, args.workers)
    check_reproducible(df)
    if args.retrain:
        retrain(man, feats)
    print(f"\nFigures and tables saved in {OUT.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="clip_id of a held-out real clip")
    ap.add_argument("--fake", help="clip_id of a held-out generated clip")
    ap.add_argument("--trace", help="clip_id to trace end to end (default: the generated example)")
    ap.add_argument("--video", help="path to any video: traced from the raw file to the verdict")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retrain", action="store_true")
    main(ap.parse_args())
