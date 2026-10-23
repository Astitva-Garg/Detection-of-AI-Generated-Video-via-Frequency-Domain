"""Pixel-domain deep baselines on raw frames: a small CNN trained from scratch and a fine-tuned
ImageNet ResNet-18.

They answer: does a standard learned pixel model beat (or generalise better than) our frequency
features? Same clips, splits and test sets as everything else.

- Input: 8 of the 16 normalised frames per clip (native 256x256 crops, no resizing — resizing would
  low-pass filter the frames). Training: random 224x224 crop + horizontal flip, frame-level labels.
- Clip score: mean sigmoid over the 8 frames, centre 224 crop.
- Loss BCE-with-logits with positive-class weight; AdamW; cosine LR; best epoch by validation AUROC.
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import CLIPS, FEATURES, FPS, N_FRAMES, RAW_REAL, RESULTS, UNSEEN
from src.data.normalize import normalize_frames
from src.data.video_io import h264_roundtrip, read_frames, read_frames_stride
from src.data.wild import WILD
from src.eval.evaluate import metrics
from src.models import data as D

FRAME_IDX = list(range(0, N_FRAMES, 2))          # 8 frames, 0.25 s apart
CROP = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- frame cache
def _clean_frames(clip_id: str) -> np.ndarray:
    return read_frames(CLIPS / f"{clip_id}.mp4", N_FRAMES, FPS)[FRAME_IDX]


def _deploy_frames(path: str) -> np.ndarray:
    """The app's preprocessing path (same as wild_eval)."""
    raw, _ = read_frames_stride(path, N_FRAMES, FPS)
    pre = 512 if min(raw.shape[1:3]) > 576 else None
    return h264_roundtrip(normalize_frames(raw, pre), FPS, crf=18)[FRAME_IDX]


def _cache(name: str, items: list[tuple[str, str]], fn) -> tuple[np.ndarray, list[str]]:
    npy, ids_p = FEATURES / f"frames_{name}.npy", FEATURES / f"frames_{name}_ids.txt"
    if npy.exists() and ids_p.exists():
        ids = ids_p.read_text().split("\n")
        if ids == [k for k, _ in items]:
            return np.load(npy, mmap_mode="r"), ids
    from concurrent.futures import ProcessPoolExecutor
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(items), len(FRAME_IDX), 256, 256, 3))
    with ProcessPoolExecutor(4) as ex:
        for i, fr in enumerate(ex.map(fn, [p for _, p in items], chunksize=4)):
            arr[i] = fr
            if i % 200 == 0:
                print(f"  cached {i}/{len(items)} {name}", flush=True)
    arr.flush()
    ids_p.write_text("\n".join(k for k, _ in items))
    return np.load(npy, mmap_mode="r"), [k for k, _ in items]


def load_frames():
    df = D.clean_table()
    keep = df[(df.split.isin(["train", "val"])) | ((df.split == "test") & ~df.aug.astype(bool))]
    clean, ids = _cache("clean", [(c, c) for c in keep.clip_id], _clean_frames)
    wild = pd.read_parquet(FEATURES / "wild.parquet")[["key", "source", "label"]]
    items = []
    for k in wild.key:
        src, stem = k.split("|", 1)
        p = RAW_REAL / f"{stem.replace('real_', '')}.mp4" if src == "real" else WILD / src / f"{stem}.mp4"
        items.append((k, str(p)))
    wframes, wids = _cache("wild", items, _deploy_frames)
    return df, clean, {c: i for i, c in enumerate(ids)}, wild.set_index("key"), wframes, {k: i for i, k in enumerate(wids)}


# ----------------------------------------------------------------------------- models
class PixelCNN(nn.Module):
    """Small from-scratch CNN on raw frames (~590k parameters; 5 stride-2 blocks of two 3x3 convs)."""

    def __init__(self):
        super().__init__()
        chans, layers, c_in = [16, 32, 64, 128, 128], [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class _Squeeze(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1)


def resnet18_ft():
    """ImageNet-pretrained ResNet-18, all layers fine-tuned, single-logit head."""
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return _Squeeze(m)


CONFIGS = {
    "pixel_cnn": dict(make=PixelCNN, lr=1e-3, wd=1e-4, epochs=15),
    "resnet18": dict(make=resnet18_ft, lr=1e-4, wd=1e-4, epochs=6),
}


# ----------------------------------------------------------------------------- train / predict
def _to_tensor(batch_u8: np.ndarray, train: bool, rng: np.random.Generator) -> torch.Tensor:
    """(B,256,256,3) uint8 -> normalised (B,3,224,224) float with crop (+flip when training)."""
    b = batch_u8.shape[0]
    if train:
        ys, xs = rng.integers(0, 256 - CROP + 1, b), rng.integers(0, 256 - CROP + 1, b)
        out = np.stack([batch_u8[i, y:y + CROP, x:x + CROP] for i, (y, x) in enumerate(zip(ys, xs))])
        flip = rng.random(b) < 0.5
        out[flip] = out[flip, :, ::-1]
    else:
        o = (256 - CROP) // 2
        out = batch_u8[:, o:o + CROP, o:o + CROP]
    t = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2).float().div_(255)
    return (t - MEAN) / STD


@torch.no_grad()
def predict_clips(model, frames: np.ndarray, rows: list[int], device="cuda", bs=16) -> np.ndarray:
    model.eval()
    out = []
    rng = np.random.default_rng(0)
    for i in range(0, len(rows), bs):
        chunk = np.asarray(frames[rows[i:i + bs]])                 # (b, 8, 256, 256, 3)
        b, f = chunk.shape[:2]
        x = _to_tensor(chunk.reshape(b * f, 256, 256, 3), False, rng).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x).float()).view(b, f).mean(1)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def train_one(kind: str, frames, rows_tr, y_tr, rows_va, y_va, seed: int, device="cuda", bs=64):
    cfg = CONFIGS[kind]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cfg["make"]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pairs = np.array([(r, f) for r in rows_tr for f in range(len(FRAME_IDX))])
    lab = np.repeat(y_tr, len(FRAME_IDX)).astype(np.float32)
    steps = cfg["epochs"] * int(np.ceil(len(pairs) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    pos = lab.mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, hist = -1.0, None, []
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(len(pairs))
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            order = np.argsort(pairs[idx, 0])                      # sorted reads are faster on memmap
            idx = idx[order]
            batch = np.stack([frames[r, f] for r, f in pairs[idx]])
            x = _to_tensor(batch, True, rng).to(device, non_blocking=True)
            y = torch.from_numpy(lab[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
        auc = roc_auc_score(y_va, predict_clips(model, frames, rows_va, device))
        hist.append({"epoch": ep, "train_loss": tot / len(pairs), "val_auc": auc, "sec": time.time() - t0})
        print(f"    {kind} seed {seed} ep {ep}: loss {tot / len(pairs):.4f} val AUROC {auc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, hist


# ----------------------------------------------------------------------------- experiment
def main(kinds: list[str], seeds: list[int]) -> pd.DataFrame:
    df, frames, cidx, wild, wframes, widx = load_frames()
    tr, va = D.select(df, "train", include_aug=True), D.select(df, "val")
    te = df[(df.split == "test") & ~df.aug.astype(bool)]
    real = te[te.generator == "real"]
    sets = {"seen": te[~te.generator.isin(UNSEEN)]}
    for g in UNSEEN:
        sets[g] = pd.concat([real, te[te.generator == g]])
    wreal = wild[wild.source == "real"]
    old = RESULTS / "pixel_deep_runs.csv"
    rows = ([r for r in pd.read_csv(old).to_dict("records") if r["model"] not in kinds]
            if old.exists() else [])               # keep earlier runs of models not being retrained
    for kind in kinds:
        for seed in seeds:
            model, hist = train_one(kind, frames, [cidx[c] for c in tr.clip_id], tr.label.to_numpy(),
                                    [cidx[c] for c in va.clip_id], va.label.to_numpy(), seed)
            r = {"seed": seed, "model": kind, "params": sum(p.numel() for p in model.parameters())}
            for sname, sub in sets.items():
                r[sname] = metrics(sub.label, predict_clips(model, frames, [cidx[c] for c in sub.clip_id]))["auroc"]
            for src in sorted(s for s in wild.source.unique() if s != "real"):
                sub = pd.concat([wild[wild.source == src], wreal])
                p = predict_clips(model, wframes, [widx[k] for k in sub.index])
                r[f"wild_{src}"] = metrics(sub.label, p)["auroc"]
            rows.append(r)
            pd.DataFrame(hist).to_csv(RESULTS / f"{kind}_history_seed{seed}.csv", index=False)
            if seed == seeds[0]:
                torch.save(model.state_dict(), RESULTS.parent / f"models{RESULTS.name[7:]}" / f"{kind}.pt")
            print(kind, seed, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "pixel_deep_runs.csv", index=False)
    res = pd.DataFrame(rows).drop(columns=["wild_mean"], errors="ignore")
    wcols = [c for c in res.columns if c.startswith("wild_")]
    res["wild_mean"] = res[wcols].mean(axis=1)
    res.to_csv(RESULTS / "pixel_deep_runs.csv", index=False)
    summ = res.drop(columns=["seed", "params"]).groupby("model").agg(["mean", "std"]).round(3)
    summ.to_csv(RESULTS / "pixel_deep.csv")
    print(summ.to_string())
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["pixel_cnn", "resnet18"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[643, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.kinds, a.seeds)
