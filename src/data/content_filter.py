"""Reject clips with prominent humans, animals or common objects (YOLOv8n, COCO classes)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_ROOT, RAW_FAKE, RAW_REAL

# All 80 COCO classes are people, animals or common objects -> all blocked.
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
BLOCKED_CLASSES = set(COCO)
THRESH = 0.05


def is_allowed(frac: float, thresh: float = THRESH) -> bool:
    return frac <= thresh


def max_blocked_area(frames: np.ndarray, model, conf: float = 0.35) -> tuple[float, str | None]:
    """Largest blocked-object box area (fraction of frame) over 4 evenly spaced frames."""
    idx = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    best, cls = 0.0, None
    for i in idx:
        f = frames[i]
        h, w = f.shape[:2]
        res = model.predict(f[..., ::-1].copy(), conf=conf, verbose=False)[0]  # expects BGR
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            name = res.names[int(c)]
            if name not in BLOCKED_CLASSES:
                continue
            a = (b[2] - b[0]) * (b[3] - b[1]) / (h * w)
            if a > best:
                best, cls = float(a), name
    return best, cls


def main() -> None:
    from ultralytics import YOLO
    from src.data.video_io import read_frames
    model = YOLO("yolov8n.pt")
    out = DATA_ROOT / "content_filter.csv"
    done = set(pd.read_csv(out)["path"]) if out.exists() else set()
    paths = sorted(RAW_REAL.glob("*.mp4")) + sorted(RAW_FAKE.glob("*/*.mp4"))
    rows = []
    for i, p in enumerate(paths):
        if str(p) in done:
            continue
        try:
            frames = read_frames(p, 16, 8)
            frac, cls = max_blocked_area(frames, model)
        except Exception as e:
            print("skip", p, e)
            continue
        rows.append({"path": str(p), "frac": frac, "cls": cls, "allowed": is_allowed(frac)})
        if len(rows) >= 50:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            rows = []
            print(f"{i + 1}/{len(paths)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    df = pd.read_csv(out)
    print(f"checked {len(df)}; rejected {int((~df.allowed).sum())}")
    print(df[~df.allowed].cls.value_counts().head(15).to_string())


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
