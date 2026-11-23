"""Single-video inference used by the Gradio app."""
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch

from src.config import FPS, MODELS, N_FRAMES, SIZE
from src.data.degrade import apply_degradation
from src.data.normalize import normalize_frames
from src.data.video_io import h264_roundtrip, probe, read_frames_stride
from src.features.pixel_baseline import clip_pixel_features
from src.features.temporal import clip_features
from src.models.cnn import SpecTempCNN, cnn_predict


class _Squeeze(torch.nn.Module):
    """Single-logit wrapper around ResNet-18, same layout as src.models.pixel_cnn (state-dict keys 'net.*')."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class PredictionResult:
    p_fake: float
    label: str
    features: dict
    seq: np.ndarray
    frames: np.ndarray
    top_features: list = field(default_factory=list)
    content_warning: str | None = None
    notes: list = field(default_factory=list)


class Predictor:
    def __init__(self, model_dir=MODELS):
        self.model_dir = model_dir
        self._cache = {}
        self._yolo = None

    def available(self) -> list[str]:
        names = sorted(p.stem for p in self.model_dir.glob("*.joblib")
                       if "_logo_" not in p.stem)
        if (self.model_dir / "cnn.pt").exists():
            names.append("cnn")
        if (self.model_dir / "resnet18.pt").exists():
            names.append("resnet18")
        return names

    def _load(self, name):
        if name not in self._cache:
            if name == "resnet18":
                from torchvision.models import resnet18
                net = resnet18(weights=None)                      # the fine-tuned weights come from the checkpoint
                net.fc = torch.nn.Linear(net.fc.in_features, 1)
                m = _Squeeze(net)
                m.load_state_dict(torch.load(self.model_dir / "resnet18.pt", map_location="cpu"))
                self._cache[name] = {"resnet18": m.eval()}
            elif name == "cnn":
                ck = torch.load(self.model_dir / "cnn.pt", weights_only=False)
                m = SpecTempCNN(ck["in_ch"], ck["n_bins"])
                m.load_state_dict(ck["state"])
                self._cache[name] = {"cnn": m.eval(), "stats": (ck["mu"], ck["sd"])}
            else:
                self._cache[name] = joblib.load(self.model_dir / f"{name}.joblib")
        return self._cache[name]

    def _content_check(self, frames) -> str | None:
        try:
            from src.data.content_filter import max_blocked_area
            if self._yolo is None:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
            frac, cls = max_blocked_area(frames, self._yolo)
            if frac > 0.05:
                return (f"This clip contains a prominent '{cls}' ({frac:.0%} of the frame). The detector was "
                        "trained on scenes without people/animals/objects, so treat this result with caution.")
        except Exception:
            return None
        return None

    def predict(self, video_path, model: str = "lgbm_all", start: float = 0.0,
                degradation: str | None = None, check_content: bool = True) -> PredictionResult:
        notes = []
        meta = probe(video_path)
        if meta["duration"] is not None and meta["duration"] < (N_FRAMES - 1) / FPS:
            notes.append(f"Clip is only {meta['duration']:.1f} s; frames were looped to fill the 14-frame window.")
        if min(meta["w"], meta["h"]) < SIZE:
            notes.append(f"Low resolution ({meta['w']}x{meta['h']}); upscaled to {SIZE}px, which removes "
                         "high-frequency detail and makes the decision less reliable.")
        raw, _ = read_frames_stride(video_path, N_FRAMES, FPS, start)
        # same final step as the stored training clips: 256x256 H.264 CRF 18 at 8 fps
        # mirror the real-clip training path: large videos are first brought to a generator-like size
        pre = 512 if min(raw.shape[1:3]) > 576 else None
        frames = h264_roundtrip(normalize_frames(raw, pre), FPS, crf=18)
        if degradation and degradation != "none":
            frames = apply_degradation(frames, degradation)
            notes.append(f"Applied degradation: {degradation}.")
        feats, seq = clip_features(frames)
        feats.update(clip_pixel_features(frames))
        entry = self._load(model)
        top = []
        if "resnet18" in entry:
            p = self._resnet_score(entry["resnet18"], frames)
        elif "cnn" in entry:
            p = float(cnn_predict(entry["cnn"], entry["stats"], seq[None])[0])
        else:
            cols = entry["cols"]
            X = np.nan_to_num(np.array([[feats[c] for c in cols]], dtype=np.float32))
            p = float(entry["model"].predict_proba(X)[0, 1])
            top = self._explain(entry, X, cols)
        warn = self._content_check(raw) if check_content else None
        return PredictionResult(p, "AI-generated" if p >= 0.5 else "Real (camera)", feats, seq, frames,
                                top, warn, notes)

    @staticmethod
    def _resnet_score(net, frames: np.ndarray) -> float:
        """Clip score as in training: every 2nd frame, centre 224 crop, mean sigmoid over those frames."""
        crop = 224
        o = (frames.shape[1] - crop) // 2
        x = np.ascontiguousarray(frames[list(range(0, N_FRAMES, 2)), o:o + crop, o:o + crop])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            return float(torch.sigmoid(net((x - mean) / std)).mean())

    def _explain(self, entry, X, cols, k: int = 10) -> list[tuple[str, float]]:
        """Top-k feature contributions (SHAP for trees, coef*standardised value for linear)."""
        m = entry["model"]
        try:
            if m.__class__.__name__ == "LGBMClassifier":
                contrib = m.predict(X, pred_contrib=True)[0][:-1]
            elif hasattr(m, "named_steps") and hasattr(m.named_steps.get("clf"), "coef_"):
                z = m.named_steps["sc"].transform(X)[0]
                contrib = m.named_steps["clf"].coef_[0] * z
            else:
                return []
        except Exception:
            return []
        order = np.argsort(-np.abs(contrib))[:k]
        return [(cols[i], float(contrib[i])) for i in order]
