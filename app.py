"""Gradio test interface: upload a clip -> real/AI verdict with frequency-domain explanations.

Run:  python app.py   (then open http://127.0.0.1:7860)
"""
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS, N_BINS, RESULTS
from src.data.degrade import DEGRADATIONS
from src.features.spatial import power_spectrum, to_luma
from src.predict import Predictor

ROOT = Path(__file__).parent
EXAMPLES = sorted((ROOT / "examples").glob("*.mp4"))
REF = MODELS / "reference_spectra.json"   # class-mean radial spectra, written by build_reference()
P = Predictor()

MODEL_LABELS = {
    "mlp_fusion": "MLP — frequency + pixel fusion (recommended)",
    "mlp_all": "MLP — frequency: spatial + temporal",
    "mlp_spatial": "MLP — spatial frequency only (fine detail)",
    "mlp_temporal": "MLP — temporal only (change over time)",
    "mlp_pixel": "MLP — pixel statistics only (control)",
    "resnet18": "ResNet-18 — fine-tuned on raw frames",
}


def _ref():
    return json.loads(REF.read_text()) if REF.exists() else None


def _verdict_html(res):
    p = res.p_fake
    color = "#c9184a" if p >= 0.5 else "#2a6f97"
    conf = max(p, 1 - p)
    html = (f"<div style='padding:16px;border-radius:10px;border:2px solid {color}'>"
            f"<div style='font-size:28px;font-weight:700;color:{color}'>{res.label}</div>"
            f"<div style='font-size:16px'>P(AI-generated) = <b>{p:.3f}</b> &nbsp;·&nbsp; confidence {conf:.0%}</div>"
            f"<div style='height:12px;background:#ddd;border-radius:6px;margin-top:8px'>"
            f"<div style='height:12px;width:{p * 100:.1f}%;background:{color};border-radius:6px'></div></div>")
    if res.content_warning:
        html += f"<p style='color:#b45309'>⚠ {res.content_warning}</p>"
    for n in res.notes:
        html += f"<p style='color:#555'>ℹ {n}</p>"
    return html + "</div>"


def _frames_fig(frames):
    idx = np.linspace(0, len(frames) - 1, 4).astype(int)
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for j, i in enumerate(idx):
        ax[0, j].imshow(frames[i])
        ax[0, j].set_title(f"frame {i}")
        s = np.log(power_spectrum(to_luma(frames[i])) + 1e-8)
        ax[1, j].imshow(s, cmap="magma")
        ax[1, j].set_title("log power spectrum")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    return fig


def _drift_fig(res):
    s = res.seq[0]
    l2 = np.linalg.norm(np.diff(s, axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, len(s)), l2, marker="o", color="k", label="this clip")
    ref = _ref()
    if ref:
        for k, c, lab in (("real", "#2a6f97", "real (training mean)"), ("fake", "#c9184a", "AI (training mean)")):
            ax.axhline(ref[k]["drift_l2_mean"], color=c, ls="--", label=lab)
    ax.set_title("Frame-to-frame spectral drift")
    ax.set_xlabel("frame step (t to t+1)")
    ax.set_ylabel("distance between radial spectra")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def detect(video, model, start, degradation):
    if video is None:
        raise gr.Error("Please upload a video or pick an example.")
    res = P.predict(video, model=model, start=float(start), degradation=degradation)
    return _verdict_html(res), _frames_fig(res.frames), _drift_fig(res)


def _perf_tables():
    out = {}
    for name in ["test_metrics", "per_generator", "degraded"]:
        p = RESULTS / f"{name}.csv"
        try:
            d = pd.read_csv(p).round(3)
            out[name] = d[d.model.str.startswith("mlp")]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            out[name] = pd.DataFrame({"info": [f"{name}.csv not available — run evaluation"]})
    try:
        r = pd.read_csv(RESULTS / "late_fusion.csv")
        r = r[r.set.isin(["seen", "modelscope", "wan"])][["set", "n", "resnet18", "frequency_mlp"]]
        out["resnet"] = r.rename(columns={"set": "test set", "n": "clips", "frequency_mlp": "frequency MLP"}).round(3)
    except FileNotFoundError:
        out["resnet"] = pd.DataFrame({"info": ["late_fusion.csv not available"]})
    return out


def build_ui():
    models = [m for m in MODEL_LABELS if m in P.available()]
    choices = [(MODEL_LABELS.get(m, m), m) for m in models]
    tables = _perf_tables()
    with gr.Blocks(title="AI-Generated Video Detector") as demo:
        gr.Markdown("# AI-Generated Video Detection via Frequency Domain\n"
                    "IE 643 course project. Upload a short clip (scenery works best: no prominent people, "
                    "animals or objects). The detector reads the **spatial frequency spectrum** of 14 frames "
                    "(every k-th frame, ≈8 fps, ≈1.8 s) and how that spectrum **drifts from frame to frame**.")
        with gr.Tab("Detect"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid = gr.Video(label="Input video", sources=["upload"])
                    model = gr.Dropdown(choices, value=models[0] if models else None, label="Model")
                    start = gr.Slider(0, 30, value=0, step=0.5, label="Start time (s)")
                    deg = gr.Dropdown(["none"] + list(DEGRADATIONS), value="none",
                                      label="Apply degradation (robustness test)")
                    btn = gr.Button("Analyse", variant="primary")
                    if EXAMPLES:
                        gr.Examples([[str(e)] for e in EXAMPLES], inputs=[vid], label="Example clips")
                with gr.Column(scale=2):
                    verdict = gr.HTML()
                    frames_plot = gr.Plot(label="Frames and their 2D spectra")
            drift_plot = gr.Plot(label="Temporal frequency drift")
            btn.click(detect, [vid, model, start, deg], [verdict, frames_plot, drift_plot])
        with gr.Tab("Performance"):
            gr.Markdown("### Test-set metrics (seen generators and unseen generators)")
            gr.Dataframe(tables["test_metrics"])
            gr.Markdown("### Per-generator detection (unseen = never used in training)")
            gr.Dataframe(tables["per_generator"])
            gr.Markdown("### Robustness to degradations")
            gr.Dataframe(tables["degraded"])
            gr.Markdown("### ResNet-18 vs the frequency MLP (AUROC, one stored run)")
            gr.Dataframe(tables["resnet"])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
