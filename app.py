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

from src.config import N_BINS, RESULTS
from src.data.degrade import DEGRADATIONS
from src.features.spatial import power_spectrum, to_luma
from src.predict import Predictor

ROOT = Path(__file__).parent
EXAMPLES = sorted((ROOT / "examples").glob("*.mp4"))
REF = ROOT / "models" / "reference_spectra.json"   # class-mean radial spectra, written by build_reference()
P = Predictor()
x_freq = (np.arange(N_BINS) + 0.5) / N_BINS

MODEL_LABELS = {
    "lgbm_all": "LightGBM — spatial + temporal (recommended)",
    "lgbm_all_robust": "LightGBM — degradation-aware",
    "cnn": "Spectro-temporal 1D-CNN",
    "mlp_all": "MLP — spatial + temporal",
    "lr_all": "Logistic regression — spatial + temporal",
    "svm_all": "Linear SVM — spatial + temporal",
    "lgbm_spatial": "LightGBM — spatial only",
    "lgbm_temporal": "LightGBM — temporal only",
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


def _spectrum_fig(res):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    rad = np.array([res.features[f"sp_rad_{i:02d}_mean"] for i in range(N_BINS)])
    res_rad = np.array([res.features[f"sp_res_rad_{i:02d}_mean"] for i in range(N_BINS)])
    ref = _ref()
    for a, v, key, title in ((ax[0], rad, "rad", "Radial power spectrum"),
                             (ax[1], res_rad, "res_rad", "Noise-residual spectrum")):
        if ref:
            a.plot(x_freq, ref["real"][key], color="#2a6f97", label="real (training mean)")
            a.plot(x_freq, ref["fake"][key], color="#c9184a", label="AI (training mean)")
        a.plot(x_freq, v, color="k", lw=2, label="this clip")
        a.set_title(title)
        a.set_xlabel("spatial frequency (fraction of Nyquist)")
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _drift_fig(res):
    s = res.seq[0]
    l2 = np.linalg.norm(np.diff(s, axis=0), axis=1)
    hi = s[:, 2 * N_BINS // 3:].mean(1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(range(1, len(s)), l2, marker="o")
    ax[0].set_title("Frame-to-frame spectral drift (L2)")
    ax[0].set_xlabel("frame")
    ref = _ref()
    if ref:
        for k, c in (("real", "#2a6f97"), ("fake", "#c9184a")):
            ax[0].axhline(ref[k]["drift_l2_mean"], color=c, ls="--", label=f"{k} mean")
        ax[0].legend(fontsize=8)
    ax[1].plot(hi - hi.mean(), marker="o", color="#e07a5f")
    ax[1].set_title("High-frequency log power over time")
    ax[1].set_xlabel("frame")
    im = ax[2].imshow(s.T, origin="lower", aspect="auto", cmap="magma", extent=[0, len(s), 0, 1])
    ax[2].set_title("Radial spectrum over time")
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("freq (frac. Nyquist)")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout()
    return fig


def _shap_fig(res):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if not res.top_features:
        ax.text(0.5, 0.5, "Per-feature attribution is shown for LightGBM / linear models.\n"
                "The CNN's frequency×time saliency is in the Performance tab.", ha="center", va="center")
        ax.axis("off")
        return fig
    names = [n for n, _ in res.top_features][::-1]
    vals = [v for _, v in res.top_features][::-1]
    ax.barh(names, vals, color=["#c9184a" if v > 0 else "#2a6f97" for v in vals])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Top feature contributions (red → AI, blue → real)")
    fig.tight_layout()
    return fig


def detect(video, model, start, degradation):
    if video is None:
        raise gr.Error("Please upload a video or pick an example.")
    res = P.predict(video, model=model, start=float(start), degradation=degradation)
    return _verdict_html(res), _frames_fig(res.frames), _spectrum_fig(res), _drift_fig(res), _shap_fig(res)


def _perf_tables():
    out = {}
    for name in ["test_metrics", "per_generator", "degraded", "logo", "metadata_baseline"]:
        p = RESULTS / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            out[name] = df.round(3)
        else:
            out[name] = pd.DataFrame({"info": [f"{name}.csv not found — run evaluation"]})
    return out


def build_ui():
    models = [m for m in MODEL_LABELS if m in P.available()] + [m for m in P.available() if m not in MODEL_LABELS]
    choices = [(MODEL_LABELS.get(m, m), m) for m in models]
    tables = _perf_tables()
    figs = RESULTS / "figures"
    with gr.Blocks(title="AI-Generated Video Detector") as demo:
        gr.Markdown("# AI-Generated Video Detection via Frequency Domain\n"
                    "IE 643 course project. Upload a short clip (scenery works best: no prominent people, "
                    "animals or objects). The detector reads the **spatial frequency spectrum** of 16 frames "
                    "(2 s at 8 fps) and how that spectrum **drifts from frame to frame**.")
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
            spec_plot = gr.Plot(label="Spectrum vs training means")
            drift_plot = gr.Plot(label="Temporal frequency drift")
            shap_plot = gr.Plot(label="Why this decision")
            btn.click(detect, [vid, model, start, deg], [verdict, frames_plot, spec_plot, drift_plot, shap_plot])
        with gr.Tab("Performance"):
            gr.Markdown("### Test-set metrics (seen generators and unseen generators)")
            gr.Dataframe(tables["test_metrics"])
            gr.Markdown("### Per-generator detection (unseen = never used in training)")
            gr.Dataframe(tables["per_generator"])
            gr.Markdown("### Robustness to degradations")
            gr.Dataframe(tables["degraded"])
            gr.Markdown("### Leave-one-generator-out and metadata-only shortcut check")
            gr.Dataframe(tables["logo"])
            gr.Dataframe(tables["metadata_baseline"])
            imgs = [str(figs / f) for f in ["roc_curves.png", "ablation_bars.png", "degradation_bars.png",
                                            "radial_profiles.png", "mean_spectrum_2d.png",
                                            "feature_importance_groups.png", "importance_over_frequency.png",
                                            "cnn_saliency.png", "drift_distributions.png", "tsne.png",
                                            "error_gallery.png", "confusion_matrices.png"] if (figs / f).exists()]
            gr.Gallery(imgs, columns=2, height="auto", label="Figures")
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
