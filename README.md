# Detection of AI-Generated Video via Frequency Domain

IE 643 course project — team *mdhkideggimirch* (Astitva Garg, 24B2440).

A lightweight detector that decides whether a short clip is **camera-captured** or **diffusion-generated**
from (a) spatial frequency-domain signatures of each frame and (b) how those signatures **drift from frame
to frame**. Includes a curated real-vs-AI dataset pipeline (5 open T2V/I2V generators), evaluation on
seen / unseen generators and degraded videos, interpretability figures and a Gradio test interface.

## Pipeline

```
Pexels-400k metadata ─► real clips ─┐
                                    ├─► content filter (YOLOv8n) ─► normalise (8 fps, 16 frames, 256², H.264)
5 generators (diffusers) ─► fakes ──┘        │
                                             ├─► manifest.csv (labels, generator, group-level splits)
                                             ├─► degraded copies (H.264 CRF 23/30/38, 0.5× down, blur, noise)
                                             ▼
             features: radial spectrum, residual spectrum, slope, HF ratios, DCT bands, angular energy
                       + temporal drift (spectral L2/cosine drift, band drift, scalar drift, residual temporal corr.)
                                             ▼
             models: LogReg, Linear SVM, LightGBM, MLP (feature vectors) · Spectro-temporal 1D-CNN (spectrum sequences)
                                             ▼
             evaluation (seen / unseen / degraded / leave-one-generator-out) · figures · Gradio app
```

| Generator | Type | Role | Hugging Face id |
|---|---|---|---|
| LTX-Video | T2V | seen (train/val/test) | `Lightricks/LTX-Video` |
| AnimateDiff v1-5-2 | T2V | seen | `guoyww/animatediff-motion-adapter-v1-5-2` + `emilianJR/epiCRealism` |
| Stable Video Diffusion XT | I2V from real first frames | seen | `stabilityai/stable-video-diffusion-img2vid-xt` |
| Wan2.1 T2V 1.3B | T2V | **unseen** (test only) | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| ModelScope T2V 1.7B | T2V | **unseen** (test only) | `ali-vilab/text-to-video-ms-1.7b` |

## Setup

Tested on Windows 11, RTX 4060 Laptop (8 GB), 15 GB RAM, Python 3.11.

```bash
uv venv --python 3.11 C:/ie643_env
uv pip install --python C:/ie643_env/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python C:/ie643_env/Scripts/python.exe -r requirements.txt
```

Data lives in `C:\ie643_data` (set `DATA_ROOT` in `src/config.py` to change it). Model weights for
the generators are downloaded automatically from Hugging Face on first use.

## Reproduce

Run from the repository root with the venv's python:

```bash
python -m src.data.pexels --n 560 --cap 40          # 1. real clips (Pexels, no API key needed)
python -m src.generate.run_queue ltx=160 animatediff=160 svd=120 modelscope=100 wan=80   # 2. AI clips
python -m src.data.content_filter                   # 3. reject people/animals/objects
python -m src.data.build_dataset                    # 4. normalise, manifest, splits, degraded copies
python -m src.features.extract                      # 5. features (parquet + spectrum sequences)
python -m src.models.train                          # 6. train all models
python -m src.eval.evaluate                         # 7. metrics -> results/*.csv
python -m src.viz.figures                           # 8. figures -> results/figures/
python -m src.viz.reference                         # 9. reference spectra + example clips for the app
python app.py                                       # 10. interface at http://127.0.0.1:7860
pytest                                              # unit tests
```

Every stage is resumable/incremental: rerunning skips work already done.

## Interface

`python app.py` opens a two-tab Gradio app:

- **Detect** — upload any video (mp4/mov/avi/mkv/webm, any resolution/length) or pick an example. Choose a
  model, a start time and optionally a degradation (to test robustness live). Output: verdict and
  P(AI-generated), frames with their 2D spectra, the clip's radial / residual spectrum against the
  training-set real and AI means, the temporal drift plots, and the top feature contributions.
- **Performance** — test metrics, per-generator results (incl. unseen), robustness table,
  leave-one-generator-out, metadata-only shortcut baseline and all figures.

## Layout

```
configs/scenes.yaml        scene vocabulary: Pexels title keywords, exclusions, T2V prompts
src/config.py              paths and constants
src/data/                  pexels.py, video_io.py, normalize.py, content_filter.py, splits.py, degrade.py, build_dataset.py
src/generate/              common.py (resumable runner), text_embed.py, ltx.py, animatediff.py, svd.py, wan.py, modelscope.py, run_queue.py
src/features/              spatial.py (per-frame), temporal.py (drift + clip features), extract.py
src/models/                data.py, classic.py (LR/SVM/LightGBM/MLP), cnn.py (spectro-temporal CNN), train.py
src/eval/evaluate.py       all evaluation tables
src/viz/                   figures.py, reference.py
src/predict.py, app.py     inference + Gradio interface
models/                    trained models (joblib / pt) and reference spectra
results/                   metrics CSV/JSON and figures
docs/                      design, plan, notes for the report
```

## Licences and credits

Real footage: Pexels (Pexels licence; videos are not redistributed here — `manifest.csv` lists Pexels
ids/URLs so the set can be rebuilt). Metadata index: `jovianzm/Pexels-400k` (Hugging Face).
Generators: see table above (each under its own licence). Object detector: Ultralytics YOLOv8n.
Libraries: PyTorch, diffusers, transformers, scikit-learn, LightGBM, Gradio.
