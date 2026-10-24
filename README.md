# Detection of AI-Generated Video via Frequency Domain

IE 643 course project — team *mdhkideggimirch* (Astitva Garg, 24B2440).

A lightweight detector that decides whether a short clip is **camera-captured** or **diffusion-generated**
from (a) spatial frequency-domain signatures of each frame and (b) how those signatures **drift from frame
to frame**. Includes a curated real-vs-AI dataset pipeline (5 open T2V/I2V generators), evaluation on
seen / unseen generators and degraded videos, interpretability figures and a Gradio test interface.

## Pipeline

```
Pexels-400k metadata ─► real clips ─┐
                                    ├─► content filter (YOLOv8n) ─► normalise (14 frames, every k-th native frame ≈8 fps, native 256² crop, H.264)
5 generators (diffusers) ─► fakes ──┘        │
                                             ├─► manifest.csv (labels, generator, group-level splits)
                                             ├─► degraded copies (H.264 CRF 23/30/38, 0.5× down, blur, noise)
                                             ▼
             features: radial spectrum, residual spectrum, slope, HF ratios, DCT bands, angular energy
                       + temporal drift (spectral L2/cosine drift, band drift, scalar drift, residual temporal corr.)
                                             ▼
             + pixel-statistics baseline (74 features) and frequency+pixel fusion
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

## Results (held-out test AUROC, mean ± std over 5 training seeds)

| Model | Params | Seen generators | Unseen ModelScope | Unseen Wan2.1 | In the wild (Sora 2 / mean of 5) |
|---|---|---|---|---|---|
| MLP, frequency + pixel fusion (app default) | ~50k | 0.977 ± 0.003 | 0.744 ± 0.024 | 0.819 ± 0.008 | 0.899 / 0.754 |
| MLP, frequency (spatial + temporal drift) | ~45k | 0.966 ± 0.006 | 0.798 ± 0.008 | 0.690 ± 0.012 | 0.855 / 0.727 |
| Spectro-temporal 1D-CNN | 65k | 0.919 ± 0.021 | 0.740 ± 0.029 | 0.640 ± 0.052 | 0.859 / 0.757 |
| MLP, pixel statistics (baseline) | ~15k | 0.889 ± 0.007 | 0.619 ± 0.027 | 0.827 ± 0.015 | 0.772 / 0.694 |
| Pixel CNN from scratch (baseline) | 590k | 0.808 ± 0.027 | 0.598 ± 0.057 | 0.871 ± 0.011 | 0.747 / 0.679 |
| ResNet-18, ImageNet-pretrained, fine-tuned (baseline) | 11.2M | 0.954 ± 0.009 | 0.900 ± 0.010 | 0.926 ± 0.017 | 0.809 / 0.781 |

Grouped 5-fold CV on seen generators (LightGBM / MLP): spatial + temporal 0.906 / 0.909, fusion 0.929 / 0.928,
pixel 0.884 / 0.842. Resizing frames to 256 px instead of a native-resolution crop cost ≈0.10 AUROC.
Frames are sampled with a fixed stride of native frames (14 frames, ≈8 fps): time-based resampling had
duplicated frames in SVD clips, a shortcut found by `src/data/audit.py` and fixed before these results.
In the wild = public clips from Sora 2, Veo 3, Kling, HunyuanVideo, Wan2 (different content; stress test only).
Full tables: `results_crop/*.csv`; figures: `results_crop/figures/`; notes: `docs/notes_for_report.md` §16.

## Setup

Tested on Windows 11, RTX 4060 Laptop (8 GB), 15 GB RAM, Python 3.11.

```bash
uv venv --python 3.11 C:/ie643_env
uv pip install --python C:/ie643_env/Scripts/python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python C:/ie643_env/Scripts/python.exe -r requirements.txt
```

Data lives in `C:\ie643_data` (set `DATA_ROOT` in `src/config.py` to change it). Model weights for
the generators are downloaded automatically from Hugging Face on first use.

## Reproduce

Run from the repository root with the venv's python:

```bash
python -m src.data.pexels --n 560 --cap 40          # 1. real clips (Pexels, no API key needed)
python -m src.generate.run_queue ltx=160 animatediff=160 svd=120 modelscope=100 wan=80   # 2. AI clips
bash run_pipeline.sh                                # 3–9 in one go: content filter, normalise + manifest +
                                                    #   splits + degraded copies, frequency + pixel features,
                                                    #   train, evaluate, grouped CV, figures, app examples
python app.py                                       # 10. interface at http://127.0.0.1:7860
python -m src.eval.bootstrap                        # extra: bootstrap 95 % CIs for test AUROC and model differences
python -m src.eval.attribution                      # extra: 6-class generator attribution (grouped CV)
python -m src.data.wild && python -m src.eval.wild_eval   # extra: in-the-wild stress test (Sora 2, Veo 3, Kling, ...)
python -m src.models.pixel_cnn                      # extra: pixel CNN from scratch + fine-tuned ResNet-18 baselines
python -m src.data.audit                            # extra: dataset audit (repeated frames, duplicates, borders)
python -m src.eval.seed_stability                   # extra: retrain main models with 5 seeds (needs wild_eval first)
pytest                                              # unit tests
```

Every stage is resumable/incremental: rerunning skips work already done.
Outputs of the main pipeline (native 256² crop) go to `C:\ie643_data\*_crop`, `models_crop/` and
`results_crop/`. Set `IE643_NORM=resize` to reproduce the resize ablation (`models/`, `results/`).
`IE643_DATA` overrides the data folder. `cloud/` holds a script to generate the Wan2.1 clips on a rented GPU.

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
src/features/              spatial.py (per-frame), temporal.py (drift + clip features), extract.py, pixel_baseline.py
src/models/                data.py, classic.py (LR/SVM/LightGBM/MLP), cnn.py (spectro-temporal CNN), train.py
src/eval/                  evaluate.py (test / unseen / degraded / LOGO / metadata tables), cv.py (grouped 5-fold CV),
                           bootstrap.py (CIs), attribution.py (which generator), wild_eval.py (stress test),
                           seed_stability.py (5-seed mean ± std)
src/viz/                   figures.py, reference.py
src/predict.py, app.py     inference + Gradio interface
models_crop/               trained models (joblib / pt) and reference spectra — main pipeline
results_crop/              metrics CSV/JSON and figures — main pipeline
models/, results/          resize-normalisation ablation
examples/                  test-split clips bundled for the demo
docs/                      design, plan, notes for the report
```

## Licences and credits

Real footage: Pexels (Pexels licence; videos are not redistributed here — `manifest.csv` lists Pexels
ids/URLs so the set can be rebuilt). Metadata index: `jovianzm/Pexels-400k` (Hugging Face).
Generators: see table above (each under its own licence). Object detector: Ultralytics YOLOv8n.
Libraries: PyTorch, diffusers, transformers, scikit-learn, LightGBM, Gradio.
