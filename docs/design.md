# Design — Detection of AI-Generated Video via Frequency Domain (IE 643)

Author: Astitva Garg (24B2440), team *mdhkideggimirch* (solo). Written 2026-10-21.

## 1. Intent and why

Course project. Build a lightweight classifier that says whether a short clip is camera-captured or
diffusion-generated, using spatial frequency-domain features per frame plus temporal features
(frame-to-frame frequency drift). It must cover the five allotted deliverables, have a live test
interface, and produce enough results/figures for a report, ≤18-slide deck, 10-min code-walkthrough
video and 5-min demo video.

**Hard dates**
- Stage 1 (report PDF, slides, two videos): **25 Oct 2026, 23:59 IST**. Today is 21 Oct.
- In-person presentation + live demo: 26–31 Oct (12 min + 3 min Q&A).
- Novelty assessment: 23/25 Nov (out of scope here, but we leave hooks for it).

So: working pipeline + numbers + figures must exist by **~24 Oct morning**, leaving the user ~1.5 days
for writing and recording.

## 2. Goals

1. **Curated dataset** — balanced real vs AI clips, ≥4 open T2V/I2V generators, content restricted to
   scenes without prominent humans, animals or common objects (landscapes, water, sky, fire, foliage,
   textures, abstract motion, city lights at a distance).
2. **Feature pipeline** — per-frame spatial frequency features + clip-level temporal drift features.
3. **Lightweight classifier(s)** on those features.
4. **Evaluation** — seen generators, unseen (held-out) generator(s), degraded videos.
5. **Visualisation/interpretability** — spectra, drift plots, feature attribution, error gallery.
6. **Interface** — upload a video → verdict + probability + its spectrum/drift plots + top contributing features.
7. **Metrics displayed** — accuracy, AUROC, per-generator breakdown, shown in the interface too.

**Mapping to the allotted brief** (5 deliverables + course submission items):
D1 dataset → goal 1 · D2 features → goal 2 · D3 model → goal 3 · D4 generalisation/robustness →
goal 4 · D5 visualisation → goal 5. Course items: trained models + documented code (README) ·
interface → goal 6 · displayed metrics + qualitative results → goal 7 · stage-wise report (user writes).
Cut order if time runs short: leave-one-generator-out table → UMAP → degradation-aware training
variant → 5th generator. Never cut: the 1D-CNN (needed for the DL parts of report/walkthrough), the
ablation, unseen + degraded eval, content filter, the interface.

**Calls made by the designer (not confirmed by user):** demo runs on the user's laptop offline
(Gradio local, bundled example clips), no hosted app; course instructions restrict report text only,
not code or use of open generators/stock footage — all external models/data are cited.

## 3. Non-goals / anti-goals

- No large end-to-end video networks (DeMamba-style). "Lightweight" is in the brief.
- No cloud GPU in the first pass (user decision). Revisit only if local generation clearly can't finish.
- **Anti-goal: the AI writes the report.** Course rules give zero for AI-generated report text and
  run a plagiarism check. We produce code, results, figures, tables and factual bullet notes
  (`docs/notes_for_report.md`); the user writes the prose.
- Anti-goal: a detector that wins on trivial shortcuts (resolution, frame rate, codec, duration,
  content). See §5.3 — this is the main technical risk and must be controlled.

## 4. Constraints

- Hardware: RTX 4060 Laptop 8 GB VRAM, 15 GB RAM, Ryzen 7 7840HS, 124 GB free on C:. Windows 11.
- Python: separate `uv` venv on Python 3.11 (system 3.14 is too new for some deps). ffmpeg via
  `imageio-ffmpeg` / PyAV wheels (no system install needed).
- RAM limit means big text encoders (T5-XXL / UMT5) must be run alone: encode all prompts, cache
  embeddings to disk, free the encoder, then load the video model.
- Real footage: Pexels clips via the public Pexels-400k metadata (no API key needed, see §5.1), plus
  optional user-shot phone clips as an extra real test set.

## 5. Decided approach

### 5.1 Data

**Real source (updated 21 Oct):** Pexels has paused new API keys. Instead we use the public
`jovianzm/Pexels-400k` metadata parquet on Hugging Face (title, duration, thumbnail, direct
`pexels.com/download/video/<id>` link — verified to redirect to the HD mp4 without any key). We
filter titles by scene keywords and exclude people/animal/object words, then download directly
(~12 MB per HD clip, ~6–10 GB total), politely rate-limited.

**Real** (~400–500 clips): Pexels videos selected by scene keywords (ocean waves, waterfall,
clouds timelapse, forest canopy, campfire, rain on window, desert dunes, snowfall, river, aurora,
lava, smoke, abstract ink, city skyline at night...). Download a mid-resolution rendition (~720p),
cut into a 2–4 s segment. Plus optional user phone clips as an extra real test set.

**Fake**, generated locally from a shared prompt list that mirrors the real search keywords:

| Generator | Type | Role | Target clips |
|---|---|---|---|
| LTX-Video (2B, distilled) | T2V | seen | ~120 |
| AnimateDiff (SD1.5 + motion adapter) | T2V | seen | ~120 |
| Stable Video Diffusion XT | I2V, conditioned on first frames of *real* Pexels clips | seen | ~80 |
| Wan2.1-T2V-1.3B *(or CogVideoX-2B if Wan is too slow)* | T2V | **unseen** | ~60 |
| ModelScope T2V 1.7B | T2V (older, low-res) | **unseen** | ~80 |

**Content filter** (brief: avoid prominent humans/animals/common objects): every real clip and every
generated clip passes an automatic check — an off-the-shelf object detector (YOLO, COCO classes:
person, animals, vehicles, furniture, etc.) on 4 sampled frames; a clip is rejected if a detected
object covers more than ~5% of the frame. Rejection counts are logged for the report. Prompts and Pexels
queries are written to avoid these from the start (no windows, cars, boats, buildings in close-up).

**Labelled manifest**: `data/manifest.csv` — one row per clip: id, label (real/fake), generator, prompt
or Pexels id + URL + author, source video id, split, duration, fps, original resolution, content-filter
result. This is the "curated, labelled dataset" artefact. Video files are not redistributed (Pexels
license); the manifest + scripts let anyone rebuild it.

*Deviation from prep slides:* the slides shortlisted SVD, AnimateDiff, CogVideo and ModelScope "to be
confirmed based on compute". We keep SVD, AnimateDiff, ModelScope, add LTX-Video (fast on 8 GB), and
use Wan2.1-1.3B instead of CogVideoX-2B because CogVideoX needs heavy CPU offload on 8 GB and 15 GB RAM
(~10+ min/clip). CogVideoX-2B is the fallback if Wan fails.

Exact counts flex with measured speed; the generation script logs seconds/clip and is resumable.
Generation runs in the background from day 1. If after ~24 h a generator is far behind, we cut its
count rather than slip the schedule. Seen/unseen assignment can be revised once we see the data.

**Splits**: real clips split by *source video* into train/val/test (70/15/15) so segments of one video
never leak across splits. Seen-generator fakes split the same way (by prompt / source image). Unseen
generators are test-only. Balanced real:fake in each split (subsample real for the unseen test).

**Degraded set**: test clips (real + fake) re-encoded with H.264 at CRF 23/30/38, downscaled 0.5×,
Gaussian blur, additive noise. Generated programmatically from the test split.

### 5.2 Features (per clip: resample to 8 fps, 16 frames = 2 s, as in the prep slides' 2–4 s / 8–16 fps)

16 frames at 8 fps is the common denominator of all generators (AnimateDiff/ModelScope produce 16
frames at 8 fps natively); real clips with enough length additionally get a second 2 s window for
training-time augmentation only. Radial spectrum uses 48 bins (slides: 30–50).

Per frame (luminance, centre-cropped to a fixed 256×256 after resizing short side to 256):
- **Azimuthal (radial) power spectrum** of log|FFT|², 1D, ~64 bins, normalised.
- Spectral slope (fit of log-power vs log-frequency), high-frequency energy ratio (several cut-offs),
  spectral centroid, roll-off.
- **DCT 8×8 block band energies** (low/mid/high), and **high-pass residual** spectrum (image minus
  denoised/blurred version) to emphasise generator noise over content.
- Angular energy distribution (a few orientation wedges) — catches directional artefacts.

Clip-level spatial = mean and std over frames of the above.

Temporal (the "frequency drift"):
- Frame-to-frame change of the radial spectrum (L2 / cosine distance between consecutive spectra) and
  of each scalar statistic → mean, std, max, and lag-1 autocorrelation of the drift series.
- Second-order drift (change of change) — smoothness vs jerkiness.
- Per-band drift: drift computed separately in low / mid / high bands.
- Temporal spectrum of the high-pass residual (FFT along time of per-pixel residual energy, pooled) —
  sensor noise is temporally independent in real video, generated clips tend to be correlated.

Total ~150–300 dims. Features cached to `features/*.parquet` so models retrain in seconds.

### 5.3 Shortcut control (important)

Generators output fixed resolutions/frame counts/fps; Pexels is H.264 at varied resolution. To stop
the model learning "resolution" instead of "generated":
- Every clip (real and fake) is normalised the same way **before** feature extraction: resize short
  side to the same size, centre-crop 256×256, resample to a fixed fps, and **re-encode both real and
  fake through the same H.264 setting** once. This last step is our main defence and is applied
  uniformly.
- Sanity check: a classifier on metadata only (orig. resolution, fps, duration, bitrate) is reported;
  and a check that features don't simply encode resolution.

### 5.4 Models

- Logistic regression (standardised features) — interpretable baseline.
- Gradient-boosted trees (LightGBM) — main model expected.
- Small MLP (PyTorch, 2 hidden layers, dropout, BCE loss, AdamW, cosine LR, early stopping on val
  AUROC) — the "deep learning" component for the report/walkthrough.
- **Ablation**: spatial-only vs temporal-only vs spatial+temporal. This directly tests the project's
  hypothesis that drift adds signal.
- Linear SVM alongside LR (both named in the prep slides).
- **Spectro-temporal 1D-CNN** over the per-frame radial-spectrum sequence (16 frames × 48 bins, plus
  frame-difference spectra as a second channel), ~50k params, BCE loss, AdamW, cosine schedule. This
  is the main deep-learning component the report and code walkthrough need (architecture, loss,
  optimiser, LR schedule), and it learns drift directly instead of from hand-made summaries.
  Gradient attribution on its input gives a frames × frequency-bins saliency map.
- Training with degradation augmentation (train on mix of clean + re-encoded) as a robustness
  variant — compare against clean-only training.

### 5.5 Evaluation

Accuracy, AUROC, F1, per-generator detection rate, confusion matrices, on:
seen-test, each unseen generator, each degradation type/level. Leave-one-generator-out (train on all
seen but one, test on the left-out) as an extra generalisation table — cheap because features are cached.

### 5.6 Visualisation

- Mean log-spectrum heatmaps (real vs each generator) and their difference maps.
- Mean radial power spectra with confidence bands per class/generator.
- Drift-over-time plots for example real vs fake clips.
- Feature importance (LR coefficients, LightGBM gain, SHAP) aggregated by feature group (spatial
  bands, temporal drift, residual).
- **Importance over frequency**: model importance mapped back onto radial-frequency bins (which
  frequencies decide), and the CNN saliency map over frames × frequency (which frequencies, *when*).
- Per-clip SHAP explanation (used in the interface).
- t-SNE/UMAP of feature space coloured by generator.
- Error gallery: misclassified clips with thumbnails + spectra.

### 5.7 Interface

Gradio app (`app.py`): upload video (or pick from bundled example clips) → model choice → verdict,
P(fake), mean spectrum vs reference real/fake means, drift plot, top-10 SHAP features, frame
thumbnails. A second tab shows the evaluation dashboard (metrics tables + figures). Runs locally on CPU.
Reviewers may bring their own clips during the viva, so the app must accept any common format
(mp4/mov/webm/avi/mkv), any resolution, length or fps (it uses the first ≤4 s, or lets the user pick a
start time), answer in under ~10 s, and show a caution note when a clip has prominent people/animals
(out of the training domain) instead of silently failing. An "optional degradation" control lets the
reviewer compress/blur the input live and watch the prediction change (robustness demo).

### 5.8 Novelty angle (to present)

1. **Content-matched I2V pairs** — SVD fakes animate the first frame of a real clip, so real/fake
   pairs share content; any detection signal on those pairs must come from the generation process.
2. **Band-wise temporal drift + residual temporal spectrum** as a compact temporal signature, with an
   ablation showing what it adds over spatial-only features.
3. Plan for novelty assessment: degradation-aware training / domain generalisation across generators.

## 6. Deliverable artefacts

```
ie643_aivideo/
  README.md               setup, usage, dependencies
  requirements.txt / pyproject
  configs/prompts.yaml    scene prompts + Pexels queries
  src/data/               pexels download, clip cutting, normalisation, degradation
  src/generate/           one script per generator (resumable, logs timing)
  src/features/           spatial.py, temporal.py, extract.py
  src/models/             train.py, mlp.py, evaluate.py
  src/viz/                figures
  app.py                  Gradio interface
  models/                 trained model files
  results/                metrics json/csv, figures (png) for report & slides
  docs/notes_for_report.md   factual notes (user writes the prose), organised by the report template:
                          work before vs after prep-presentation review, workflow, dataset
                          extraction steps, preprocessing, every experiment with settings,
                          hardware, results tables, interface methodology, novelty plan, references
                          in the course's required format
data/ (outside OneDrive, C:\ie643_data)  raw/real, raw/fake/<gen>, clips/, degraded/
```

GitHub repo: user creates it and pushes (code + small models; no video data).

## 7. Schedule

| When | What |
|---|---|
| 21 Oct (today) | env, prompt list, Pexels download, start LTX + AnimateDiff generation in background; feature code |
| 22 Oct | SVD + Wan/ModelScope generation continue; normalisation; first features + baseline models on partial data |
| 23 Oct | generation finishes ~evening; full training, ablations, unseen/degraded eval, all figures; Gradio app |
| 24 Oct | polish app, results notes, slide figures; user writes report + slides, records videos |
| 25 Oct | buffer; submit by 23:59 |

## 8. Left to the builder

Exact feature bins/cut-offs, hyperparameters, prompt wording, resolution/frame settings per
generator (fit to 8 GB), which LTX/SVD checkpoints, scene keyword list details.
