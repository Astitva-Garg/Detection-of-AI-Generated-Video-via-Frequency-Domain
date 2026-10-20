# Notes for the report (facts only — write the report text yourself)

> The course gives zero marks for AI-written report text and runs a plagiarism check.
> Everything below is raw material: facts, numbers, settings and the reasons behind decisions.
> Put it into your own words. Sections follow the report template.

## Work before the prep-presentation review
- Literature survey: GAN frequency artefacts (Frank 2020, Durall 2020), diffusion high-frequency deficit
  (Ricker 2024, Corvi 2023), Synthbuster, BiHPF/FrePGAN, video detectors (DeMamba/GenVideo, DIVID,
  He et al. local/global temporal defects, D3, Kim et al. temporal frequency).
- Problem formulation: binary clip classification, 2–4 s clips, 8–16 fps, ~30–50 radial bins/frame,
  ~6–10 temporal drift statistics. Candidate generators: SVD, AnimateDiff, CogVideo, ModelScope.
- Planned: LR / linear SVM baseline, shallow MLP / small CNN, seen/unseen/degraded evaluation, spectrum
  heatmaps, drift plots, attribution, error gallery.

## Work after the review (this phase)
- Built the complete pipeline (below), generated the dataset locally, trained/evaluated all models,
  built the interface. Changes vs. the prep plan and why:
  - Generators: kept SVD, AnimateDiff, ModelScope; added LTX-Video (fast on 8 GB) and Wan2.1-1.3B in place
    of CogVideoX-2B (CogVideoX needed heavy CPU offload on 8 GB VRAM / 15 GB RAM, ~10+ min per clip).
  - Real data: Pexels API key issuance was paused → used the public `jovianzm/Pexels-400k` metadata index
    and Pexels' direct download links (no key needed).
  - Added an automatic content filter (YOLOv8n) to enforce the "no prominent humans/animals/common
    objects" constraint objectively, instead of relying only on prompt/title wording.
  - Added shortcut controls (identical normalisation + re-encoding of real and fake, pre-downscaling of
    real clips to generator-like resolutions, metadata-only baseline).
  - Added a spectro-temporal 1D-CNN (deep model) next to the feature-vector classifiers.
  - Added content-matched SVD pairs (SVD animates the first frame of a real clip) as a control for scene
    content.
  - Added degradation-aware training as a robustness variant.

## Workflow (bullet list for the Workflow section / slide)
1. Scene vocabulary (`configs/scenes.yaml`): 18 scene types (water, sky, fire, terrain, vegetation,
   weather, abstract), title keywords, an exclusion word list, 3–4 T2V prompts per scene (66 prompts).
2. Real clip collection → 3. AI clip generation (5 generators) → 4. content filter → 5. normalisation →
   6. manifest + group splits + degraded copies → 7. features → 8. models → 9. evaluation →
   10. visualisation → 11. interface.

## Data
### Real (camera-captured)
- Source: Pexels (free stock video). Index: Hugging Face dataset `jovianzm/Pexels-400k` (400,476 rows:
  title, duration, view count, SFW flag, download URL).
- Selection: SFW, duration 5–60 s, title contains a scene keyword and none of ~150 excluded words
  (people, animals, vehicles, buildings, objects, "3d/animation/render" etc.); tag-style titles
  ("Free stock video of …") dropped. 37,460 candidates → capped at 40 per scene → 560 sampled,
  544 downloaded (rest: dead links). Mostly 1080p–4K, 24–60 fps.
- Content filter rejected <FILL from content_filter.csv> real clips (top classes: person, dog, …).
- Per real clip: main 2 s window starting at 1 s; train clips ≥ 7 s give a second window at 4 s
  (augmentation, train only).

### AI-generated
| Generator | Type | Resolution, frames, fps | Steps | Seconds/clip on RTX 4060 | Role |
|---|---|---|---|---|---|
| LTX-Video 2B | T2V (DiT) | 704×480, 49 f, 24 fps | 40 | ~43 | seen |
| AnimateDiff (SD1.5 epiCRealism + motion adapter v1-5-2) | T2V (UNet + motion modules) | 512×512, 16 f, 8 fps | 25 DDIM | ~54 | seen |
| Stable Video Diffusion XT | I2V from real first frame | 768×448, 16 f (first 2 dropped), 7 fps | 20 | ~130 | seen |
| ModelScope T2V 1.7B | T2V (UNet3D) | 256×256, 16 f, 8 fps | 30 DPM++ | ~20 | unseen |
| Wan2.1 T2V 1.3B | T2V (DiT, flow matching) | 832×480, 33 f, 16 fps | 30 | ~286 | unseen |

- Prompts: the shared 66 scene prompts, cycled; a different seed each clip; same negative prompt
  (people, animals, text, watermark, cars, buildings, low quality).
- Memory tricks for 8 GB VRAM / 15 GB RAM: T5-XXL / UMT5-XXL text encoders run alone (bf16, split
  GPU/CPU), prompt embeddings cached to disk, encoder freed before loading the video model; SVD resolution
  reduced from 1024×576×25 (150 s/step due to VRAM overflow) to 768×448×16 (6.3 s/step).
- SVD artefact: first 1–2 decoded frames are smeared → dropped so the classifier cannot use it.
- Counts per generator/split: <FILL from results/summary.json "counts">.

### Normalisation (identical for real and fake)
- Sample 16 frames at 8 fps (2 s) by nearest timestamp; loop only if the clip is shorter than 2 s.
- Real only: downscale to a short side drawn from {256, 320, 480, 512, 576} (the generators' native
  sizes) and H.264-encode (CRF 18) — so real clips go through a similar resample + compression chain as
  generated ones (prevents a "4K vs 512 px" shortcut).
- All: resize short side to 256 (area interpolation when shrinking), centre-crop 256×256, write H.264
  CRF 18 at 8 fps.

### Splits
- Group-level hash split 70/15/15 (seed 643): real clips grouped by source video; T2V clips grouped by
  prompt; SVD clips share the group (and split) of the real clip they animate. Unseen generators are
  test-only. Balanced unseen evaluation pairs each unseen generator with an equal number of random real
  test clips.

### Degraded set
- Every test clip: H.264 CRF 23 / 30 / 38, 0.5× down-up sampling, Gaussian blur σ=1, Gaussian noise σ=5.
- Every train clip: one randomly assigned degradation (used only by the degradation-aware model).

## Features (263 per clip)
Per frame, on luminance, Hann window, mean removed:
- 48-bin azimuthally averaged log power spectrum (mean removed → shape not brightness).
- 48-bin radial spectrum of the noise residual (frame − 3×3 median) — emphasises generator noise.
- Spectral slope (log power vs log frequency fit), high-frequency energy ratios above 25/50/75 % of
  Nyquist, spectral centroid, 85 % roll-off.
- 8×8 block-DCT energy fractions (low / mid / high zig-zag bands).
- Energy in 8 orientation wedges above 25 % Nyquist.
Clip level: mean and std over the 16 frames (prefix `sp_`).
Temporal drift (prefix `drift_` / `rt_`):
- L2 and cosine distance between consecutive radial spectra → mean, std, max, lag-1 autocorrelation;
  second-order drift (|Δ of drift|); drift in low / mid / high thirds of the spectrum.
- |Δ| of slope, HF ratio, centroid, DCT-high per frame → mean, std, max, second-order.
- Residual-temporal: per-pixel lag-1 temporal correlation of the noise residual, fraction of temporal
  frequency energy in the upper half, residual energy, drift of the residual spectrum.
- CNN input: (2, 16, 48) — per-frame radial spectra and radial spectra of frame differences.
- Extraction time ≈ 0.25 s per clip on CPU.

## Models
- Logistic regression (standardised, class-balanced, C ∈ {0.01, 0.1, 1, 10} chosen on val AUROC).
- Linear SVM (same C grid, Platt-calibrated via 3-fold CalibratedClassifierCV).
- LightGBM (600 trees max, lr 0.05, 15 leaves, feature fraction 0.5, bagging 0.8, early stopping 50
  rounds on val AUC).
- MLP: in → 128 → 64 → 1, ReLU, dropout 0.3; BCEWithLogits loss with positive-class weight; AdamW
  lr 1e-3, weight decay 1e-4; cosine LR schedule over 200 epochs; batch 64; early stopping (patience 20)
  on validation AUROC.
- Spectro-temporal CNN: shared frequency encoder per frame (Conv1d 2→16, k5 → Conv1d 16→32, k5,
  stride 2 → adaptive pool to 8) → temporal Conv1d stack (256→64→64, k3) over the 16 frames → mean+max
  pooling over time → dropout 0.3 → linear. <FILL param count> parameters. BCEWithLogits with positive
  weight, AdamW lr 1e-3, wd 1e-4, cosine schedule, 60 epochs, batch 32, early stopping (patience 10) on
  val AUROC, time-reversal augmentation, per-bin standardisation from train statistics.
- Feature groups for the ablation: spatial only (`sp_`), temporal only (`drift_`, `rt_`), all.
- Degradation-aware LightGBM: train on clean + degraded train clips.

## Experiments / hardware
- Hardware: Windows 11, AMD Ryzen 7 7840HS, 15 GB RAM, NVIDIA RTX 4060 Laptop 8 GB. Python 3.11,
  PyTorch 2.11 (CUDA 12.8), diffusers 0.41, transformers 5.20, scikit-learn, LightGBM 4.x.
- Generation time: <FILL total hours from queue.log>.
- Experiments: (1) model × feature-group grid; (2) seen vs unseen generator test; (3) degradations;
  (4) leave-one-generator-out (train on 2 seen generators, test on the third); (5) metadata-only
  shortcut baseline; (6) degradation-aware training; (7) CNN vs feature models.

## Results
<FILL — copy numbers from results/test_metrics.csv, per_generator.csv, degraded.csv, logo.csv,
metadata_baseline.csv; figures in results/figures/>

## Interface methodology
- Gradio 2-tab app (`app.py`). Upload → identical preprocessing to training (16 frames at 8 fps,
  256×256, H.264 CRF 18 round trip) → features → chosen model → verdict.
- Explanations: frames + 2D spectra; clip radial/residual spectrum vs training-set real/fake means;
  drift-over-time plots; top feature contributions (LightGBM SHAP values / linear coef × value).
- Robustness control: user can apply a degradation before analysis.
- Guards: short clips looped with a note; low resolution warning; YOLO check warns when a prominent
  person/animal/object is present (out of training domain).
- Performance tab: metric tables + figures.

## Limitations (honest)
- Small dataset (hundreds of clips per class), single-prompt-set; real clips are stock footage
  (professionally shot, often graded) — results may not transfer to phone footage.
- Content limited to scenery by design.
- Generators run at reduced settings to fit 8 GB.

## Plan for novelty assessment (ideas)
- Degradation-aware / frequency-perturbation training for robustness (cf. FrePGAN).
- Generator-agnostic features: learn on residual temporal spectra only; domain generalisation across
  generators (leave-one-out training objective).
- Larger, more recent generators (cloud GPU), and phone-captured real footage.
