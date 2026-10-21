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

## Results (main pipeline = native crop; files in results_crop/, figures in results_crop/figures/)
Final run 21 Oct 22:30: all 5 generators. Wan2.1 generation stopped at 70 clips (57 after content filter), ≈4.6 min/clip locally.

Clips after content filter (main windows, no augmentation windows): real 444 (train 310 / val 66 /
test 68), LTX 151, AnimateDiff 130, SVD 115, ModelScope 99 (test only), Wan2.1 57 (test only). Total 996.
Generation time on the laptop: LTX 104 min, AnimateDiff 147 min, SVD 388 min, ModelScope ≈100 min, Wan2.1 ≈330 min (≈18 h total).

### 1. Normalisation ablation (grouped 5-fold CV, real + LTX + AnimateDiff, LightGBM spatial+temporal)
- resize-to-256: AUROC 0.825 ± 0.073 · native 256 crop: 0.929 ± 0.023 (MLP: 0.815 → 0.931).
- Pixel baseline barely changes (0.874 → 0.897): resizing hurts frequency features specifically.

### 2. Feature-group ablation (grouped 5-fold CV, all 3 seen generators, native crop) — AUROC mean ± std
| features | LightGBM | MLP | LR | SVD-only AUROC (LGBM / MLP) |
|---|---|---|---|---|
| pixel baseline | 0.889 ± 0.007 | 0.844 ± 0.033 | 0.845 | 0.899 / 0.873 |
| spatial | 0.867 ± 0.051 | 0.906 ± 0.024 | 0.848 | 0.862 / 0.904 |
| temporal | 0.887 ± 0.044 | 0.899 ± 0.018 | 0.805 | 0.924 / 0.948 |
| spatial+temporal | 0.922 ± 0.021 | 0.919 ± 0.023 | 0.860 | 0.948 / 0.942 |
(cv_summary.csv)

### 3. Held-out test (seen generators: 68 real + 46 fake) — test_metrics.csv
- MLP all 0.965 AUROC / 0.886 acc; CNN 0.962 / 0.886; LR all 0.946 / 0.886; LGBM all 0.915 / 0.842;
  best pixel model 0.898 (SVM) / 0.890 (LR).
- Per generator (per_generator.csv), MLP all: AnimateDiff 0.973, LTX 0.931, SVD 0.995.
  Pixel (LGBM): AnimateDiff 0.960, LTX 0.882, SVD 0.839 → pixel cues fail most on content-matched SVD.

### 4. Unseen generator: ModelScope (99 fake vs 68 real test clips)
- MLP spatial 0.792, LR spatial 0.796, MLP all 0.767, CNN 0.754, LGBM all 0.682.
- Pixel baseline: 0.570–0.644 (near chance) → frequency features generalise far better than pixel stats.
- Accuracy at the 0.5 threshold is low (≈0.45–0.67) because most unseen fakes score below 0.5:
  ranking (AUROC) transfers partly, calibration does not.

### 5. Leave-one-generator-out (train on 2 seen, test on the 3rd) — logo.csv
- LGBM all: held-out LTX 0.772, AnimateDiff 0.746, SVD 0.684. CNN: 0.682 / 0.757 / 0.707.

### 6. Degradations (seen test, AUROC) — degraded.csv
| model | H.264 CRF23 | CRF30 | CRF38 | 0.5× down | blur σ1 | noise σ5 |
|---|---|---|---|---|---|---|
| CNN | 0.936 | 0.884 | 0.838 | 0.691 | 0.724 | 0.877 |
| MLP all | 0.916 | 0.836 | 0.741 | 0.624 | 0.637 | 0.912 |
| LGBM all | 0.904 | 0.827 | 0.753 | 0.709 | 0.794 | 0.852 |
| LGBM all, degradation-aware | 0.899 | 0.828 | 0.753 | 0.764 | 0.808 | 0.858 |
- Mild compression is harmless; resampling/blur remove high frequencies and hurt most; training with
  degraded copies helps for downscale/blur.

### 7. Shortcut checks
- Metadata-only LR (original width, height, fps, duration): AUROC 1.0 — file metadata alone identifies
  every generator, which is why all clips are normalised and the detector never sees metadata.
- Residual resolution effect: on real test clips, mean P(fake) rises from 0.06 (pre-downscaled to 256)
  to 0.13 (576) for MLP-all; correlation 0.14 (LGBM 0.14, CNN 0.09); all far below 0.5 → small.

### 8. Interpretability (figures)
- feature_importance_groups.png: noise-residual spectrum is the largest SHAP group, then mid-band
  frame spectrum, then residual-temporal features; top single feature = std over frames of a mid-high
  radial bin (a temporal-variability measure).
- radial_profiles.png / mean_spectrum_2d.png: each generator deviates from real differently (LTX: excess
  low-mid residual energy and HF deficit; ModelScope: strong mid-band residual peak).
- cnn_saliency.png: which frequencies/frames drive the CNN.
- per_scene.png: test subsets per scene are tiny (6–17 clips); treat as anecdotal.

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


### 9. Second unseen generator: Wan2.1 (57 fake vs 57 random real test clips) — per_generator.csv
- Frequency: MLP spatial 0.755, MLP all 0.729, CNN 0.699, LGBM all 0.702.
- Pixel baseline: 0.765–0.813 — on Wan2.1 the pixel cues transfer *better* than frequency cues,
  the opposite of ModelScope. Which cue transfers depends on the generator.

### 10. New idea after the results: frequency + pixel fusion (feature group `fusion`, 337 features)
| model | seen test AUROC / acc | ModelScope | Wan2.1 | unseen pooled | 5-fold CV |
|---|---|---|---|---|---|
| MLP fusion | 0.977 / 0.939 | 0.739 | 0.841 | 0.776 | 0.929 ± 0.018 |
| LR fusion | 0.962 / 0.921 | 0.747 | 0.795 | 0.764 | 0.900 ± 0.029 |
| LGBM fusion | 0.939 / 0.851 | 0.680 | 0.773 | 0.712 | 0.943 ± 0.018 |
| MLP frequency only | 0.965 / 0.886 | 0.767 | 0.729 | 0.751 | 0.919 ± 0.023 |
| MLP pixel only | 0.883 / 0.772 | 0.570 | 0.765 | 0.646 | 0.844 ± 0.033 |
- Degradations (seen test) MLP fusion: CRF23 0.956, CRF30 0.928, CRF38 0.824, down 0.799, blur 0.819,
  noise 0.923 — much more robust to blur/downscaling than frequency-only (0.62–0.64).
- Interpretation: the two cue types fail on different generators, so combining them gives the best
  seen performance and the best average over unseen generators. Single runs, small unseen test sets
  (57–99 fakes) — differences of ~0.03 are within noise.
- The demo app's default model is MLP fusion (0.6 s per clip on CPU).


### 11. Bootstrap 95 % confidence intervals (2000 stratified resamples; bootstrap_auroc.csv, bootstrap_diffs.csv)
| test set | MLP fusion | MLP frequency | CNN | MLP pixel |
|---|---|---|---|---|
| seen (46 fake / 68 real) | 0.977 | 0.965 [0.934, 0.988] | 0.958 [0.921, 0.986] | 0.883 [0.820, 0.936] |
| SVD only (16 / 68) | 0.980 | 0.995 [0.982, 1.000] | 0.980 | 0.825 [0.728, 0.904] |
| ModelScope (99 / 68) | 0.739 [0.661, 0.812] | 0.767 [0.693, 0.838] | 0.728 | 0.570 [0.482, 0.660] |
| Wan2.1 (57 / 68) | 0.838 [0.765, 0.905] | 0.722 [0.628, 0.809] | 0.686 | 0.778 [0.692, 0.858] |
| unseen pooled (156 / 68) | 0.776 [0.707, 0.837] | 0.751 [0.678, 0.820] | 0.713 | 0.646 [0.562, 0.724] |
Paired differences (AUROC difference, 95 % CI):
- frequency − pixel: seen +0.082 [0.031, 0.144]; ModelScope +0.197 [0.123, 0.271]; SVD +0.170 [0.092, 0.267];
  unseen pooled +0.105 [0.038, 0.171] → significant. Wan2.1 −0.056 [−0.134, 0.019] → not significant.
- fusion − frequency: Wan2.1 +0.116 [0.062, 0.175] significant; seen +0.012 [−0.007, 0.033], unseen pooled
  +0.025 [−0.012, 0.064], ModelScope −0.028 [−0.069, 0.012] → not significant.
- frequency MLP − CNN: all CIs include 0.
→ Defensible claims: frequency > pixel (seen, ModelScope, SVD); adding pixel features helps on Wan2.1;
  fusion is not a significant overall improvement.

### 12. Generator attribution (6-class: real + 5 generators; grouped 5-fold CV; attribution.csv,
attribution_confusion.png). Chance = 0.167 balanced accuracy.
| features | LR | LightGBM |
|---|---|---|
| pixel | 0.636 | 0.627 |
| spatial | 0.606 | 0.554 |
| temporal | 0.645 | 0.634 |
| spatial + temporal | 0.712 | 0.658 |
| fusion | 0.775 | 0.749 |
- Per-class recall (LR, frequency): LTX 0.77, AnimateDiff 0.75, SVD 0.86, Wan2.1 0.53, ModelScope 0.73, real 0.63.
- Temporal features alone identify SVD with 0.95 recall (LR) — image-to-video has a distinctive drift signature.
- Each generator leaves a distinguishable frequency fingerprint; Wan2.1 is the hardest (most confused with real).


### 13. Out-of-domain stress test (wild_eval.csv, wild_predictions.csv, wild_stress_test.png)
- Data: random clips from public Hugging Face collections `34data/gen-videos-*`: Sora 2 (60), Veo 3 (60),
  Kling (30), HunyuanVideo (60), "Wan2" (40; exact Wan version unknown). 1280×704–720, 24–30 fps, 8–15 s.
  Content is NOT scenery (lab demos, people, objects); prompts and post-processing unknown. Accessed 21 Oct 2026.
- Compared against the 68 real test clips. ALL clips (fake and real) go through the app's deployment path
  (src/predict.py: first 2 s at 8 fps, short side > 576 px downscaled to 512, native 256 crop, H.264 CRF 18).
- AUROC (all clips):
| source | CNN | MLP freq | MLP fusion | LGBM freq | MLP pixel |
|---|---|---|---|---|---|
| Sora 2 | 0.873 | 0.856 | 0.852 | 0.846 | 0.727 |
| HunyuanVideo | 0.805 | 0.722 | 0.657 | 0.665 | 0.566 |
| Wan2 | 0.731 | 0.731 | 0.756 | 0.638 | 0.715 |
| Veo 3 | 0.694 | 0.698 | 0.763 | 0.675 | 0.769 |
| Kling | 0.706 | 0.609 | 0.615 | 0.533 | 0.547 |
| mean | 0.762 | 0.723 | 0.729 | 0.671 | 0.665 |
- Scenery-only subset (YOLO filter; only 6–16 clips per source) — similar ordering, Sora 2 CNN 0.941.
- Detection rate at 0.5 is low (Sora 2 45–70 %, Veo 3 7–65 %, Kling 13–23 %) and real false-positive rate on the
  deployment path is 7 % (CNN) – 26 % (pixel): ranking transfers partly, calibration does not.
- Takeaways: frequency features rank Sora 2 fakes well (0.85–0.87) without ever seeing Sora; the spectro-temporal
  CNN generalises best out of domain (mean 0.76) though it ties the MLP in domain; pixel stats are worst on average
  but best on Veo 3. Caveat: content/encoding confounds — this is a stress test, not a controlled comparison.
