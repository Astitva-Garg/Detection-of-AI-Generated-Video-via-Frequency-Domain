# Results map: where every reported number and figure comes from

Every number in the report and the slides should be traceable to a file in `results_crop/` (the main pipeline) and
to the stage of `run_pipeline.sh` that writes it. If a number is not in this table, it is not backed by a file.

Stage names are those of `bash run_pipeline.sh list`. `results/` and `models/` hold the separate resize ablation
(see the last section).

## Numbers

| What is reported | File (in `results_crop/` unless noted) | Stage |
|---|---|---|
| **Main comparison** (AUROC, mean and std over 5 seeds): MLP fusion, MLP frequency, CNN, MLP pixel on seen, ModelScope, Wan2.1 and five public sources | `seed_stability.csv` (per seed: `seed_stability_runs.csv`) | `seeds` |
| Same table for the pixel CNN (from scratch) and the fine-tuned ResNet-18 | `pixel_deep.csv` (per seed: `pixel_deep_runs.csv`, training curves: `pixel_cnn_history_seed*.csv`, `resnet18_history_seed*.csv`) | `deep` |
| Held-out test metrics of the shipped models (AUROC, accuracy, F1, detection and rejection rate), seen and pooled unseen | `test_metrics.csv` | `evaluate` |
| Per-generator and per-scene results | `per_generator.csv`, `per_scene.csv` | `evaluate` |
| Robustness to the six degradations | `degraded.csv` | `evaluate` |
| Leave-one-generator-out | `logo.csv` | `evaluate` |
| Metadata-only shortcut baseline (AUROC 1.0) | `metadata_baseline.csv` | `evaluate` |
| Grouped 5-fold cross-validation (LR, LightGBM, MLP x 5 feature groups) | `cv_summary.csv`, `cv_folds.csv` | `cv` |
| Bootstrap 95 % intervals and paired differences | `bootstrap_auroc.csv`, `bootstrap_diffs.csv` | `bootstrap` |
| Six-class generator attribution | `attribution.csv` | `attribution` |
| Public-clip stress test (Sora 2, Veo 3, Kling, HunyuanVideo, Wan2) | `wild_eval.csv`, `wild_predictions.csv` | `wild_eval` |
| Validation metrics used for model selection, CNN training history | `val_metrics.csv`, `cnn_history.csv` | `train` |
| Model-by-feature-group table (seen and unseen) | `ablation.csv` | `figures` |
| Feature importance (SHAP, LightGBM) | `feature_importance.csv` | `figures` |
| Dataset counts and splits | `data/manifest_crop.csv` (also printed by `python -m src.demo_live`) | `dataset` |
| Object filter, audit | `data/content_filter.csv`, `data/audit.csv`, `data/audit_duplicates.csv` | `filter`, `audit` |
| Live comparison of pixel, frequency and fusion on the held-out clips, with intervals | `demo_output/04_comparison.csv` | `python -m src.demo_live` |

## Figures (`results_crop/figures/`)

| Figure | Written by |
|---|---|
| `dataset_examples`, `mean_spectrum_2d`, `radial_profiles`, `radial_deviation`, `drift_examples`, `drift_distributions` | `src/viz/figures.py` (stage `figures`) |
| `feature_importance_groups`, `importance_over_frequency`, `cnn_saliency`, `cnn_training_curves` | `src/viz/figures.py` |
| `roc_curves`, `confusion_matrices`, `ablation_bars`, `per_scene`, `degradation_bars`, `tsne`, `error_gallery` | `src/viz/figures.py` |
| `seed_stability` | `src/eval/seed_stability.py` (stage `seeds`) |
| `attribution_confusion` | `src/eval/attribution.py` (stage `attribution`) |
| `wild_stress_test` | `src/eval/wild_eval.py` (stage `wild_eval`) |
| Live demo figures (`01_examples`, `02_spectra_live`, `04_comparison`) | `src/demo_live.py` -> `demo_output/` |

Two files that no script wrote were removed from this folder: `fusion_unseen.csv` (numbers from an earlier run, no
longer matching the final results) and `figures/svd_source_pairs.png`.

## Resize versus native crop (stage `ablation`)

| What is reported | File | Stage |
|---|---|---|
| Held-out AUROC of crop and resize on identical clips and splits, with paired bootstrap intervals (pixel, frequency, fusion x LR, LightGBM, MLP; seen, ModelScope, Wan2.1) | `results_crop/resize_ablation.csv` | `ablation` |
| Grouped 5-fold CV of both variants side by side | `results_crop/resize_ablation_cv.csv`; the resize CV itself is `results/cv_summary.csv`, the crop CV is `results_crop/cv_summary.csv` | `ablation`, `cv` |

Headline (grouped CV, seen generators, frequency features = spatial + temporal): LightGBM 0.906 with the crop and
0.832 resized (-0.074), MLP 0.909 and 0.825 (-0.084), logistic regression 0.876 and 0.782 (-0.094); the pixel
baseline is unchanged (LightGBM 0.884 and 0.880). An earlier run on a smaller clip set, quoted in earlier drafts as
"0.929 to 0.825", is superseded by these like-for-like files.

## Extra experiments (stage `extras`; clips of the prompt-disjoint test: stage `generate_fresh`)

| What is reported | File (`results_crop/`) | Script |
|---|---|---|
| Frame order: AUROC of the shipped models when the held-out clips' frames are reversed or shuffled (features recomputed), with paired intervals. Spatial-only models cannot change (check); reversal changes nothing; shuffling costs the temporal features 0.14 (MLP) to 0.20 (LightGBM) on seen generators | `temporal_order.csv` | `src/eval/temporal_order.py` |
| Prompt-disjoint test: ModelScope regenerated from 36 new prompts (`configs/fresh_prompts.yaml`, 72 clips), scored by the shipped models with no retraining, next to the old-prompt ModelScope test clips. Every pixel and spectral model loses 0.05 to 0.15 AUROC (ResNet-18: 0.905 to 0.893); frequency still beats pixel (MLP +0.259, interval [+0.173, +0.341]) | `fresh_prompts.csv`, `fresh_prompts_diffs.csv` | `src/eval/fresh_prompts.py` |
| Frequency + ResNet-18 late fusion (a 2-weight logistic stacker fitted on the seen validation clips only). Negative result: it raises the frequency MLP on unseen generators (+0.06 to +0.08) but does not improve ResNet-18 (seen +0.004 [-0.028, +0.038], ModelScope -0.055, Wan2.1 -0.195) | `late_fusion.csv` | `src/eval/late_fusion.py` |

Caveats: the ResNet-18 here is the single stored seed-643 checkpoint; the prompt-disjoint test has one unseen generator
(ModelScope) and 72 clips; the fresh clips are not content-filtered by YOLO (the prompts exclude people and vehicles).

## Poster (`bash run_pipeline.sh poster`, or `python -m tools.build_poster`)

`tools/build_poster.py` regenerates every figure below from the files named, then prints `poster/poster.html` to
`poster/poster.pdf` with headless Chrome or Edge.

| Poster item | Source file | Written by |
|---|---|---|
| Figure 1: how the spectrum maps are made (one real frame, its spectrum, the real average, LTX minus real) and each generator's difference from real (40 clips per source) | `data/manifest_crop.csv`, `data/clips_crop/` | `tools/poster_spectra.py` |
| Figures 2 and 3: workflow and architecture diagrams (Figure 3 holds the real curves of one held-out clip and the shipped model's verdict) | `data/clips_crop/`, `results_crop/predictions.csv` | `tools/poster_diagrams.py` |
| Figure 4: resize vs native crop | `resize_ablation_cv.csv` | stage `ablation` |
| Figure 5: frame shuffling | `temporal_order.csv` | stage `extras` |
| Figure 6: pixel vs frequency vs fusion with intervals | `live_comparison.csv` | stage `demo` (`python -m src.demo_live`) |
| Table 3 (all models, 5 seeds; Figure 7 was dropped as it repeated this table) | `seed_stability.csv`, `pixel_deep.csv` | stages `seeds`, `deep` |
| Table 1: features in groups A, B and control (counts and definitions) | `src/features/spatial.py`, `temporal.py`, `pixel_baseline.py` | by hand from the code |
| Table 2: clips per source and split | `data/manifest_crop.csv` | stage `dataset` |
| Numbers in the novelty text | `resize_ablation_cv.csv`, `temporal_order.csv`, `fresh_prompts*.csv`, `late_fusion.csv`, `bootstrap_diffs.csv` | stages above |
| QR code of the repository link | the link itself | `tools/make_poster_figures.py` (`qr`, needs the optional package `segno`) |
| IITB and IE 643 logos | the course poster template | not generated |

## What is not backed by a file

- Values quoted in earlier drafts of the report, slides and notes that differ from the files above are superseded.
- The GPU-trained CNN is seeded but not bit-exact: regenerated CNN rows differ slightly from the stored ones.
