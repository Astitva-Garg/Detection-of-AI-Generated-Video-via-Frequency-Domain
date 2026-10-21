# Video outlines (talking points, not a script — say it in your own words)

Rules from the instructions: code walkthrough ≤ 10 min, demo ≈ 5 min, face visible (webcam overlay),
clear audio, readable screen (zoom the editor to ~150 %). Formats: .mp4/.mkv/.mov. OBS: add a
"Display capture" + "Video capture device" source, set output to mp4, 1080p, 30 fps.

Before recording: close other apps; open the files below as tabs in VS Code in this order; have a
terminal in the repo root with the venv active; have `python app.py` already running for the demo.

---

## A. Code walkthrough (target 9:30)

| Time | Show | Points to make |
|---|---|---|
| 0:00–0:30 | Webcam + README pipeline diagram | Task in one sentence; repo layout; everything is a resumable script stage. |
| 0:30–1:30 | `configs/scenes.yaml`, `src/data/pexels.py` (`select_titles`, `main`) | Scene vocabulary shared by real selection and prompts (keeps content comparable). Pexels API paused → Pexels-400k index + direct links. Keyword include/exclude, duration filter, cap per scene. |
| 1:30–2:45 | `src/generate/common.py` (`Job`, `run_jobs`), `src/generate/text_embed.py`, `ltx.py`, `svd.py` | One runner for all 5 generators: resumable manifest, failures logged and retried. 8 GB tricks: encode prompts with the big T5 alone, cache, free memory. SVD animates the first frame of a real clip → content-matched pairs; first 2 frames dropped (smear artefact). Mention per-generator settings table. |
| 2:45–3:30 | `src/data/content_filter.py`, `src/data/normalize.py` | YOLO rejects clips where a person/animal/object box > 5 % of frame. Normalisation identical for real and fake: 16 frames @ 8 fps, real pre-downscaled to a generator-like size + H.264, then 256² crop + H.264 CRF 18 → removes resolution/codec shortcuts. |
| 3:30–4:00 | `src/data/splits.py`, `build_dataset.py` (`candidate_rows`) | Group-level hash split 70/15/15: all windows of one source video / all seeds of one prompt stay in one split; SVD fake inherits the split of its source clip; unseen generators test-only. Degraded copies. |
| 4:00–5:30 | `src/features/spatial.py` (`power_spectrum`, `_radial_mean`, `frame_features`) | Hann window, FFT, log power, azimuthal average into 48 bins; residual spectrum (frame − median); slope, HF ratios, centroid, roll-off, DCT bands, angular wedges. Explain why: diffusion = HF deficit / spectral shape changes. |
| 5:30–6:30 | `src/features/temporal.py` (`drift_features`, `residual_temporal`, `clip_features`) | Frequency drift = distance between consecutive radial spectra (L2/cosine), its autocorrelation and second order, per-band drift, scalar drift; residual temporal correlation (sensor noise is independent frame to frame). CNN input tensor (2,16,48). |
| 6:30–8:00 | `src/models/classic.py` (`make_model`, `fit_model`, `TorchMLPClassifier.fit`), `src/models/cnn.py` (`SpecTempCNN`, `train_cnn`) | LR/SVM: C chosen on val AUROC. LightGBM: early stopping. MLP architecture. **CNN**: frequency encoder per frame → temporal conv → mean+max pool → linear. **Loss**: BCEWithLogits with positive-class weight (class balance). **Optimiser**: AdamW lr 1e-3, weight decay 1e-4. **LR schedule**: cosine annealing. Early stopping on validation AUROC; time-reversal augmentation; per-bin standardisation. |
| 8:00–9:00 | `src/models/train.py`, `src/eval/evaluate.py` (`metrics`, `main`) | Model × feature-group grid (pixel / spatial / temporal / all) = ablation; degradation-aware model. Metrics: accuracy, AUROC, F1, TPR/TNR; seen test, unseen generators (balanced with real), degraded, leave-one-generator-out, metadata-only shortcut check. |
| 9:00–9:30 | `src/predict.py`, `app.py`, `tests/` (run `pytest -q`) | Same preprocessing at inference; SHAP explanations; 40+ unit tests. |

## B. Demo video (target 4:45)

| Time | Action | Say |
|---|---|---|
| 0:00–0:20 | Webcam, app open | What the app does; runs locally on CPU. |
| 0:20–1:20 | Detect tab → pick a **real** example → Analyse | Verdict + P(AI); frames and their 2D spectra; spectrum vs real/AI training means; drift plot; top features. |
| 1:20–2:10 | Pick a **seen-generator** example (e.g. LTX) | Compare the spectrum curve and the drift plot to the real one. |
| 2:10–2:50 | Pick an **unseen** generator example (Wan / ModelScope) | Never seen in training — still detected? (say honestly if not). |
| 2:50–3:30 | Upload **your own phone clip** | Real camera, not stock footage. |
| 3:30–4:10 | Same clip with degradation `h264_crf38`, then switch model to `lgbm_all_robust` | Show how compression changes the probability and that the robust model holds up better (if it does). |
| 4:10–4:45 | Performance tab | Headline metrics table, per-generator table, ROC / ablation / robustness figures. |

Tip: record each segment separately and join them; if a prediction is wrong on camera, explain why
(e.g. the clip is outside the training domain) — that shows understanding.
