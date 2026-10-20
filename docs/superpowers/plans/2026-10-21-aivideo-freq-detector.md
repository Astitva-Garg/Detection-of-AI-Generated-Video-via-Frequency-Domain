# AI-Generated Video Frequency Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real-vs-diffusion video classifier from spatial frequency + temporal drift features, with a curated dataset, evaluation on seen/unseen/degraded data, interpretability figures and a Gradio test interface.

**Architecture:** Scripts in `src/` form a file-based pipeline: download/generate raw videos → normalise every clip identically → content-filter → manifest + splits + degraded copies → cached features (parquet + spectrum sequences npz) → models (sklearn/LightGBM/PyTorch) → evaluation JSON/CSV + figures → `app.py` loads saved models.

**Tech Stack:** Python 3.11 (uv venv), PyTorch CUDA, diffusers/transformers/accelerate, PyAV + imageio-ffmpeg, NumPy/SciPy, pandas/pyarrow, scikit-learn, LightGBM, SHAP, matplotlib, ultralytics (YOLOv8n), Gradio, pytest.

**Spec:** `docs/design.md`

## Global Constraints

- Data root `C:\ie643_data` (outside OneDrive); code root `C:\Users\astit\OneDrive\Desktop\ie643_aivideo`.
- Clip unit: 16 frames at 8 fps (2 s); luminance; resize short side 256, centre-crop 256×256.
- Radial spectrum: 48 bins.
- Seen generators: `ltx`, `animatediff`, `svd`. Unseen (test-only): `wan`, `modelscope`.
- Real and fake both go through the *same* normalisation and H.264 re-encode (CRF 18) before features.
- Real clips are first downscaled to a short side drawn from the generators' native short sides {256, 320, 480, 512, 576} and H.264 encoded, so real and fake share the resampling chain.
- Splits by source video / prompt group, 70/15/15, seed 643. Unseen generators → test only.
- Content filter: YOLOv8n COCO classes person, animals, vehicles, common indoor objects; reject if any box > 5 % of frame area on any of 4 sampled frames.
- Video files never go to git; `.gitignore` covers data, `.venv`, large checkpoints.
- No report prose generated; `docs/notes_for_report.md` holds facts/numbers/bullets only.
- 8 GB VRAM / 15 GB RAM: text encoders (T5/UMT5) run alone, embeddings cached, then freed.

## Review Focus

1. Uploaded clip shorter than 2 s or with < 16 frames → frames are looped/padded to 16, prediction still returned with a note. (test in Task 3)
2. Tiny (< 256 px) or portrait/odd-aspect uploads → upscaled/cropped to 256×256 without error. (test in Task 3)
3. Container with missing/variable fps metadata → falls back to frame count / pts timing, no crash. (test in Task 3)
4. Static or flat frames (constant colour, frozen video) → features finite (no NaN/inf from log 0 or divide by zero). (tests in Tasks 6 and 7)
5. Generator OOM or crash mid-run → runner logs failure, skips that item, resumes from manifest on restart without regenerating done clips. (test in Task 4)

---

### Task 1: Scaffold, environment, config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/__init__.py`, `src/config.py`, `tests/test_config.py`, `README.md` (stub)

**Interfaces:**
- Produces: `src.config` constants `DATA_ROOT: Path`, `RAW_REAL`, `RAW_FAKE`, `CLIPS`, `DEGRADED`, `FEATURES`, `MODELS`, `RESULTS` (Paths), `N_FRAMES=16`, `FPS=8`, `SIZE=256`, `N_BINS=48`, `SEED=643`, `SEEN=["ltx","animatediff","svd"]`, `UNSEEN=["wan","modelscope"]`, `REAL_PRE_SIDES=[256,320,480,512,576]`, `ensure_dirs() -> None`.

- [ ] Step 1: `git init`; `uv venv --python 3.11 .venv`; install torch (CUDA 12.x wheel index) + the stack above + pytest.
- [ ] Step 2: test `test_constants`: `N_FRAMES==16 and FPS==8 and N_BINS==48 and set(SEEN).isdisjoint(UNSEEN)`; `ensure_dirs()` creates all dirs under a tmp `DATA_ROOT` (monkeypatched).
- [ ] Step 3: implement `src/config.py`; run `pytest tests/test_config.py` → PASS; `python -c "import torch;print(torch.cuda.is_available())"` → True.
- [ ] Step 4: commit.

### Task 2: Real clips from Pexels-400k

**Files:**
- Create: `configs/scenes.yaml`, `src/data/pexels.py`, `tests/test_pexels.py`

**Interfaces:**
- Consumes: config paths.
- Produces: `select_titles(df: pd.DataFrame, include: list[str], exclude: list[str], min_dur=5, max_dur=60) -> pd.DataFrame` (adds `scene` column = first matching include keyword); CLI `python -m src.data.pexels --n 500` downloads to `RAW_REAL/<pexels_id>.mp4` and appends rows to `RAW_REAL/meta.csv` (`pexels_id,title,scene,url,duration`). Resumable (skips existing files), ≤ 2 requests/s, User-Agent header.

`configs/scenes.yaml` holds `include` (ocean waves, waterfall, river, clouds, sky, sunset, aurora, fog, mist, forest, trees, leaves, grass, field, mountain, desert, sand dunes, snow, rain, ice, lava, fire, flames, smoke, ink, abstract, bokeh, stars, night sky, lake, stream, water surface, timelapse landscape, galaxy …) and `exclude` (person, people, man, woman, girl, boy, child, hand, face, couple, dog, cat, bird, horse, animal, fish, insect, car, boat, ship, train, plane, drone shot of city, building, house, room, food, phone, laptop, window, street, road, bike …). Also used by Task 4 prompts: each scene key maps to 3–5 T2V prompts.

- [ ] Step 1: tests: `select_titles` keeps "Ocean Waves Crashing · Free Stock Video", drops "Man Walking on Beach at Sunset", drops duration 3, sets `scene=="ocean waves"`; max 60 per scene enforced via `cap_per_scene(df, cap)`.
- [ ] Step 2: run → FAIL; implement; run → PASS.
- [ ] Step 3: run download for `--n 500` in background (log to `RAW_REAL/download.log`).
- [ ] Step 4: commit.

### Task 3: Video I/O and normalisation

**Files:**
- Create: `src/data/video_io.py`, `src/data/normalize.py`, `tests/test_video_io.py`

**Interfaces:**
- Produces:
  - `read_frames(path, n_frames=16, fps=8, start=0.0) -> np.ndarray` uint8 `(n_frames,H,W,3)`; samples at times `start + k/fps`, nearest decoded frame by pts; if the video is shorter, loops frames; fps missing → derive from pts or assume 24.
  - `write_video(path, frames: np.ndarray, fps: int, crf: int = 18) -> None` (PyAV libx264, yuv420p, even dims).
  - `resize_crop(frames, size=256) -> np.ndarray` (short side → size with area interpolation for downscale, cubic for upscale, centre crop).
  - `normalize_clip(src, dst, pre_short_side: int | None = None, start=0.0) -> dict` — read 16 frames @8 fps, optionally downscale to `pre_short_side` and round-trip through H.264 CRF 18, then `resize_crop` to 256, write H.264 CRF 18 at 8 fps; returns `{"orig_w","orig_h","orig_fps","duration","n_src_frames"}`.

- [ ] Step 1: tests with synthetic videos written by `write_video` in tmp: 40-frame 24 fps 640×360 clip → `read_frames` shape `(16,360,640,3)`; 5-frame clip → still 16 frames (Review Focus 1); 100×60 portrait 60×100 clip → `normalize_clip` output decodes to `(16,256,256,3)` (Review Focus 2); file with fps metadata stripped/variable → no exception (Review Focus 3).
- [ ] Step 2: run → FAIL; implement; run → PASS.
- [ ] Step 3: commit.

### Task 4: Prompt list and generators

**Files:**
- Create: `configs/prompts.yaml` (generated from scenes.yaml + hand-written variations), `src/generate/common.py`, `src/generate/text_embed.py`, `src/generate/ltx.py`, `src/generate/animatediff.py`, `src/generate/svd.py`, `src/generate/wan.py`, `src/generate/modelscope.py`, `src/generate/run_queue.py`, `tests/test_generate_common.py`

**Interfaces:**
- Produces:
  - `common.Job(id: str, generator: str, prompt: str, scene: str, seed: int, image: str | None)`.
  - `common.run_jobs(jobs: list[Job], gen_fn: Callable[[Job], tuple[np.ndarray, int]], out_dir: Path) -> None` — skips ids already in `out_dir/manifest.jsonl`; writes `out_dir/<id>.mp4` (native fps, CRF 12) and appends `{id,generator,prompt,scene,seed,image,fps,n_frames,h,w,seconds,status}`; on exception (incl. `torch.cuda.OutOfMemoryError`) logs `status:"failed"` with error, empties CUDA cache and continues; failed ids are retried once on the next run.
  - Each generator module: `make_jobs(n: int) -> list[Job]` and `main(n)`; CLI `python -m src.generate.<name> --n N`.
  - `text_embed.cache_t5(model_id, prompts, out_path)` — loads only the text encoder (bf16, on GPU with CPU offload as needed), saves embeddings + masks, frees memory. Used by `ltx` and `wan`.
  - `run_queue.py` runs generators in order ltx → animatediff → svd → modelscope → wan with target counts from CLI, so one background process covers all.
- Settings (fit to 8 GB, adjust if OOM, record in manifest): LTX-Video `Lightricks/LTX-Video` 704×480, 49 frames, 24 fps, ~30 steps; AnimateDiff motion adapter v1-5-2 on a realistic SD1.5 base, 512×512 (or 512×384), 16 frames, 8 fps, 25 steps; SVD `stabilityai/stable-video-diffusion-img2vid-xt` 1024×576 (or 768×432 if OOM), 25 frames, fps=7, decode_chunk_size 2, conditioned on frame 0 of real train-pool clips; ModelScope `ali-vilab/text-to-video-ms-1.7b` 256×256, 16 frames, 8 fps; Wan2.1 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 832×480, 33 frames, 16 fps. All `enable_model_cpu_offload()` + VAE tiling/slicing. Negative prompt for all T2V: "people, person, human, animal, text, watermark, car, building".
- SVD conditioning images: frame 0 of real clips; `svd.make_jobs` sets `Job.scene` and uses group id `real_<pexels_id>` so in Task 6 the SVD fake gets the *same* `group_id`, hence the same split, as its source clip. `assign_split` (Task 6) is pure hashing, so no ordering dependency.
- T2V group ids: `prompt_<index>` (all seeds of one prompt share a split).

- [ ] Step 1: tests for `run_jobs` with a fake `gen_fn` returning random frames: 3 jobs → 3 mp4 + 3 manifest lines; second call regenerates nothing; a `gen_fn` raising `RuntimeError` for job 2 → status failed, jobs 1 and 3 written, rerun retries job 2 (Review Focus 5).
- [ ] Step 2: run → FAIL; implement `common.py`; run → PASS.
- [ ] Step 3: implement each generator; smoke test `--n 1` for each; record seconds/clip.
- [ ] Step 4: set targets from measured speed (design §5.1 table), launch `run_queue.py` in background, logging to `RAW_FAKE/queue.log`.
- [ ] Step 5: commit.

### Task 5: Content filter

**Files:**
- Create: `src/data/content_filter.py`, `tests/test_content_filter.py`

**Interfaces:**
- Produces: `BLOCKED_CLASSES: set[str]`; `max_blocked_area(frames: np.ndarray, model) -> tuple[float, str | None]` (largest blocked-box area fraction and its class, over 4 evenly spaced frames); `is_allowed(frac: float, thresh=0.05) -> bool`; CLI writes `DATA_ROOT/content_filter.csv` (`path,frac,cls,allowed`) for all raw real + fake clips, cached per path.

- [ ] Step 1: tests: `is_allowed(0.04)` True, `is_allowed(0.06)` False; `BLOCKED_CLASSES` ⊇ {"person","dog","bird","car","boat","chair","tv"} (all 80 COCO classes are blocked — none are scenery); `max_blocked_area` on blank frames → `(0.0, None)`.
- [ ] Step 2: run → FAIL; implement; run → PASS. Commit.

### Task 6: Dataset build — normalise, manifest, splits, degraded set

**Files:**
- Create: `src/data/splits.py`, `src/data/degrade.py`, `src/data/build_dataset.py`, `tests/test_splits.py`, `tests/test_degrade.py`

**Interfaces:**
- Produces:
  - `assign_split(group_id: str, seed=643) -> str` — deterministic hash → "train" (70 %), "val" (15 %), "test" (15 %).
  - `DEGRADATIONS: dict[str, Callable[[np.ndarray], np.ndarray] | tuple]` with keys `h264_crf23`, `h264_crf30`, `h264_crf38`, `down0.5`, `blur1.0`, `noise5` (h264 ones implemented by encode/decode round trip); `apply_degradation(frames, name) -> np.ndarray`.
  - `build_dataset.py` CLI: normalises every allowed raw clip to `CLIPS/<clip_id>.mp4` (real: `pre_short_side` = `REAL_PRE_SIDES[hash % 5]`; real clips ≥ 6 s also yield a second window at start=3 s marked `aug=True`, train only), writes `DATA_ROOT/manifest.csv` with columns `clip_id,label(0 real/1 fake),generator("real"|name),group_id,split,scene,prompt,source,orig_w,orig_h,orig_fps,duration,content_frac,aug`; unseen generators forced to `split="test"`; writes degraded copies of every test clip to `DEGRADED/<name>/<clip_id>.mp4` and `DATA_ROOT/manifest_degraded.csv`. Incremental: re-running only processes new raw clips.

- [ ] Step 1: tests: `assign_split` deterministic and ≈70/15/15 over 10 000 ids (±2 %); every degradation returns same shape uint8; `h264_crf38` changes frames (mean abs diff > 0) but `blur1.0` of a constant frame is unchanged; `noise5` on constant frame stays finite (Review Focus 4 groundwork).
- [ ] Step 2: run → FAIL; implement; run → PASS. Commit.

### Task 7: Spatial frequency features

**Files:**
- Create: `src/features/spatial.py`, `tests/test_spatial.py`

**Interfaces:**
- Produces:
  - `to_luma(frame_rgb_uint8) -> np.ndarray float32 [0,1]` (BT.601).
  - `radial_spectrum(gray, n_bins=48) -> np.ndarray (48,)` — Hann window, `log(|FFT|²+1e-8)` azimuthally averaged over equal-width radius bins up to Nyquist, minus its mean (shape not brightness).
  - `residual(gray) -> np.ndarray` — gray minus 3×3 median (high-pass noise residual).
  - `frame_features(gray) -> dict[str, float]` keys: `rad_00..rad_47`, `res_rad_00..res_rad_47` (radial spectrum of residual), `slope` (linear fit of log power vs log radius over bins 2–47), `hf_ratio_25/50/75` (power above that fraction of Nyquist / total, linear power), `centroid`, `rolloff85`, `dct_low/mid/high` (8×8 block DCT energy fractions, zigzag index <6, 6–27, ≥28), `ang_0..ang_7` (energy fraction in 8 orientation wedges, radius > 25 % Nyquist).
- [ ] Step 1: tests: white noise 256×256 → `slope` in [-0.3, 0.3]; synthetic 1/f² power image → `slope` in [-2.4, -1.6]; low-pass blurred noise has lower `hf_ratio_50` than raw noise; constant image → all values finite (Review Focus 4); `len(radial_spectrum(x))==48`.
- [ ] Step 2: run → FAIL; implement; run → PASS. Commit.

### Task 8: Temporal drift features and extraction

**Files:**
- Create: `src/features/temporal.py`, `src/features/extract.py`, `tests/test_temporal.py`

**Interfaces:**
- Consumes: `spatial.frame_features`, `spatial.radial_spectrum`, `video_io.read_frames`.
- Produces:
  - `drift_features(spectra: np.ndarray (T,48), scalars: dict[str, np.ndarray (T,)]) -> dict[str, float]` — for spectra: consecutive L2 distance and cosine distance series → `mean,std,max,ac1` (lag-1 autocorr; 0 if std==0); second-order (`|d_t - d_{t-1}|`) mean/std; per-band drift for bins 0–15/16–31/32–47 (mean, std). For each scalar in {`slope`,`hf_ratio_50`,`centroid`,`dct_high`}: `|Δ|` mean/std/max and second-order mean. Prefix `drift_`.
  - `residual_temporal(frames_gray: np.ndarray (T,H,W)) -> dict[str,float]` — per-pixel residual time series; mean over pixels of lag-1 temporal correlation, and FFT-along-time energy fraction in upper half of temporal frequencies. Prefix `rt_`.
  - `clip_features(frames_rgb) -> tuple[dict[str,float], np.ndarray]` — spatial per-frame features aggregated as `sp_<name>_mean` / `sp_<name>_std`, plus drift + residual-temporal; second output `seq` float32 `(2,16,48)`: channel 0 = per-frame radial spectra, channel 1 = radial spectrum of frame differences (frame 0 diff = zeros).
  - `FEATURE_GROUPS: dict[str, list[str] prefix]` = `{"spatial": ["sp_"], "temporal": ["drift_","rt_"]}`.
  - `extract.py` CLI: for `manifest.csv` and `manifest_degraded.csv`, writes `FEATURES/features.parquet` / `features_degraded.parquet` (clip_id + feature columns, plus `degradation` col) and `FEATURES/seq.npz` / `seq_degraded.npz` (`ids`, `seq`); multiprocessing over CPU cores; incremental.
- [ ] Step 1: tests: identical frames ×16 → all `drift_*` == 0 and finite, `ac1`==0 (Review Focus 4); frames with i.i.d. noise each frame → `rt_` lag-1 corr ≈ 0 (|x| < 0.1); same static noise frame repeated → corr ≈ 1 (> 0.9); `clip_features` on random 16×256×256×3 returns `seq.shape==(2,16,48)` and no NaN.
- [ ] Step 2: run → FAIL; implement; run → PASS. Commit.

### Task 9: Models

**Files:**
- Create: `src/models/data.py`, `src/models/classic.py`, `src/models/cnn.py`, `src/models/train.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `data.load_split(split: str, groups: list[str] | None = None, include_aug=False) -> tuple[pd.DataFrame X, np.ndarray y, pd.DataFrame meta]` (joins features + manifest; `groups` selects feature columns by `FEATURE_GROUPS`; excludes unseen generators unless split=="test").
  - `classic.make_model(name) -> sklearn Pipeline` for `lr` (StandardScaler + LogisticRegression C tuned on val ∈ {0.01,0.1,1,10}), `svm` (StandardScaler + LinearSVC wrapped in CalibratedClassifierCV), `lgbm` (LGBMClassifier, ~400 trees, lr 0.05, num_leaves 15, early stopping on val), `mlp` (torch MLP: in→128→64→1, ReLU, dropout 0.3, BCEWithLogits, AdamW lr 1e-3 wd 1e-4, cosine schedule, 200 epochs max, early stop patience 20 on val AUROC; wrapped with sklearn-style `fit/predict_proba`).
  - `cnn.SpecTempCNN(in_ch=2, n_bins=48)`: shared per-frame frequency encoder Conv1d over bins (2→16→32, k=5, stride 2 on the second) → flatten to a per-frame vector → Conv1d over time (k=3, 64 ch, two layers) → global average over time → dropout 0.3 → Linear(64→1). < 200k params. BCEWithLogits, AdamW lr 1e-3, wd 1e-4, cosine schedule, batch 32, 60 epochs, early stop (patience 10) on val AUROC; per-bin standardisation from train stats; extra real windows (`include_aug=True`) used in training.
  - `train.py` CLI: trains `lr, svm, lgbm, mlp` × feature groups {`spatial`, `temporal`, `all`} and `cnn`; saves `MODELS/<model>_<group>.joblib` (cnn: `.pt` + norm stats json) and `RESULTS/val_metrics.csv`. `--robust` flag additionally trains `lgbm_all_robust` on train + degraded copies of *train* clips (degraded train copies created on the fly via Task 6 functions + Task 8 extraction, cached as `features_train_degraded.parquet`).
- [ ] Step 1: tests on a synthetic separable feature table: each classic model reaches val AUROC > 0.9; `SpecTempCNN` forward on `(4,2,16,48)` → `(4,1)` and param count < 200 000; saved/loaded models give identical probabilities.
- [ ] Step 2: run → FAIL; implement; run → PASS.
- [ ] Step 3: run `python -m src.models.train` on real features once ≥ 50 % of data exists (early check); rerun when generation finishes. Commit.

### Task 10: Evaluation

**Files:**
- Create: `src/eval/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces: `metrics(y, p, thresh=0.5) -> dict` (`acc, auroc, f1, tpr_fake, tnr_real, n`); CLI writes:
  - `RESULTS/seen_test.csv` (model × group: metrics on seen-test),
  - `RESULTS/per_generator.csv` (detection rate per generator incl. unseen; unseen sets paired with an equal-size random sample of real test clips for AUROC),
  - `RESULTS/degraded.csv` (model × degradation),
  - `RESULTS/logo.csv` (lgbm_all & cnn: leave-one-seen-generator-out),
  - `RESULTS/metadata_baseline.csv` (LR on `orig_w,orig_h,orig_fps,duration` only — shortcut check),
  - `RESULTS/predictions.csv` (clip-level probs for error gallery),
  - `RESULTS/summary.json` (headline numbers for the app dashboard).
- [ ] Step 1: tests: perfect predictions → acc 1, auroc 1; constant p → auroc 0.5; single-class y → auroc reported as NaN not crash.
- [ ] Step 2: run → FAIL; implement; run → PASS. Run full eval. Commit.

### Task 11: Visualisations

**Files:**
- Create: `src/viz/figures.py`

**Interfaces:**
- Consumes: features, seq.npz, manifest, models, results CSVs.
- Produces (PNG, 200 dpi, in `RESULTS/figures/`): `mean_spectrum_2d.png` (mean centred log-power 2D spectrum per class/generator + difference to real), `radial_profiles.png` (mean ± std radial spectrum per generator), `drift_examples.png` (L2 spectral drift series + hf_ratio over frames for 3 real vs 3 fake clips), `drift_distributions.png` (violin of key drift features by generator), `feature_importance_groups.png` (LGBM gain + mean |SHAP| aggregated by group: radial low/mid/high, residual, DCT, angular, drift, residual-temporal), `importance_over_frequency.png` (|SHAP| per radial bin, spatial vs residual), `cnn_saliency.png` (mean |grad×input| frames × bins, real vs fake), `confusion_matrices.png`, `roc_curves.png` (seen + each unseen), `degradation_bars.png`, `ablation_bars.png`, `tsne.png`, `error_gallery.png` (top-8 most confident errors: thumbnail + radial spectrum vs class means). Also `plot_clip(clip_path, predictor) -> dict[str, matplotlib Figure]` reused by the app.
- [ ] Step 1: implement; run `python -m src.viz.figures`; open each PNG and check it is non-empty and labelled. Commit.

### Task 12: Predictor and Gradio app

**Files:**
- Create: `src/predict.py`, `app.py`, `examples/` (6 short normalised clips: 2 real, 1 per seen gen, 1 unseen), `tests/test_predict.py`

**Interfaces:**
- Produces: `Predictor(model_dir=MODELS)` with `.predict(video_path, model="lgbm_all", start=0.0, degradation: str | None = None) -> PredictionResult` dataclass (`p_fake: float, label: str, features: dict, seq: np.ndarray, frames: np.ndarray, top_features: list[tuple[str,float]], content_warning: str | None, notes: list[str]`). Uses identical preprocessing to training (`normalize` path in memory, real pre-downscale not applied — note in README).
- `app.py`: tab "Detect" (upload / example picker, model dropdown, start-time slider, degradation dropdown → verdict + gauge, frame strip, spectrum vs class means, drift plot, SHAP top-10 bar, warnings); tab "Performance" (tables from results CSVs + key figures). `python app.py` → http://127.0.0.1:7860.
- [ ] Step 1: tests: `predict` on an example real and fake clip returns 0 ≤ p ≤ 1 within 10 s on CPU; a 0.5 s clip returns a result with a note (Review Focus 1); a 100×60 clip works (Review Focus 2).
- [ ] Step 2: run → FAIL; implement; run → PASS.
- [ ] Step 3: launch app, click through both tabs with an example and an uploaded file in the browser pane. Commit.

### Task 13: Documentation and report notes

**Files:**
- Create/Modify: `README.md`, `docs/notes_for_report.md`, `requirements.txt` (`uv pip freeze`)

- [ ] Step 1: README: purpose, setup (uv commands), full pipeline commands in order, app usage, dataset rebuild from manifest, model files, hardware, licences/citations (Pexels, each generator, YOLOv8).
- [ ] Step 2: notes_for_report.md, sections mirroring the report template: work before vs after prep review; workflow bullets; dataset extraction steps + counts (downloaded, filtered, per generator, per split); preprocessing; feature list; model details (architecture, loss, optimiser, LR, schedule); experiments + hardware + timings; result tables (copied numbers); figure list with one-line factual captions; interface methodology; limitations; novelty plan; references in course format. Facts only — no ready-to-submit prose.
- [ ] Step 3: fresh clone check: `pytest` all pass; commit.
