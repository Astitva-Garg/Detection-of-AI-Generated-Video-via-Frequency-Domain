#!/usr/bin/env bash
# Reproduce the results, one stage at a time. Every stage is resumable: it skips work already done.
#
#   bash run_pipeline.sh list             show every stage (safe: runs nothing)
#   bash run_pipeline.sh                  the stages that start from the normalised clips (needs the data bundle)
#   bash run_pipeline.sh all              the same, with the raw-clip stages `filter` and `dataset` first and `ablation` last
#   bash run_pipeline.sh train evaluate   only the stages you name, in the order you give
#
# `real` (download from Pexels) and `generate` (about 18 GPU-hours) are slow and are never part of the above;
# name them explicitly:   bash run_pipeline.sh real generate
#
# Environment: PY = python to use (default: the C:/ie643_env venv if it exists, else `python`), WORKERS = 3.
# Full output of each stage goes to data/logs/pipeline/<stage>.log. Nothing here deletes any file.
set -uo pipefail
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  if [ -x /c/ie643_env/Scripts/python.exe ]; then PY=/c/ie643_env/Scripts/python.exe; else PY=python; fi
fi
WORKERS="${WORKERS:-3}"
LOG=data/logs/pipeline
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

# stage | what it does | what it needs
STAGES="real|download the real clips from Pexels (selection by scene keywords)|internet
generate|generate the fake clips with the five generators (LTX, AnimateDiff, SVD, ModelScope, Wan2.1)|GPU, 18 h
filter|object filter (YOLOv8) over the raw clips -> data/content_filter.csv|data/raw
dataset|normalise (14 frames, 256 crop, H.264), splits, manifest, degraded copies|data/raw
features|frequency features (263) and the CNN input -> data/features_crop|data/clips_crop
pixel|pixel baseline features (74)|data/clips_crop
audit|dataset audit: repeated frames, near-duplicates, borders -> data/audit*.csv|data/clips_crop
train|train 22 models (LR, SVM, LightGBM, MLP x 5 feature groups, degradation-aware LightGBM, CNN)|features
evaluate|held-out test, per generator, degraded, leave-one-generator-out, metadata baseline|train
cv|grouped 5-fold cross-validation|features
bootstrap|bootstrap 95 % intervals and paired differences|evaluate
attribution|6-class generator attribution|features
wild|download the 250 public clips (only if not all present)|internet
wild_eval|score the public clips through the deployment path, cache their features|wild, train
deep|pixel CNN from scratch and fine-tuned ResNet-18, 5 seeds|GPU, wild_eval
seeds|retrain the main models with 5 seeds, mean +/- std|wild_eval, deep
figures|all figures -> results_crop/figures|evaluate, seeds
reference|app reference spectra and example clips|train
ablation|resize versus native crop on identical clips: rebuild the resize variant, cross-validate, compare|data/raw, features
demo|live demo: re-extract features of the held-out clips, compare pixel / frequency / fusion -> results_crop/live_comparison.csv|features, train
generate_fresh|ModelScope on 36 new prompts x 2 seeds for the prompt-disjoint test (about 30 GPU-minutes)|GPU
poster|rebuild every poster figure from the result files and print poster/poster.pdf|demo, extras, ablation results
extras|frame-order test, prompt-disjoint test, frequency + ResNet-18 late fusion|train, deep (+ GPU for the last one)"

BUNDLE="features pixel audit train evaluate cv bootstrap attribution wild wild_eval deep seeds extras figures reference demo poster"
RAW_STAGES="filter dataset"
RAW_LAST="ablation"

run() {   # run <stage> <command...>
  local name=$1; shift
  echo; echo "=== $(date '+%H:%M:%S') $name"
  mkdir -p "$LOG"
  if "$@" > "$LOG/$name.log" 2>&1; then tail -n 6 "$LOG/$name.log"
  else echo "FAILED: $name (full log: $LOG/$name.log)"; tail -n 15 "$LOG/$name.log"; exit 1; fi
}

stage() {
  case "$1" in
    real)        run real        "$PY" -m src.data.pexels --n 560 --cap 40 ;;
    generate)    run generate    "$PY" -m src.generate.run_queue ltx=160 animatediff=160 svd=140 modelscope=120 wan=70 ;;
    filter)      run filter      "$PY" -m src.data.content_filter ;;
    dataset)     run dataset     "$PY" -m src.data.build_dataset --workers "$WORKERS" ;;
    features)    run features    "$PY" -m src.features.extract --workers "$WORKERS" ;;
    pixel)       run pixel       "$PY" -m src.features.pixel_baseline --workers "$WORKERS" ;;
    audit)       run audit       "$PY" -m src.data.audit --workers "$WORKERS" ;;
    train)       run train       "$PY" -m src.models.train ;;
    evaluate)    run evaluate    "$PY" -m src.eval.evaluate ;;
    cv)          run cv          "$PY" -m src.eval.cv ;;
    bootstrap)   run bootstrap   "$PY" -m src.eval.bootstrap ;;
    attribution) run attribution "$PY" -m src.eval.attribution ;;
    wild)
      n=$(find data/wild -name '*.mp4' 2>/dev/null | wc -l)
      if [ "$n" -ge 250 ]; then echo; echo "=== wild: all $n clips present, skipping the download"
      else run wild "$PY" -m src.data.wild; fi ;;
    wild_eval)   run wild_eval   "$PY" -m src.eval.wild_eval --workers "$WORKERS" ;;
    deep)
      if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        run deep "$PY" -m src.models.pixel_cnn
      else echo; echo "=== deep: no CUDA GPU found, skipping (the stored results in results_crop/ are kept)"; fi ;;
    seeds)       run seeds       "$PY" -m src.eval.seed_stability ;;
    figures)     run figures     "$PY" -m src.viz.figures ;;
    reference)   run reference   "$PY" -m src.viz.reference ;;
    demo)        run demo        "$PY" -X utf8 -m src.demo_live --workers "$WORKERS" ;;
    poster)      run poster      "$PY" -m tools.build_poster ;;
    generate_fresh) run generate_fresh "$PY" -m src.generate.modelscope --fresh ;;
    extras)
      run temporal_order "$PY" -m src.eval.temporal_order --workers "$WORKERS"
      run fresh_prompts  "$PY" -m src.eval.fresh_prompts --workers "$WORKERS"
      if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        run late_fusion "$PY" -m src.eval.late_fusion
      else echo; echo "=== late_fusion: no CUDA GPU found, skipping (the stored results in results_crop/ are kept)"; fi ;;
    ablation)
      mkdir -p results
      run ablation_dataset  env IE643_NORM=resize "$PY" -m src.data.build_dataset --workers "$WORKERS" --skip-degraded
      run ablation_features env IE643_NORM=resize "$PY" -m src.features.extract --workers "$WORKERS"
      run ablation_pixel    env IE643_NORM=resize "$PY" -m src.features.pixel_baseline --workers "$WORKERS"
      run ablation_cv       env IE643_NORM=resize "$PY" -m src.eval.cv
      run ablation_compare  "$PY" -m src.eval.resize_ablation ;;
    *) echo "unknown stage: $1 (try: bash run_pipeline.sh list)"; exit 2 ;;
  esac
}

case "${1:-}" in
  list) echo "$STAGES" | awk -F'|' '{printf "  %-12s %s   [needs: %s]\n", $1, $2, $3}'; exit 0 ;;
  all)  set -- $RAW_STAGES $BUNDLE $RAW_LAST ;;
  "")   set -- $BUNDLE ;;
esac
for s in "$@"; do stage "$s"; done
echo; echo "=== done"
