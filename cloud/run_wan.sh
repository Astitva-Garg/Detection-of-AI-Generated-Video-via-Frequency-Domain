#!/usr/bin/env bash
# Generate the Wan2.1 (unseen) clips on a rented Linux GPU (24 GB+ recommended, e.g. RTX 4090 / L40S / A100).
# Same prompts, seeds, resolution (832x480), frames (33), fps (16) and steps (30) as the local run, so the
# clips are interchangeable with the local ones.
#
# Usage (inside the unzipped project folder on the cloud machine):
#   bash cloud/run_wan.sh            # 100 clips (default)
#   N=60 bash cloud/run_wan.sh       # fewer clips
# Output: wan_clips.tgz in the project folder -> download it, then on the laptop:
#   tar xzf wan_clips.tgz -C C:/ie643_data/raw/fake      (replaces/extends raw/fake/wan)
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-100}"
export IE643_DATA="${IE643_DATA:-$PWD/ie643_data}"
export IE643_TE_GPU_MEM="${IE643_TE_GPU_MEM:-20GiB}"   # big GPU: keep the text encoder fully on GPU
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"
pip install -q "diffusers>=0.35" "transformers>=4.50" accelerate sentencepiece ftfy protobuf \
    imageio imageio-ffmpeg av opencv-python-headless pandas pyarrow pyyaml requests hf_transfer

mkdir -p "$IE643_DATA/raw/fake"
# run in the background-safe way: survives a closed browser tab if started with nohup (see README)
python -m src.generate.wan --n "$N" 2>&1 | tee "$IE643_DATA/raw/fake/wan_cloud.log"

ok=$(grep -c '"status": "ok"' "$IE643_DATA/raw/fake/wan/manifest.jsonl" || true)
echo "generated OK: $ok clips"
tar czf wan_clips.tgz -C "$IE643_DATA/raw/fake" wan
ls -lh wan_clips.tgz
