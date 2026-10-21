# Running the Wan2.1 generation on a cloud GPU

Only the unseen-generator Wan2.1 clips remain to be generated (~4.8 min/clip on the laptop; roughly
0.5–1 min/clip on a 24 GB+ GPU, so 100 clips take about 1–2 hours).

## 1. Rent a GPU
RunPod or Vast.ai: pick an **RTX 4090 / L40S / A100**, template **"PyTorch 2.x"** (Ubuntu + CUDA),
container disk **≥ 60 GB** (model weights ~27 GB + clips). Open its **Web Terminal** or **Jupyter**.

## 2. Upload the code
Upload `cloud/wan_job.zip` (created on the laptop, ~100 KB, code only) through Jupyter's upload
button, then in the terminal:

```bash
cd /workspace && unzip -o wan_job.zip -d ie643 && cd ie643
nohup bash cloud/run_wan.sh > run.log 2>&1 &
tail -f run.log        # Ctrl+C stops watching, not the job
```

## 3. Bring the clips back
When `run.log` ends with `generated OK: …` and `wan_clips.tgz`, download `wan_clips.tgz`
(Jupyter: right-click → Download), put it in `C:\ie643_data\raw\fake\` on the laptop and run
(Git Bash):

```bash
cd /c/ie643_data/raw/fake && mv wan wan_local_backup && tar xzf wan_clips.tgz
```

Then tell Claude — it reruns filter → build → features → train → evaluate → figures.

**Stop the cloud machine afterwards** so it stops billing.
