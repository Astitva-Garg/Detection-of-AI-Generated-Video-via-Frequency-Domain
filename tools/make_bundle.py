"""Build dist/ie643_data_bundle.zip: everything needed to reproduce the results from the normalised clips.

    python tools/make_bundle.py

Unzip it in the project root (it contains the folder `data/`), then check it with
`python -m src.data.checksums --verify`. It leaves out the raw Pexels videos (19 GB: ids and links are in
data/raw/real/meta.csv), the 250 public clips (downloaded by `bash run_pipeline.sh wild`), the degraded clips
(their features are included) and the frame caches.
"""
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.config import DATA_ROOT  # noqa: E402
from src.data.checksums import core_files  # noqa: E402

extra = [DATA_ROOT / "README.md", DATA_ROOT / "SHA256SUMS.txt", DATA_ROOT / "raw" / "pexels_400k.parquet"]
extra += [p for p in (DATA_ROOT / "features_crop").glob("*") if p.suffix in (".parquet", ".npz")]
extra += [p for p in (DATA_ROOT / "fresh").rglob("*") if p.is_file()]   # prompt-disjoint test clips
files = sorted({p for p in core_files() + extra if p.exists()})
out = root / "dist" / "ie643_data_bundle.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, "data/" + p.relative_to(DATA_ROOT).as_posix())
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files)")
