"""Run the demo notebook (demo.ipynb: real and generated clips, spectra, drift, verdicts, pixel vs frequency vs fusion) from top to bottom, without opening Jupyter.

    python tools/run_demo_notebook.py

Writes demo_output/demo_executed.ipynb and demo_output/demo_executed.html; demo.ipynb itself is not changed.
Needs the packages of requirements-app.txt.
"""
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NAME = "demo"
KERNEL = "ie643"

# make sure the notebook's kernel exists for this Python
subprocess.run([sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", KERNEL,
                "--display-name", "IE643 demo"], check=True, capture_output=True)

nb = nbformat.read(ROOT / f"{NAME}.ipynb", as_version=4)
start = time.time()
try:
    ExecutePreprocessor(timeout=900, kernel_name=KERNEL).preprocess(nb, {"metadata": {"path": str(ROOT)}})
    status = "ok"
except Exception as e:                      # keep the partly executed notebook to see where it stopped
    status = f"FAILED\n{str(e)[-600:]}"

out = ROOT / "demo_output"
out.mkdir(exist_ok=True)
nbformat.write(nb, out / f"{NAME}_executed.ipynb")
html, _ = HTMLExporter().from_notebook_node(nb)
(out / f"{NAME}_executed.html").write_text(html, encoding="utf-8")
print(f"{NAME}: {status} in {time.time() - start:.0f} s -> {out / (NAME + '_executed.html')}")
sys.exit(0 if status == "ok" else 1)
