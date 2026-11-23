"""Rebuild the poster: regenerate every figure from the result files, then print poster/poster.html to
poster/poster.pdf (one page, A1 proportions) with headless Chrome or Edge.

    python -m tools.build_poster            # figures + PDF
    python -m tools.build_poster --no-pdf   # figures only

Figure sources: see the poster table in RESULTS.md. The workflow and architecture diagrams are drawn by
tools/poster_diagrams.py (the architecture one contains real curves of a held-out clip); the two logos come from the course template (poster/figs/logo_*.png).
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools import make_poster_figures as F
from tools import poster_diagrams as DG
from tools import poster_spectra as SP

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser"]


def find_browser():
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    return None


def main(pdf: bool) -> None:
    for f in (F.compare, F.resize, SP.spectra, F.temporal_order, F.prompts, F.fusion, DG.workflow, DG.architecture, F.qr):
        f()
    if not pdf:
        return
    browser = find_browser()
    if browser is None:
        sys.exit("no Chrome/Edge found: open poster/poster.html in a browser and print to PDF (A1, no margins)")
    html, out = ROOT / "poster" / "poster.html", ROOT / "poster" / "poster.pdf"
    tmp = out.with_name("poster_new.pdf")
    tmp.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as prof:      # a fresh profile avoids a stale cached copy
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=8000",
                        f"--user-data-dir={prof}", f"--print-to-pdf={tmp}", html.as_uri()],
                       check=True, capture_output=True)
    try:
        tmp.replace(out)
        print("wrote", out)
    except PermissionError:      # Windows: the old PDF is open in a viewer
        print(f"wrote {tmp}; {out.name} is open in another program: close it and rename {tmp.name} to {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    main(not ap.parse_args().no_pdf)
