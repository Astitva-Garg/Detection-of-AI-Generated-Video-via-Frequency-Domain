"""SHA-256 checksums of the core dataset files, so anyone can confirm they hold the same data.

    python -m src.data.checksums            # write data/SHA256SUMS.txt
    python -m src.data.checksums --verify   # check the files against data/SHA256SUMS.txt
"""
import argparse
import hashlib

from src.config import DATA_ROOT


def core_files():
    files = sorted((DATA_ROOT / "clips_crop").glob("*.mp4"))
    files += [DATA_ROOT / n for n in ("manifest_crop.csv", "manifest_degraded_crop.csv", "content_filter.csv",
                                      "audit.csv", "audit_duplicates.csv")]
    files += [DATA_ROOT / "raw" / "real" / "meta.csv"]
    files += sorted((DATA_ROOT / "raw" / "fake").glob("*/manifest.jsonl"))
    files += sorted((DATA_ROOT / "features_crop").glob("*.parquet"))
    files += sorted((DATA_ROOT / "features_crop").glob("seq_*.npz"))
    return [p for p in files if p.exists()]


def digest(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(verify: bool) -> None:
    sums = DATA_ROOT / "SHA256SUMS.txt"
    if not verify:
        files = core_files()
        with open(sums, "w", newline="\n") as out:
            for p in files:
                out.write(f"{digest(p)}  {p.relative_to(DATA_ROOT).as_posix()}\n")
        print(f"wrote {sums} ({len(files)} files)")
        return
    bad = 0
    lines = sums.read_text().splitlines()
    for line in lines:
        h, rel = line.split("  ", 1)
        p = DATA_ROOT / rel
        if not p.exists() or digest(p) != h:
            bad += 1
            print("MISMATCH" if p.exists() else "MISSING", rel)
    print(f"{len(lines) - bad} of {len(lines)} files match" + ("" if not bad else f", {bad} problems"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    main(ap.parse_args().verify)
