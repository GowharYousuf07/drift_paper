"""Ingest a drift_results.zip produced on Kaggle/Colab and rebuild the paper assets.

    python src/ingest_results.py path/to/drift_results.zip [--model roberta-base]

Extracts the result JSONs into runs/, then regenerates every table and figure and
reports what is present so far and what is still missing.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNS = os.path.join(ROOT, "runs")


def extract(zip_path, overwrite=False):
    """Copy result and profile-summary JSONs from a kernel's zip into runs/.

    Existing files are never overwritten unless asked: a kernel's zip also holds
    every result it restored at start-up, and for the revision notebooks those
    are slim copies (arguments and scores only) of the full local records."""
    n_before = len(glob.glob(os.path.join(RUNS, "results", "*.json")))
    new, kept = 0, 0
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for n in names:
            # entries are stored relative to runs/ (e.g. "results/xyz.json")
            dest = os.path.join(RUNS, n.replace("\\", "/"))
            if os.path.exists(dest) and not overwrite:
                kept += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(n) as src, open(dest, "wb") as out:
                out.write(src.read())
            new += 1
    n_after = len(glob.glob(os.path.join(RUNS, "results", "*.json")))
    print(f"{len(names)} json files in the zip: {new} written, {kept} already present "
          f"and kept; {n_after - n_before} new results, {n_after} total")


def coverage(model):
    """Report which cells of the experiment grid are populated, counting only
    runs the tables would use (analyze.is_default)."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import analyze
    have = defaultdict(set)
    for r in analyze.load_all():
        if analyze.is_default(r):
            have[(r["model"].replace("__", "/"), r["task"], r["method"])].add(r["seed"])

    methods = ["full", "linear", "bitfit", "lora", "dora", "pissa",
               "adalora", "eva", "eva_white", "drift"]
    tasks = ["chemprot", "rct20k", "hoc"]
    print(f"\nmain-table coverage for {model} (seeds present per cell):")
    header = "  " + "method".ljust(10) + "".join(t.ljust(12) for t in tasks)
    print(header)
    for m in methods:
        row = "  " + m.ljust(10)
        for t in tasks:
            s = sorted(x for x in have.get((model, t, m), set()) if x is not None)
            row += (",".join(map(str, s)) if s else "-").ljust(12)
        print(row)

    others = sorted({k[0] for k in have if k[0] != model})
    if others:
        print("\nother backbones present:", ", ".join(others))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", nargs="?", default=None)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--tasks", default="chemprot,rct20k,hoc")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing files (never needed for kernel zips)")
    a = ap.parse_args()

    if a.zip:
        if not os.path.exists(a.zip):
            raise SystemExit("no such file: " + a.zip)
        extract(a.zip, overwrite=a.overwrite)

    coverage(a.model)

    print("\nregenerating tables and figures ...")
    sys.stdout.flush()   # keep our output ordered ahead of the subprocesses
    for cmd in (
        [PY, os.path.join(ROOT, "src", "analyze.py"), "--model", a.model,
         "--tasks", a.tasks],
        [PY, os.path.join(ROOT, "src", "figures.py"), "--model", a.model,
         "--tasks", a.tasks],
    ):
        subprocess.run(cmd, check=False)

    tectonic = os.path.join(ROOT, "tools", "tectonic.exe")
    if os.path.exists(tectonic):
        print("\ncompiling paper ...")
        # --keep-logs: check_paper.py reads main.log, which would otherwise be stale
        subprocess.run([tectonic, "-X", "compile", "main.tex", "--outdir", ".",
                        "--keep-logs"],
                       cwd=os.path.join(ROOT, "paper"), check=False)
    subprocess.run([PY, os.path.join(ROOT, "src", "check_paper.py")], check=False)


if __name__ == "__main__":
    main()
