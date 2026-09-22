"""Assemble the code-and-results release that accompanies the paper.

    python src/make_release.py            # writes release/ and drift_release.zip

The release holds everything needed to rerun or re-analyse the study and nothing
personal: the source tree, the notebook that runs every experiment on a free
Kaggle/Colab GPU, every per-run result record (arguments, scores, per-epoch
telemetry and, for the newer runs, per-example test predictions), the profile
summaries and the divergence files. It leaves out the Kaggle API client, which
carries an account name, the API token file, the datasets (fetched by
src/download_data.py) and the multi-hundred-megabyte profile tensors (recomputed
by src/profile_drift.py in minutes).
"""
import glob
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "release")

SRC = ["common.py", "data.py", "download_data.py", "drift.py", "peft_methods.py",
       "engine.py", "profile_drift.py", "run.py", "runspec.py", "grid.py",
       "analyze.py", "figures.py", "verify_results.py", "paper_numbers.py",
       "divergence.py", "make_kaggle.py", "validate_notebook.py", "check_paper.py"]
NOTEBOOKS = ["drift_experiments.ipynb", "drift_revision.ipynb"]
FORBIDDEN = ("kaggle API.txt", "Kaggle_API_2.txt", "kaggle_accounts.json",
             "kaggle_api.py", "gowhar", "@gmail", "KGAT_")

README = """# Where Should the Adapter Go? — code and results

Code, experiment plans and every result record behind the paper *Where Should the
Adapter Go? Outlier Dimensions, Rank Allocation and Placement in Low-Rank
Adaptation of Small Language Models*.

## Layout
```
src/                    all code (methods, training, profiling, planning, analysis)
  drift.py              activation profiling, reference contrast, GEV, allocation
  peft_methods.py       LoRA / DoRA / PiSSA / AdaLoRA adapters, one codebase
  engine.py             training and evaluation shared by every method
  run.py, runspec.py    one run and its result id
  grid.py               every experiment plan and the learning-rate protocol
  profile_drift.py      cached profiles (target and reference covariances)
  divergence.py         tau-free divergences between covariances
  analyze.py, figures.py, paper_numbers.py   tables, figures, quoted numbers
  verify_results.py     independent re-derivation of every table
  make_kaggle.py        builds the self-contained notebooks
notebooks/              notebooks that run the study on a free Kaggle/Colab GPU
runs/results/           one JSON per run: arguments, scores, telemetry, predictions
runs/profiles/          profile summaries (the tensors are recomputed on demand)
runs/divergence/        divergence files
```

## Environment
Python 3.11+, PyTorch 2.10, Transformers 5.0, pandas + pyarrow, scikit-learn,
scipy (all preinstalled on Kaggle and Colab). One NVIDIA T4 per run.

## Reproduce
1. `python src/download_data.py` fetches ChemProt, RCT-20k, HoC and the reference
   corpora.
2. Open `notebooks/drift_revision.ipynb` on Kaggle (GPU T4 x2, Internet on) or
   Colab and run all cells; finished runs are skipped, so a later session resumes.
   Single runs: `python src/run.py --model roberta-base --task chemprot --method drift --seed 1 --amp`.
3. `python src/analyze.py --model roberta-base --tasks chemprot,rct20k,hoc` and
   `python src/figures.py` rebuild the tables and figures from `runs/`;
   `python src/verify_results.py` re-derives them independently.

## Learning-rate protocol
Every configuration a conclusion is drawn from has its rate selected on its own
dev set (seed 1), and every grid is extended until the selection is interior;
see `grid.tuning_cells` and the paper's Appendix A.

## Licence
To be chosen by the authors before public release.
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "src"))
    for f in SRC:
        shutil.copy(os.path.join(ROOT, "src", f), os.path.join(OUT, "src", f))
    os.makedirs(os.path.join(OUT, "notebooks"))
    for nb in NOTEBOOKS:
        p = os.path.join(ROOT, "kaggle", nb)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, "notebooks", nb))
    for sub in ("results", "profiles", "divergence"):
        d = os.path.join(OUT, "runs", sub)
        os.makedirs(d)
        for p in glob.glob(os.path.join(ROOT, "runs", sub, "*.json")):
            shutil.copy(p, d)
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)

    # nothing personal may leave: scan every text file of the release
    leaks = []
    for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
        if os.path.isfile(p) and p.endswith((".py", ".md", ".ipynb", ".json")):
            with open(p, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            leaks += [(os.path.relpath(p, OUT), w) for w in FORBIDDEN if w in txt]
    if leaks:
        raise SystemExit(f"personal strings in the release: {leaks[:10]}")

    zpath = os.path.join(ROOT, "drift_release.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
            if os.path.isfile(p):
                z.write(p, os.path.relpath(p, OUT))
    n = sum(1 for _ in glob.glob(os.path.join(OUT, "runs", "results", "*.json")))
    print(f"release: {n} result records -> {OUT} and {zpath} "
          f"({os.path.getsize(zpath)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
