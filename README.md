# DRIFT — Domain-Residual Rank Allocation for Parameter-Efficient Adaptation of SLMs

Research code and paper for a conference submission on parameter-efficient
adaptation of small language models to domain-specific classification.

## The idea

Existing activation-geometry PEFT methods (EVA, CorDA, AIRA, TLoRA, RSLoRA)
decide where to put adapter capacity by profiling the **target** task's
activations in isolation. For *domain* adaptation that is the wrong question:
biomedical text is still English, so its dominant activation directions are the
directions the backbone was already optimised for, and rank spent there is
wasted.

DRIFT instead **contrasts two distributions**. It profiles the target corpus and
a general-domain reference corpus (WikiText-103), deflates the target
second-moment matrix by the reference's principal subspace,

```
Sigma_tilde = (I - P_G) Sigma_D (I - P_G)
```

and uses the spectrum of that *drift covariance* both to allocate a global
parameter budget across modules (greedy marginal analysis, provably optimal for
the continuous relaxation) and to initialise each adapter inside the residual
subspace.

Profiling is two forward passes: no labels, no gradients, no training.
`tau = 0` recovers single-distribution profiling (EVA) exactly, so any gain is
attributable to the contrast itself.

## Layout

```
src/
  common.py         seeding, metrics, parameter accounting
  data.py           ChemProt / RCT-20k / HoC loaders + reference corpus
  download_data.py  fetches every dataset (idempotent)
  drift.py          the method: covariance collection, subspace contrast, allocation
  peft_methods.py   LoRA / DoRA / PiSSA / AdaLoRA / BitFit adapters, all hand-rolled
  engine.py         training + evaluation loop shared by every method
  profile_drift.py  computes and caches a DRIFT profile for (model, task)
  run.py            one experiment
  grid.py           experiment plans (tune / main / ablation / budget / ladder)
  analyze.py        result JSONs -> LaTeX tables + paired significance tests
  figures.py        result JSONs + profiles -> paper figures
  make_kaggle.py    packages everything into a self-contained notebook
data/               datasets (downloaded)
runs/results/*.json one file per experiment
runs/profiles/      cached drift profiles (.pt) + timing summaries (.json)
paper/              main.tex, refs.bib, generated tables, figures, main.pdf
kaggle/             self-contained notebook for running the grid on a free GPU
tools/tectonic.exe  LaTeX engine (no system TeX install needed)
```

## Reproducing

Training runs on Kaggle or Google Colab, not locally. Import
`kaggle/drift_experiments.ipynb`:

- **Kaggle:** Accelerator = GPU T4 x2, Internet = On, then *Save & Run All
  (Commit)*. Runs are split across both GPUs, and no new run starts after
  `DEADLINE_HOURS` (10.5 h), so the session ends inside Kaggle's 12 h limit and
  saves `drift_results.zip`. To continue, upload that zip as a Kaggle Dataset,
  attach it with *Add Input*, and commit again.
- **Colab:** T4 GPU runtime, *Run all*. Results are written to Google Drive
  (`MyDrive/drift_results/`); after a disconnect just *Run all* again.

Every run writes `runs/results/<id>.json` and is skipped if that file already
exists, so all plans are resumable. `DRIFT_AMP=1` (set in the notebook) enables
fp16 autocast for every method alike.

The runs can also be launched and collected without the browser:
`python kaggle/kaggle_api.py push kaggle/drift_experiments.ipynb <slug> "<title>" [earlier-kernel-to-restore]`,
then `status <slug>` / `watch <slug>`. Outputs land in `kaggle/output/<slug>/`.
The API token is read from `kaggle API.txt` (keep that file out of anything you
share). `drift_ladder.ipynb`, `drift_smoke.ipynb` and `drift_stability.ipynb` are
small single-purpose notebooks built by the same script.

After changing anything in `src/`, rebuild and check the notebook:

```bash
python src/make_kaggle.py && python src/validate_notebook.py
```

Bring results back and rebuild tables, figures and the PDF:

```bash
python src/ingest_results.py path/to/drift_results.zip
```

## Building the paper

```bash
tools/tectonic.exe -X compile paper/main.tex --outdir paper
```

Numbers still to be filled from results appear in red as `[TODO: ...]`.

## Validation

The harness reproduces the published reference point: RoBERTa-base on ChemProt
reaches **82.4** test micro-F1 with LoRA at 1.06% trainable parameters, against
**81.9 ± 1.0** reported for full fine-tuning by Gururangan et al. (ACL 2020) on
the same split.
