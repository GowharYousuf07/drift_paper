# Where Should the Adapter Go? Outlier Dimensions, Rank Allocation and Placement in Low-Rank Adaptation of Small Language Models

Code, experiment plans, per-run results and paper for an evaluation study of
training-free, activation-guided rank allocation for LoRA on small language
models and biomedical classification.

## What the study finds

Activation-guided methods such as EVA decide where a low-rank adapter's
capacity goes by profiling the target task's activations before training. At
matched parameter budgets, with every configuration's learning rate tuned on its
own, none of eleven methods beats a tuned uniform LoRA once multiple comparisons
are corrected (1,328 runs over three benchmarks, a budget sweep, a 4M-110M
backbone ladder, a decoder SLM and a clinical task). The methods differ in
stability instead: on RoBERTa-base the leading directions of activation PCA lie
on the backbone's outlier dimensions 77 and 588, and EVA's initialisation is
fragile as a result. Whitening those directions, deflating a general-domain
reference subspace (DRIFT) or the exact generalised-eigenvector contrast largely
restore stability but add no accuracy we can detect, and a random-token
reference serves as well as real text. The paper closes with a protocol for
evaluating adapter allocation.

## DRIFT, the instrument

DRIFT **contrasts two distributions**. It profiles the target corpus and a
general-domain reference corpus (WikiText-103), deflates the target covariance
by the reference's principal subspace,

```
Sigma_tilde = (I - P_G) Sigma_D (I - P_G)
```

and uses the spectrum of that *drift covariance* both to allocate a global
parameter budget across modules (greedy marginal analysis) and to initialise
each adapter inside the residual subspace. Profiling is two forward passes: no
labels, no gradients, no training. `tau = 0` recovers single-distribution
profiling (EVA) exactly, which isolates the effect of the contrast. In the paper
DRIFT serves as an instrument for that comparison, not as a recommended method.

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
runs/profiles/      profile summaries (.json); the covariance tensors (.pt, 3.2 GB)
                    are recomputed by profile_drift.py and not in the repository
paper/              main.tex (the 10-page paper), supplement.tex (full analyses
                    and appendices), refs.bib, generated tables, figures, PDFs
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
python src/build_paper.py
```

This compiles `paper/main.tex` (the paper, ten pages including references) and
`paper/supplement.tex` (the full analyses and the appendices) in the order that
resolves their references to each other, then reports any unresolved reference.

`python src/analyze.py --model roberta-base --tasks chemprot,rct20k,hoc` rebuilds
every table from `runs/results/`. `python src/verify_results.py` re-derives the
tables and every result number quoted in the text from the same files, sharing
no code with `analyze.py` (its profile checks need the `.pt` tensors), and
`python src/number_coverage.py` lists any quoted number no check covers.

## Validation

The harness reproduces both published reference points on these splits: full
fine-tuning of RoBERTa-base reaches **81.7 ± 0.4** test micro-F1 on ChemProt,
against **81.9 ± 1.0** reported by Gururangan et al. (ACL 2020), and **80.0 ± 2.4**
micro-F1 on HoC, against **79.7** reported for RoBERTa-base on the BLURB split.

## License

Code in this repository is released under the [MIT License](LICENSE).

The paper text and figures under `paper/` are not covered by the MIT License;
all rights to those are reserved pending publication.
