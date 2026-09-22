"""Aggregate result JSONs into the paper's LaTeX tables and the numbers quoted in
its text.

Every number is taken at the learning rate selected for its own configuration
(grid.resolve_lr): each method, adapter placement, budget, deflation level and
backbone has its own dev-set selection, and ablation variants inherit their
parent's. The test set never chooses anything.

    python src/analyze.py --model roberta-base --tasks chemprot,rct20k,hoc
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "runs", "results")
OUT = os.path.join(ROOT, "paper")

PRETTY = {
    "full": "Full fine-tuning", "linear": "Linear probe", "bitfit": "BitFit",
    "lora": "LoRA", "dora": "DoRA", "pissa": "PiSSA", "adalora": "AdaLoRA",
    "eva": "EVA (budget-matched)", "eva_white": "EVA (whitened)", "drift": r"\method{}",
    "drift_abs": r"\method{}-abs", "gev": "GEV",
}
TASK_PRETTY = {"chemprot": "ChemProt", "rct20k": "RCT-20k", "hoc": "HoC",
               "mtsamples": "MTSamples"}
TASK_METRIC = {"chemprot": "micro_f1", "rct20k": "micro_f1", "hoc": "example_f1"}
MODEL_PRETTY = {
    "google__bert_uncased_L-2_H-128_A-2": "BERT-Tiny",
    "google__bert_uncased_L-4_H-256_A-4": "BERT-Mini",
    "google__bert_uncased_L-4_H-512_A-8": "BERT-Small",
    "google__bert_uncased_L-8_H-512_A-8": "BERT-Medium",
    "google__bert_uncased_L-12_H-768_A-12": "BERT-Base",
    "roberta-base": "RoBERTa-base",
    "distilroberta-base": "DistilRoBERTa",
    "HuggingFaceTB__SmolLM2-360M": "SmolLM2-360M",
}
MODEL_PARAMS = {
    "BERT-Tiny": 4.4, "BERT-Mini": 11.2, "BERT-Small": 28.8,
    "BERT-Medium": 41.4, "BERT-Base": 110.1, "RoBERTa-base": 125.0,
    "DistilRoBERTa": 82.1,
}
FIVE = ("lora", "eva", "eva_white", "drift")      # methods with seeds 4-5
SEED_TAGS = ("", "seeds45")
FAIL_MARGIN = 0.10                                # 10 F1 points


def load_all(tag_filter=None, exclude_tags=("smoke", "tune")):
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        a = r.get("args", {})
        tag = a.get("tag", "") or ""
        if exclude_tags and tag in exclude_tags and tag_filter != tag:
            continue
        if tag_filter is not None and tag != tag_filter:
            continue
        metric = TASK_METRIC.get(a.get("task"), r.get("metric", "micro_f1"))
        res = r["result"]
        rows.append({
            "model": a.get("model", "").replace("/", "__"),
            "task": a.get("task"), "method": a.get("method"),
            "seed": a.get("seed"), "lr": a.get("lr"),
            "budget_rank": a.get("budget_rank"), "tau": a.get("tau"),
            "score_mode": a.get("score_mode"), "init_mode": a.get("init_mode"),
            "alloc_mode": a.get("alloc_mode"), "target": a.get("target"),
            "rho": a.get("rho"), "cov": a.get("cov"), "scale": a.get("scale"),
            "max_train": a.get("max_train"), "tag": tag,
            # fields added in the revision; older runs used the defaults
            "ref": a.get("ref", "wikitext"), "det": bool(a.get("deterministic", False)),
            "amp": bool(a.get("amp", False)),
            "n_ref": a.get("n_ref", 1024), "n_dom": a.get("n_dom", 1024),
            "head": a.get("head", "default"), "scaling": a.get("scaling", "alpha_r"),
            "batch_size": a.get("batch_size"),
            # per-example predictions are read on demand (load_preds): thousands
            # per run, too many to hold for every run at once
            "has_preds": "test_preds" in res, "path": p,
            "score": res["test"].get(metric),
            "dev": res["dev_best"].get(metric),
            "micro": res["test"].get("micro_f1"),
            "macro_f1": res["test"].get("macro_f1"),
            "best_epoch": res.get("best_epoch"),
            # AdaLoRA reports its post-pruning budget; every other method keeps
            # exactly what it allocated.
            "adapter_params": res.get("params_adapter_effective", res.get("params_adapter")),
            "adapter_params_peak": res.get("params_adapter"),
            "head_params": res.get("params_head"),
            "trainable": res.get("params_trainable"),
            "train_time_s": res.get("train_time_s"),
            "peak_mem": res.get("peak_mem_bytes"),
            "history": res.get("history"),
            "n_train": r.get("n_train"),
            "ranks": r.get("ranks", {}),
            "rank_hist": r.get("rank_hist", {}),
            "id": r.get("id"),
        })
    return rows


PROFILED = {"eva", "eva_white", "drift", "drift_abs", "drift_nodeflate", "gev"}
# The configuration every method runs with unless a sweep varies one field.
DEFAULT_CFG = {"tag": "", "target": "all", "budget_rank": 8, "max_train": None,
               "ref": "wikitext", "det": False, "n_ref": 1024, "n_dom": 1024,
               "head": "default", "scaling": "alpha_r"}
DEFAULT_PROFILED = {"tau": 0.95, "score_mode": "relative", "init_mode": "drift",
                    "alloc_mode": "drift", "rho": 2.0,
                    # runs from before the fix to match EVA's reference
                    # implementation lack these fields and are excluded
                    "cov": "centered", "scale": "adjusted"}


def is_default(r, free=()):
    """True if `r` is its method's canonical configuration, ignoring the fields
    named in `free`. Without this, sweep runs that carry no tag (e.g. DRIFT at
    tau=0.5) would be averaged into the headline numbers."""
    want = dict(DEFAULT_CFG)
    if r["method"] in PROFILED:
        want.update(DEFAULT_PROFILED)
        if r["method"] == "gev":
            want.update(init_mode="gev", alloc_mode="uniform")
    return all(_eq(r.get(k), v) for k, v in want.items() if k not in free)


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-9
    return a == b


def _same_lr(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def hf_name(model):
    return model.replace("__", "/")


def lr_for(model, task, method, target="all", budget_rank=8, tau=None,
           scaling="alpha_r", head="default"):
    """The rate selected for this configuration (see grid.resolve_lr)."""
    import grid
    return grid.resolve_lr(hf_name(model), task, method, target=target,
                           budget_rank=budget_rank, tau=tau, scaling=scaling, head=head)


def pick(rows, model, task, method, lr="tuned", tags=("",), **want):
    """Runs of one configuration at one learning rate. `want` fixes fields that
    differ from the method's default (target, budget_rank, tau, ref, ...); by
    default the rate is the one selected for the configuration."""
    if lr == "tuned":
        lr = lr_for(model, task, method, target=want.get("target", "all"),
                    budget_rank=want.get("budget_rank", 8), tau=want.get("tau"),
                    scaling=want.get("scaling", "alpha_r"),
                    head=want.get("head", "default"))
    if "tag" in want:
        tags = (want.pop("tag"),)
    free = set(want) | {"tag"}
    out = [r for r in rows
           if r["model"] == model and r["task"] == task and r["method"] == method
           and r["tag"] in tags and is_default(r, free)
           and all(_eq(r.get(k), v) for k, v in want.items())
           and (lr is None or _same_lr(r["lr"], lr))]
    seeds = [r["seed"] for r in out]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"duplicate seeds for {model} {task} {method} {want} lr={lr}: "
                         f"{sorted(seeds)}")
    return out


def load_preds(r):
    """(predictions, gold) of one run, in test-set order."""
    with open(r["path"], encoding="utf-8") as f:
        res = json.load(f)["result"]
    return res["test_preds"], res["test_gold"]


def example_scores(preds, gold, multilabel):
    """Per-example contribution to the task metric: correctness for the
    single-label tasks (whose micro-F1 is accuracy), example-based F1 for HoC."""
    p, g = np.asarray(preds, dtype=np.int64), np.asarray(gold, dtype=np.int64)
    if not multilabel:
        return (p == g).astype(float)
    inter = np.array([bin(int(x)).count("1") for x in (p & g)], dtype=float)
    size = np.array([bin(int(x)).count("1") for x in p], dtype=float) + \
        np.array([bin(int(x)).count("1") for x in g], dtype=float)
    return np.where(size > 0, 2.0 * inter / np.maximum(size, 1e-9), 1.0)


def bootstrap(score_by_seed, base_by_seed=None, B=2000, seed=0):
    """Hierarchical bootstrap over seeds and test examples.

    score_by_seed: {seed: per-example scores}. Each replicate resamples seeds
    with replacement, then examples with replacement (the same examples for every
    seed, and for the baseline, so paired comparisons stay paired), and averages.
    With base_by_seed, returns the distribution of the paired difference.
    Returns (point estimate, 2.5th and 97.5th percentiles)."""
    rng = np.random.default_rng(seed)
    seeds = sorted(score_by_seed)
    if base_by_seed is not None:
        seeds = [s for s in seeds if s in base_by_seed]
        mat = np.stack([score_by_seed[s] - base_by_seed[s] for s in seeds])
    else:
        mat = np.stack([score_by_seed[s] for s in seeds])
    n_s, n_x = mat.shape
    est = float(mat.mean())
    reps = np.empty(B)
    for b in range(B):
        si = rng.integers(0, n_s, n_s)
        xi = rng.integers(0, n_x, n_x)
        reps[b] = mat[np.ix_(si, xi)].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, float(lo), float(hi)


def cell(rs, value="score"):
    """(mean, sd, n, {seed: value}) over the runs `rs`."""
    v = {r["seed"]: r[value] for r in rs if r[value] is not None}
    if not v:
        return (float("nan"), float("nan"), 0, {})
    x = np.array([v[s] for s in sorted(v)], dtype=float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x), v)


def paired_test(a_by_seed, b_by_seed):
    """Paired t-test over shared seeds; returns (mean_diff, p, n)."""
    from scipy import stats
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2:
        return (float("nan"), float("nan"), len(seeds))
    a = np.array([a_by_seed[s] for s in seeds], dtype=float)
    b = np.array([b_by_seed[s] for s in seeds], dtype=float)
    d = a - b
    if np.allclose(d, 0):
        return (0.0, 1.0, len(seeds))
    t, p = stats.ttest_rel(a, b)
    return (float(d.mean()), float(p), len(seeds))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def fmt(mean, std, n, bold=False, scale=100):
    if n == 0 or mean != mean:
        return "--"
    s = f"{mean*scale:.1f}\\textsubscript{{$\\pm$\\,{std*scale:.1f}}}"
    return "\\textbf{" + s + "}" if bold else s


def r1(x):
    """Round to the one decimal the tables print (ties are judged on this)."""
    return float(f"{100 * x:.1f}")


def fail_floor(rows, model, task):
    """A run fails when its best dev score is more than FAIL_MARGIN below the
    median dev score of LoRA's runs on the same task."""
    lora = pick(rows, model, task, "lora", tags=SEED_TAGS)
    return float(np.median([r["dev"] for r in lora])) - FAIL_MARGIN if lora else None


def main_cells(rows, model, methods, tasks):
    """{(task, method): runs} for Table I: seeds 1-3, plus seeds 4-5 for FIVE."""
    return {(t, m): pick(rows, model, t, m, tags=SEED_TAGS if m in FIVE else ("",))
            for t in tasks for m in methods}


# --------------------------------------------------------------------------
def table_main(rows, model, methods, tasks, out_path):
    C = main_cells(rows, model, methods, tasks)
    A = {k: cell(v) for k, v in C.items()}
    P = {k: cell(v, "adapter_params") for k, v in C.items()}
    floor = {t: fail_floor(rows, model, t) for t in tasks}

    peft = [m for m in methods if m not in ("full", "linear")]
    best = {t: max((r1(A[(t, m)][0]) for m in peft if A[(t, m)][2]), default=None)
            for t in tasks}

    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Test-set results with a " + MODEL_PRETTY.get(model, model) +
             " backbone (mean$\\pm$s.d.\\ over three seeds; $^\\dagger$five seeds). "
             "Low-rank methods are held to the adapter-parameter budget of uniform "
             "rank 8 (DoRA and AdaLoRA exceed it slightly; the column gives the "
             "parameters actually spent, excluding the classification head that every "
             "method trains). Every method's learning rate is selected on the dev set "
             "from a grid extended until the selection is interior "
             "(Appendix~\\ref{app:lr}). HoC med.: median over seeds. Failed: runs "
             "whose best dev score is more than 10 points below the median of LoRA's "
             "runs on the same task, over all three tasks. Best parameter-efficient "
             "result per column in bold, ties included. Below the rule, the two "
             "instruments of Section~\\ref{sec:family}: \\method{} and the exact "
             "contrast GEV, which keeps uniform rank so that only its "
             "initialisation differs from LoRA's.}",
             "\\label{tab:main}",
             "\\begin{tabular}{l r " + " ".join(["c"] * len(tasks)) + " c c}",
             "\\toprule",
             "Method & Adapter params & " + " & ".join(TASK_PRETTY[t] for t in tasks)
             + " & HoC med. & Failed \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for t in tasks:
            mu, sd, n, _ = A[(t, m)]
            cells.append(fmt(mu, sd, n, bold=(m in peft and n and r1(mu) == best[t])))
        pv = [P[(t, m)][0] for t in tasks if P[(t, m)][2]]
        pstr = "0" if m == "linear" else (f"{np.mean(pv)/1e6:.2f}M" if pv else "--")
        hoc = A.get(("hoc", m))
        med = f"{100*np.median(list(hoc[3].values())):.1f}" if hoc and hoc[2] else "--"
        if m == "linear":
            failed = "--"
        else:
            runs = [r for t in tasks for r in C[(t, m)]]
            nf = sum(r["dev"] < floor[r["task"]] for r in runs if floor[r["task"]] is not None)
            failed = f"{nf}/{len(runs)}" if runs else "--"
        name = PRETTY.get(m, m) + ("$^\\dagger$" if m in FIVE else "")
        if m == "drift":
            lines.append("\\midrule")
        lines.append(f"{name} & {pstr} & " + " & ".join(cells) + f" & {med} & {failed} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return C, A


def main_stats(rows, model, methods, tasks):
    """Paired tests of every parameter-efficient method against LoRA, with a Holm
    correction over the whole family, and the failed runs of each cell."""
    C = main_cells(rows, model, methods, tasks)
    A = {k: cell(v) for k, v in C.items()}
    out, keys, ps = {}, [], []
    for t in tasks:
        for m in methods:
            if m in ("lora", "full", "linear") or not A[(t, m)][2]:
                continue
            d, p, n = paired_test(A[(t, m)][3], A[(t, "lora")][3])
            out[f"{t}:{m}-lora"] = {"delta": 100 * d, "p": p, "n": n}
            keys.append(f"{t}:{m}-lora")
            ps.append(p if p == p else 1.0)
    for k, a in zip(keys, holm(ps)):
        out[k]["p_holm"] = a
    floor = {t: fail_floor(rows, model, t) for t in tasks}
    out["failed"] = {f"{t}:{m}": sorted(r["seed"] for r in C[(t, m)]
                                        if floor[t] is not None and r["dev"] < floor[t])
                     for t in tasks for m in methods if m != "linear"}
    out["fail_floor"] = floor
    return out


# --------------------------------------------------------------------------
def ablation_rows(rows, model, task):
    """(label, runs) of the ChemProt ablation, seeds 1-3."""
    d_lr = lr_for(model, task, "drift")
    e_lr = lr_for(model, task, "eva")
    return [
        ("LoRA (uniform rank, random init)", pick(rows, model, task, "lora")),
        ("Rank allocation only (random init)",
         pick(rows, model, task, "drift", lr=d_lr, tag="allocOnly", init_mode="random")),
        ("Drift init only (uniform rank)",
         pick(rows, model, task, "drift", lr=d_lr, tag="initOnly", alloc_mode="uniform")),
        ("Random orthonormal init, drift ranks",
         pick(rows, model, task, "drift", lr=d_lr, tag="randOrtho", init_mode="rand_ortho")),
        (r"No deflation ($\tau{=}0$), \method{}'s rate",
         pick(rows, model, task, "drift", lr=d_lr, tau=0.0)),
        ("Absolute (unnormalised) drift score",
         pick(rows, model, task, "drift_abs", lr=d_lr)),
        (r"\method{} (full)", pick(rows, model, task, "drift")),
        None,
        ("GEV init (uniform rank)", pick(rows, model, task, "gev")),
        ("EVA, rank-unit budget (its own rule)",
         pick(rows, model, task, "eva", lr=e_lr, alloc_mode="units")),
        (r"\method{}, news reference", pick(rows, model, task, "drift", lr=d_lr, ref="news")),
        (r"\method{}, word-shuffled reference",
         pick(rows, model, task, "drift", lr=d_lr, ref="shuffled")),
        (r"\method{}, random-token reference",
         pick(rows, model, task, "drift", lr=d_lr, ref="random")),
    ]


def table_ablation(rows, model, tasks, out_path):
    """Factorises the method: allocation vs initialisation vs the deflation itself,
    then the principled contrast (GEV), EVA's own budget rule and the reference
    controls, on every task where the variant was run."""
    per_task = {t: ablation_rows(rows, model, t) for t in tasks}
    # parameters EVA's rank-unit rule actually spends, over the tasks it ran on
    units = [cell(dict(x for x in per_task[t] if x is not None)
                  ["EVA, rank-unit budget (its own rule)"], "adapter_params")[0]
             for t in tasks]
    units = sorted(u / 1e6 for u in units if u == u)
    spent = ("\\NUM{X}" if not units else f"{units[0]:.2f}" if f"{units[0]:.2f}" ==
             f"{units[-1]:.2f}" else f"{units[0]:.2f}$--${units[-1]:.2f}")
    # the reference controls were extended to five seeds (Section V-B); give both
    ref5 = {}
    for t in tasks:
        for label, key in ((r"\method{}, news reference", "news"),
                           (r"\method{}, word-shuffled reference", "shuffled"),
                           (r"\method{}, random-token reference", "random")):
            rs = pick(rows, model, t, "drift", lr=lr_for(model, t, "drift"),
                      ref=key, tags=SEED_TAGS)
            if len(rs) == 5:
                ref5.setdefault(t, []).append(f"{100*cell(rs)[0]:.1f}")
    five = "; ".join(f"{TASK_PRETTY[t]} {', '.join(v)}" for t, v in ref5.items())
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Ablation (test F1, mean$\\pm$s.d.; seeds $1$--$3$ throughout, "
             "so the LoRA and \\method{} rows differ from the five-seed values of "
             "Table~\\ref{tab:main}; $1.33$M "
             "adapter parameters except EVA's rank-unit rule, which spends "
             f"${spent}$M). Over five seeds the reference rows give {five} "
             "(Section~\\ref{sec:outliers}). Upper block: only the allocation rule and the "
             "initialisation change, at \\method{}'s learning rate. Lower block: the "
             "generalised-eigenvector contrast (own rate), EVA with its own budget "
             "rule (EVA's rate), and \\method{} with its WikiText reference replaced. "
             "--: not run.}",
             "\\label{tab:ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{l" + "c" * len(tasks) + "}", "\\toprule",
             "Variant & " + " & ".join(TASK_PRETTY[t] for t in tasks) + " \\\\",
             "\\midrule"]
    for i, item in enumerate(per_task[tasks[0]]):
        if item is None:
            lines.append("\\midrule")
            continue
        name = item[0]
        # a cell is printed once all three seeds exist (runs still in progress: --)
        cells = [fmt(*c[:3]) if c[2] >= 3 else "--"
                 for c in (cell(per_task[t][i][1]) for t in tasks)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
PLACEMENT_ROWS = [
    ("All modules", "lora", {"target": "all", "budget_rank": 8}),
    ("All modules", "lora", {"target": "all", "budget_rank": 4}),
    ("Feed-forward only", "lora", {"target": "ffn", "budget_rank": 8}),
    ("Feed-forward only", "lora", {"target": "ffn", "budget_rank": 14}),
    ("Attention only", "lora", {"target": "attn", "budget_rank": 8}),
    None,
    (r"\method{}, feed-forward only", "drift", {"target": "ffn", "budget_rank": 8}),
    (r"\method{}, attention only", "drift", {"target": "attn", "budget_rank": 8}),
]


def placement_runs(rows, model, task, method, cfg):
    if method == "drift":          # the placement variants of DRIFT inherit its rate
        return pick(rows, model, task, method, lr=lr_for(model, task, "drift"), **cfg)
    return pick(rows, model, task, method, **cfg)


def table_placement(rows, model, tasks, out_path):
    # The feed-forward rank-14 HoC cell is at a rate two of its three seeds cannot
    # train at; the caption points at the repaired value so the table is not read
    # as a placement effect (Section V-D).
    rep = cell(pick(rows, model, "hoc", "lora", target="ffn", budget_rank=14,
                    lr=3e-4, tag="ffn14lr"))
    repair = (f" Two of three seeds fail in the feed-forward rank-14 HoC cell at "
              f"its selected rate; at the rate a three-seed dev mean selects, all "
              f"three train and it reaches {rep[0] * 100:.1f}$\\pm${rep[1] * 100:.1f} "
              f"(Section~\\ref{{sec:placement}})." if rep[2] else "")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Adapter placement with LoRA, each configuration at its own "
             "tuned learning rate (test F1, mean$\\pm$s.d., seeds $1$--$3$; "
             "Table~\\ref{tab:main} reports five seeds for the methods that have "
             "them). Rank 14 on "
             "the feed-forward matrices spends about the all-module rank-8 budget; "
             "rank 4 on all modules about the feed-forward rank-8 budget. \\method{} "
             "rows use \\method{}'s rate." + repair + "}",
             "\\label{tab:placement}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrr" + "c" * len(tasks) + "}", "\\toprule",
             "Placement & $r$ & Params & " + " & ".join(TASK_PRETTY[t] for t in tasks)
             + " \\\\", "\\midrule"]
    for item in PLACEMENT_ROWS:
        if item is None:
            lines.append("\\midrule")
            continue
        name, m, cfg = item
        cells, par = [], []
        for t in tasks:
            rs = placement_runs(rows, model, t, m, cfg)
            mu, sd, n, _ = cell(rs)
            cells.append(fmt(mu, sd, n))
            p = cell(rs, "adapter_params")
            if p[2]:
                par.append(p[0])
        pstr = f"{np.mean(par)/1e6:.2f}M" if par else "--"
        lines.append(f"{name} & {cfg['budget_rank']} & {pstr} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
LADDER_ORDER = ["google__bert_uncased_L-2_H-128_A-2", "google__bert_uncased_L-4_H-256_A-4",
                "google__bert_uncased_L-4_H-512_A-8", "google__bert_uncased_L-8_H-512_A-8",
                "google__bert_uncased_L-12_H-768_A-12"]
SWEEP = ("lora", "eva", "eva_white", "drift")


def _stacked(label):
    """Two-line column header for labels like 'EVA (whitened)', so a one-column
    table fits the IEEE column width."""
    head, sep, tail = label.partition(" (")
    return f"\\shortstack{{{head}\\\\({tail}}}" if sep else label


def ladder_cells(rows, task="chemprot"):
    """{(backbone, column): runs}: every method at the rate tuned on that backbone,
    and EVA at the rate tuned for it on RoBERTa-base (the inherited-rate pitfall)."""
    out = {}
    for mdl in LADDER_ORDER:
        for m in SWEEP:
            out[(mdl, m)] = pick(rows, mdl, task, m)
        out[(mdl, "eva_rb")] = pick(rows, mdl, task, "eva",
                                    lr=lr_for("roberta-base", task, "eva"))
    return out


def table_ladder(rows, out_path, task="chemprot"):
    L = ladder_cells(rows, task)
    cols = ["lora", "eva", "eva_white", "drift", "eva_rb"]
    head = {"lora": "LoRA", "eva": "EVA", "eva_white": _stacked("EVA (whitened)"),
            "drift": "\\method{}", "eva_rb": "\\shortstack{EVA\\\\(RoBERTa's rate)}"}
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Backbone ladder on " + TASK_PRETTY.get(task, task) +
             " (test micro-F1, mean$\\pm$s.d., three seeds): the BERT miniatures share "
             "one vocabulary and pretraining recipe. Each method runs at the learning "
             "rate tuned on that backbone's dev set; the last column keeps EVA at the "
             "rate tuned for it on RoBERTa-base.}",
             "\\label{tab:ladder}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.4pt}",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "BERT & " + " & ".join(head[m] for m in cols) + " \\\\", "\\midrule"]
    for mdl in LADDER_ORDER:
        name = MODEL_PRETTY[mdl]
        A = {m: cell(L[(mdl, m)]) for m in cols}
        best = max((r1(A[m][0]) for m in SWEEP if A[m][2]), default=None)
        cells = [fmt(*A[m][:3], bold=(m in SWEEP and A[m][2] and r1(A[m][0]) == best))
                 for m in cols]
        lines.append(f"{name.replace('BERT-', '')} ({MODEL_PARAMS[name]:g}M) & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return L


DECODER = "HuggingFaceTB__SmolLM2-360M"


def _lr_tex(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"$10^{{{int(e)}}}$" if m == "1" else f"${m}{{\\times}}10^{{{int(e)}}}$"


def table_decoder(rows, out_path, tasks=("chemprot", "hoc"), methods=SWEEP):
    """The decoder SLM on ChemProt and HoC: each method at the rate its own
    dev-set tuning selected (HoC at batch size 8)."""
    A = {(t, m): cell(pick(rows, DECODER, t, m)) for t in tasks for m in methods}
    if not any(v[2] for v in A.values()):
        return None
    tasks = [t for t in tasks if any(A[(t, m)][2] for m in methods)]
    best = {t: max(r1(A[(t, m)][0]) for m in methods if A[(t, m)][2]) for t in tasks}
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Decoder SLM: SmolLM2-360M (test F1, mean$\\pm$s.d., three "
             "seeds), each method at its own tuned learning rate (rate above the "
             "score).}",
             "\\label{tab:decoder}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "c" * len(tasks) + "}", "\\toprule",
             "Method & " + " & ".join(TASK_PRETTY[t] for t in tasks) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for t in tasks:
            if not A[(t, m)][2]:
                cells.append("--")
                continue
            lr = _lr_tex(lr_for(DECODER, t, m))
            cells.append(f"\\shortstack{{{lr}\\\\"
                         f"{fmt(*A[(t, m)][:3], bold=(r1(A[(t, m)][0]) == best[t]))}}}")
        lines.append(f"{PRETTY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return A


# --------------------------------------------------------------------------
def _lr_short(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def lr_status_rows(model):
    """(group, configuration, grid tried, selected, interior?) of every tuning
    cell, for the appendix table."""
    import grid
    out = []
    cells = (grid.tuning_cells(hf_name(model)) + grid.tuning_cells_rev2(hf_name(model))
             + grid.tuning_cells_clinical(hf_name(model)))
    seen = set()
    defaults = {"target": "all", "budget_rank": 8, "tau": None,
                "scaling": "alpha_r", "head": "default"}
    for (mdl, task, m, extra, first), tried, todo in grid.tuning_status(hf_name(model), cells):
        # the same cell can be listed by two plans, once with a default spelled out
        key = (mdl, task, m, tuple(sorted((k, v) for k, v in extra.items()
                                          if defaults.get(k, object()) != v)))
        if not tried or key in seen:
            continue
        seen.add(key)
        best = grid.best_lr(tried)
        lrs = sorted(tried)
        interior = not (grid._same(best, lrs[0]) or grid._same(best, lrs[-1]))
        out.append((mdl, task, m, extra, lrs, best, interior, bool(todo)))
    # the decoder's ChemProt tuning predates the revision cells; report it too
    T = grid.load_tuning()
    for key, tried in sorted(T.items(), key=str):
        if key[0] == hf_name(DECODER) and key[1] == "chemprot":
            best = grid.best_lr(tried)
            lrs = sorted(tried)
            interior = not (grid._same(best, lrs[0]) or grid._same(best, lrs[-1]))
            out.append((key[0], key[1], key[2], {}, lrs, best, interior, False))
    return out


def table_lr(model, out_path):
    """Appendix: the selected rate and the grid it was chosen from, for every
    configuration a conclusion is drawn from."""
    rows = lr_status_rows(model)

    def label(mdl, task, m, extra):
        bb = MODEL_PRETTY.get(mdl.replace("/", "__"), mdl)
        bits = [bb, TASK_PRETTY.get(task, task), PRETTY.get(m, m).replace(" (budget-matched)", "")]
        if extra.get("target", "all") != "all":
            bits.append({"ffn": "FFN only", "attn": "attention only"}[extra["target"]])
        if extra.get("budget_rank", 8) != 8:
            bits.append(f"$r{{=}}{extra['budget_rank']}$")
        if extra.get("tau") is not None:
            bits.append(f"$\\tau{{=}}{extra['tau']:g}$")
        if extra.get("scaling", "alpha_r") != "alpha_r":
            bits.append("rsLoRA scale")
        if extra.get("head", "default") != "default":
            bits.append("linear head")
        return ", ".join(bits)

    edges = sum(1 for *_, interior, _ in rows if not interior)
    lines = ["\\begin{table*}[!ht]", "\\centering",
             "\\caption{Learning-rate selection (dev set, seed 1) for all "
             f"{len(rows)} tuned configurations. Each grid starts from the listed "
             "first values and is extended one step past whichever edge holds the "
             "best score until the selection is interior"
             + ("; no selection lies on an edge of its final grid.}" if not edges else
                "; $^\\ast$ marks a selection that is still on an edge because the "
                "next step lies outside the admissible range or diverges.}"),
             "\\label{tab:lr}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{llll}", "\\toprule",
             "Configuration & Selected & Configuration & Selected \\\\", "\\midrule"]
    entries = []
    for mdl, task, m, extra, lrs, best, interior, open_ in rows:
        star = "" if interior else "$^\\ast$"
        rng = f"[{_lr_short(lrs[0])}, {_lr_short(lrs[-1])}]"
        entries.append((label(mdl, task, m, extra),
                        f"{_lr_short(best)}{star} {rng}" + (" (open)" if open_ else "")))
    half = (len(entries) + 1) // 2
    for i in range(half):
        a = entries[i]
        b = entries[i + half] if i + half < len(entries) else ("", "")
        lines.append(f"{a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def table_seeds(rows, model, methods, tasks, out_path):
    """Appendix: every seed of the main table."""
    from scipy import stats as sst
    C = main_cells(rows, model, methods, tasks)
    lines = ["\\begin{table*}[!ht]", "\\centering",
             "\\caption{Per-seed test scores of Table~\\ref{tab:main} (seeds 1, 2, 3"
             " and, for LoRA, EVA, whitened EVA and \\method{}, seeds 4 and 5), with the "
             "95\\% $t$-interval of the mean over seeds.}",
             "\\label{tab:seeds}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{l" + "ll" * len(tasks) + "}", "\\toprule",
             "Method & " + " & ".join(f"{TASK_PRETTY[t]} & 95\\% CI" for t in tasks) + " \\\\",
             "\\midrule"]
    for m in methods:
        cells = []
        for t in tasks:
            mu, sd, n, v = cell(C[(t, m)])
            cells.append(" / ".join(f"{100*v[s]:.1f}" for s in sorted(v)) or "--")
            if n > 1:
                h = sst.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                cells.append(f"[{100*(mu-h):.1f}, {100*(mu+h):.1f}]")
            else:
                cells.append("--")
        lines.append(f"{PRETTY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def table_cost(rows, model, tasks, out_path):
    """Profiling cost against the cost of a single fine-tuning run."""
    sweep, single = {}, {}
    for p in glob.glob(os.path.join(ROOT, "runs", "profiles", "*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        key = d["key"]
        if not key.startswith(model + "__") or not d.get("centered"):
            continue
        if d.get("ref", "wikitext") != "wikitext" or key.endswith("__gev"):
            continue
        task = key[len(model) + 2:].split("__")[0]
        if "__ref1024__dom1024" not in key:
            continue
        (single if len(d.get("taus", [])) == 1 else sweep)[task] = d
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Profiling cost on one NVIDIA T4.}",
             "\\label{tab:cost}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Task & Forward & Spectral (1 $\\tau$) & Spectral (5 $\\tau$) "
             "& vs.\\ LoRA \\\\", "\\midrule"]
    for t in tasks:
        d, d1 = sweep.get(t), single.get(t)
        tr = [r["train_time_s"] for r in pick(rows, model, t, "lora")]
        if not d and not d1:
            lines.append(f"{TASK_PRETTY.get(t,t)} & -- & -- & -- & -- \\\\")
            continue
        ref = d1 or d
        fwd = ref["t_ref_s"] + ref["t_dom_s"]
        e1 = f"{d1['t_eig_s']:.0f}\\,s" if d1 else "--"
        e5 = f"{d['t_eig_s']:.0f}\\,s" if d else "--"
        rel = (f"{100*(fwd + d1['t_eig_s'])/np.mean(tr):.0f}\\%" if d1 and tr else "--")
        lines.append(f"{TASK_PRETTY.get(t,t)} & {fwd:.0f}\\,s & {e1} & {e5} & {rel} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# tables added after the review audit
# --------------------------------------------------------------------------
MULTILABEL = {"chemprot": False, "rct20k": False, "hoc": True}


def ci_cells(rows, model, methods, tasks):
    """{(task, method): {seed: per-example scores}} from the replicate of
    Table I that stores predictions (tag 'preds'), at the table's rates."""
    out = {}
    for t in tasks:
        for m in methods:
            rs = [r for r in pick(rows, model, t, m, tags=("preds",)) if r["has_preds"]]
            out[(t, m)] = {r["seed"]: example_scores(*load_preds(r), MULTILABEL[t])
                           for r in rs}
    return out


def table_ci(rows, model, methods, tasks, out_path):
    """Appendix: hierarchical (seed x example) bootstrap intervals of every Table I
    cell and of its difference to LoRA, from the replicate that stores
    predictions."""
    # the comparisons with LoRA are between parameter-efficient methods
    methods = [m for m in methods if m not in ("full", "linear")]
    S = ci_cells(rows, model, methods, tasks)
    C = main_cells(rows, model, methods, tasks)
    stats = {}
    lines = ["\\begin{table*}[!ht]", "\\centering",
             "\\caption{Instance-level uncertainty for Table~\\ref{tab:main}: a "
             "replication of every parameter-efficient run that stores per-example "
             "test predictions, with 95\\% hierarchical bootstrap intervals (seeds, "
             "then test examples, $2{,}000$ replicates). $\\Delta$: paired difference "
             "to LoRA (same seeds and examples). Rep.: replicate mean minus the "
             "Table~I mean (fp16 run-to-run variation).}",
             "\\label{tab:ci}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{l" + "ccc" * len(tasks) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{TASK_PRETTY[t]}}}" for t in tasks)
             + " \\\\",
             "Method" + " & Mean [95\\% CI] & $\\Delta$ vs LoRA [95\\% CI] & Rep." * len(tasks)
             + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for t in tasks:
            sc = S[(t, m)]
            if not sc:
                cells += ["--", "--", "--"]
                continue
            est, lo, hi = bootstrap(sc)
            d = bootstrap(sc, S[(t, "lora")]) if m != "lora" and S[(t, "lora")] else None
            rep = est - cell(C[(t, m)])[0] if C[(t, m)] else float("nan")
            stats[f"{t}:{m}"] = {"mean": est, "ci": [lo, hi], "n_seeds": len(sc),
                                 "delta": d, "rep_diff": rep}
            cells.append(f"{100*est:.1f} [{100*lo:.1f}, {100*hi:.1f}]")
            cells.append("--" if d is None else
                         f"{100*d[0]:+.1f} [{100*d[1]:+.1f}, {100*d[2]:+.1f}]")
            cells.append(f"{100*rep:+.1f}" if rep == rep else "--")
        lines.append(f"{PRETTY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return stats


def fp32_cells(rows, model, methods, task="hoc"):
    return {m: pick(rows, model, task, m, tag="fp32", det=True) for m in methods}


def table_fp32(rows, model, methods, out_path, task="hoc"):
    """HoC in fp16 (Table I) against deterministic fp32 reruns at the same
    rates: mean, median and failed runs of each method."""
    C = main_cells(rows, model, methods, [task])
    F = fp32_cells(rows, model, methods, task)
    floor = fail_floor(rows, model, task)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{HoC in fp16 (Table~\\ref{tab:main}) and rerun in fp32 at the "
             "same learning rates, PyTorch's deterministic algorithms on (example-based F1, "
             "three seeds; fp16 rows with $^\\dagger$ have five). Failed: best dev "
             "score more than 10 points below LoRA's median.}",
             "\\label{tab:fp32}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{fp16} & \\multicolumn{3}{c}{fp32} \\\\",
             "Method & Mean & Med. & Failed & Mean & Med. & Failed \\\\", "\\midrule"]

    def trio(rs, m):
        mu, sd, n, v = cell(rs)
        if not n:
            return ["--", "--", "--"]
        med = f"{100*np.median(list(v.values())):.1f}"
        if m == "linear":          # cannot reach LoRA's level by construction
            return [fmt(mu, sd, n), med, "--"]
        nf = sum(r["dev"] < floor for r in rs) if floor is not None else 0
        return [fmt(mu, sd, n), med, f"{nf}/{n}"]

    for m in methods:
        name = PRETTY.get(m, m).replace(" (budget-matched)", "") + \
            ("$^\\dagger$" if m in FIVE else "")
        lines.append(f"{name} & " + " & ".join(trio(C[(task, m)], m) + trio(F[m], m))
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return F


BUDGET_X_TASKS = ("rct20k", "hoc")
BUDGET_X_RANKS = (4, 8, 16)


def table_budget_x(rows, model, out_path):
    """Budgets of 0.5%, 1.06% and 2% on the other two tasks, each (method,
    budget) at its own tuned rate; rank 8 is Table I (seeds 1-3)."""
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{Adapter budget on RCT-20k and HoC (mean$\\pm$s.d., three "
             "seeds, every method and budget tuned on its own). Rank $4$, $8$ and "
             "$16$ spend $0.53\\%$, $1.06\\%$ and $2.1\\%$ of the backbone.}",
             "\\label{tab:budgetx}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "Task & $r$ & LoRA & EVA & \\shortstack{EVA\\\\(whitened)} & \\method{} \\\\",
             "\\midrule"]
    for t in BUDGET_X_TASKS:
        for r in BUDGET_X_RANKS:
            A = {m: cell(pick(rows, model, t, m, budget_rank=r)) for m in SWEEP}
            best = max((r1(A[m][0]) for m in SWEEP if A[m][2]), default=None)
            cells = [fmt(*A[m][:3], bold=(A[m][2] and r1(A[m][0]) == best)) for m in SWEEP]
            lines.append(f"{TASK_PRETTY[t] if r == BUDGET_X_RANKS[0] else ''} & {r} & "
                         + " & ".join(cells) + " \\\\")
        if t != BUDGET_X_TASKS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


LINHEAD = ("lora", "eva", "eva_white", "drift", "bitfit")


def table_linhead(rows, model, out_path, task="chemprot"):
    """The main comparison with a single linear classification head."""
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{ChemProt with the backbone's classification head "
             "(Table~\\ref{tab:main}, seeds 1--3) and with a single linear head on the "
             "first token ($768\\times13$), every configuration tuned on its own "
             "(test micro-F1, three seeds).}",
             "\\label{tab:linhead}", "\\footnotesize",
             "\\begin{tabular}{lcc}", "\\toprule",
             "Method & \\shortstack{RoBERTa head\\\\($0.60$M)} & "
             "\\shortstack{Linear head\\\\($0.01$M)} \\\\", "\\midrule"]
    for m in LINHEAD:
        a = cell(pick(rows, model, task, m))
        b = cell(pick(rows, model, task, m, head="linear"))
        lines.append(f"{PRETTY.get(m, m)} & {fmt(*a[:3])} & {fmt(*b[:3])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def table_divergence(out_path, model="roberta-base", tasks=("chemprot", "rct20k", "hoc")):
    """Tau-free divergences between each task's activation covariances and the
    reference's, next to the reference-vs-reference floor, averaged over the
    distinct input sites."""
    try:
        import figures
    except ImportError:          # the Kaggle notebooks do not carry the plotting code
        return False
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{Tau-free divergence of each corpus's input covariances from "
             "the WikiText reference (trace-normalised; mean over the $48$ distinct "
             "input sites). Floor: a disjoint half of WikiText at the same sequence "
             "length. News and random tokens at ChemProt's length.}",
             "\\label{tab:div}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Corpus vs.\\ WikiText & CORAL & Bures & Jeffreys \\\\", "\\midrule"]
    found = False
    for t in tasks:
        p = os.path.join(ROOT, "runs", "divergence", f"{model}__{t}.json")
        if not os.path.exists(p):
            continue
        found = True
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sites = {}
        for n, v in d["modules"].items():
            sites[(figures.layer_of(n), figures.SITE_OF[figures.classify(n)])] = v
        def mean(pair, key):
            vals = [v[pair][key] for v in sites.values() if pair in v]
            return float(np.mean(vals)) if vals else float("nan")
        for pair, label in (("target", TASK_PRETTY[t]), ("ref2", f"\\quad floor ({TASK_PRETTY[t]} length)"),
                            ("news", "News (CNN/DailyMail)"), ("random", "Random tokens")):
            if pair in next(iter(sites.values())):
                lines.append(f"{label} & {mean(pair, 'coral'):.3f} & {mean(pair, 'bures'):.4f} "
                             f"& {mean(pair, 'jeffreys'):.3f} \\\\")
    if not found:                   # placeholder until the divergence job has run
        lines += [f"{TASK_PRETTY[t]} & -- & -- & -- \\\\" for t in tasks]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return found


# --------------------------------------------------------------------------
CLIN = "mtsamples"


def clinical_rows(rows, model):
    """(label, runs) of the clinical-notes table: the four methods at their own
    tuned rates, and DRIFT with its reference replaced, at DRIFT's rate."""
    d_lr = lr_for(model, CLIN, "drift")
    out = [(PRETTY[m], pick(rows, model, CLIN, m)) for m in SWEEP]
    out += [(r"\method{}, news reference", pick(rows, model, CLIN, "drift", lr=d_lr, ref="news")),
            (r"\method{}, random-token reference",
             pick(rows, model, CLIN, "drift", lr=d_lr, ref="random"))]
    return out


def table_clinical(rows, model, out_path):
    """Clinical notes (MTSamples specialties): micro- and macro-F1, failed runs
    and the paired difference to LoRA."""
    R = clinical_rows(rows, model)
    stats = {}
    lora = dict(R)[PRETTY["lora"]]
    floor = fail_floor(rows, model, CLIN) if lora else None
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Clinical notes (MTSamples, 12 specialties; test micro- and "
             "macro-F1, mean$\\pm$s.d.\\ over three seeds). Each method runs at the rate "
             "tuned on the clinical dev set; the reference rows use \\method{}'s rate. "
             "$\\Delta$: paired difference to LoRA in micro-F1. Failed: as in "
             "Table~\\ref{tab:main}.}",
             "\\label{tab:clinical}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "Method & Micro-F1 & Macro-F1 & $\\Delta$ ($p$) & Failed \\\\", "\\midrule"]
    for i, (label, rs) in enumerate(R):
        if i == len(SWEEP):
            lines.append("\\midrule")
        a, b = cell(rs), cell(rs, "macro_f1")
        if rs and label != PRETTY["lora"] and lora:
            d, p, n = paired_test(a[3], cell(lora)[3])
            delta = f"{100*d:+.1f} ({p:.2f})" if n >= 2 else "--"
        else:
            d = p = None
            delta = "--"
        nf = sum(r["dev"] < floor for r in rs) if rs and floor is not None else None
        stats[label] = {"micro": a[:3], "macro": b[:3], "delta": d, "p": p, "failed": nf,
                        "lr": rs[0]["lr"] if rs else None,
                        "n_train": rs[0].get("n_train") if rs else None}
        lines.append(f"{label} & {fmt(*a[:3])} & {fmt(*b[:3])} & {delta} & "
                     + (f"{nf}/{len(rs)}" if nf is not None else "--") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    # written even before the runs exist (rows of --), so the reference resolves
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return stats


# --------------------------------------------------------------------------
def summary(A):
    return {f"{k[0]}:{k[1]}" if isinstance(k, tuple) else str(k):
            (round(100 * v[0], 2), round(100 * v[1], 2), v[2]) for k, v in A.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--tasks", default="chemprot,rct20k,hoc")
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()

    model = a.model.replace("/", "__")
    tasks = a.tasks.split(",")
    rows = load_all()
    print(f"loaded {len(rows)} runs")
    if a.dump:
        seen = defaultdict(int)
        for r in rows:
            seen[(r["model"], r["task"], r["method"])] += 1
        for k in sorted(seen, key=str):
            print(" ", k, seen[k])

    # the last two rows are the instruments: DRIFT (deflation) and GEV (the exact
    # contrast), separated from the published methods by a rule in table_main
    methods = ["full", "linear", "bitfit", "lora", "dora", "pissa",
               "adalora", "eva", "eva_white", "drift", "gev"]
    os.makedirs(OUT, exist_ok=True)
    C, A = table_main(rows, model, methods, tasks, os.path.join(OUT, "tab_main.tex"))
    for k, v in sorted(A.items(), key=str):
        print(f"  {k}: {100*v[0]:.2f} +- {100*v[1]:.2f}  (n={v[2]})")
    table_ablation(rows, model, tasks, os.path.join(OUT, "tab_ablation.tex"))
    table_placement(rows, model, tasks, os.path.join(OUT, "tab_placement.tex"))
    table_cost(rows, model, tasks, os.path.join(OUT, "tab_cost.tex"))
    table_ladder(rows, os.path.join(OUT, "tab_ladder.tex"))
    table_decoder(rows, os.path.join(OUT, "tab_decoder.tex"))
    table_lr(model, os.path.join(OUT, "tab_lr.tex"))
    table_seeds(rows, model, methods, tasks, os.path.join(OUT, "tab_seeds.tex"))
    table_fp32(rows, model, methods, os.path.join(OUT, "tab_fp32.tex"))
    table_budget_x(rows, model, os.path.join(OUT, "tab_budgetx.tex"))
    table_linhead(rows, model, os.path.join(OUT, "tab_linhead.tex"))
    table_divergence(os.path.join(OUT, "tab_div.tex"),
                     tasks=("chemprot", "rct20k", "hoc", CLIN))
    ci = table_ci(rows, model, methods, tasks, os.path.join(OUT, "tab_ci.tex"))
    with open(os.path.join(OUT, "ci.json"), "w") as f:
        json.dump(ci, f, indent=1)
    clin = table_clinical(rows, model, os.path.join(OUT, "tab_clinical.tex"))
    with open(os.path.join(OUT, "clinical.json"), "w") as f:
        json.dump(clin, f, indent=1, default=str)
    print("wrote tab_main, tab_ablation, tab_placement, tab_cost, tab_ladder, "
          "tab_decoder, tab_lr, tab_seeds, tab_fp32, tab_budgetx, tab_linhead, "
          "tab_div, tab_ci, tab_clinical")
    st = main_stats(rows, model, methods, tasks)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(st, f, indent=2)


if __name__ == "__main__":
    main()
