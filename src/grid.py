"""Experiment orchestration.

Runs a priority-ordered list of configurations as subprocesses, skipping any run
whose result JSON already exists, so the whole grid is resumable and partial
results are always usable.

    python src/grid.py --plan main --dry
    python src/grid.py --plan main
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable
RUN = os.path.join(ROOT, "src", "run.py")
PROF = os.path.join(ROOT, "src", "profile_drift.py")

TASKS = ["chemprot", "rct20k", "hoc"]
NEEDS_PROFILE = {"eva", "eva_white", "drift", "drift_abs", "drift_nodeflate", "gev"}

# Per-task training configuration.
TASK_CFG = {
    "chemprot": dict(epochs=10, batch_size=32, max_len=128),
    "rct20k":   dict(epochs=8,  batch_size=32, max_len=96),
    "hoc":      dict(epochs=12, batch_size=16, max_len=512),
    # clinical notes (MTSamples specialties): long documents like HoC's
    "mtsamples": dict(epochs=10, batch_size=16, max_len=512),
}

AMP = os.environ.get("DRIFT_AMP", "0") == "1"

# Learning rates; filled in by the tuning plan and then frozen here.
LR = {
    "full": 2e-5, "linear": 1e-3, "bitfit": 1e-3,
    "lora": 3e-4, "dora": 3e-4, "pissa": 3e-4, "adalora": 3e-4,
    "eva": 3e-4, "eva_white": 3e-4, "drift": 3e-4, "drift_abs": 3e-4,
    "drift_nodeflate": 3e-4, "gev": 3e-4,
}

LADDER = [
    "google/bert_uncased_L-2_H-128_A-2",    # 4.4M   BERT-Tiny
    "google/bert_uncased_L-4_H-256_A-4",    # 11.2M  BERT-Mini
    "google/bert_uncased_L-4_H-512_A-8",    # 28.8M  BERT-Small
    "google/bert_uncased_L-8_H-512_A-8",    # 41.4M  BERT-Medium
    "google/bert_uncased_L-12_H-768_A-12",  # 110M   BERT-Base
]


def cfg_for(task, overrides=None):
    c = dict(TASK_CFG[task])
    if overrides:
        c.update(overrides)
    return c


# Every method is tuned on its own over the same dev-set protocol, and so is
# every configuration a conclusion is drawn from: each adapter placement, each
# budget of the budget sweep, each deflation level tau and each backbone of the
# ladder. Variants used only in the ablation (allocation/initialisation
# factorisation, reference-corpus controls, fp32 reruns) inherit the rate tuned
# for their parent configuration; a method with no tuning results at all falls
# back to LoRA's.
LORA_FAMILY = {"lora", "dora", "pissa", "adalora", "eva", "eva_white", "drift",
               "drift_abs", "drift_nodeflate", "gev"}
PARENT = {"drift_abs": "drift", "drift_nodeflate": "drift"}
_TUNE = None


def tune_key(model, task, method, target="all", budget_rank=8, tau=None,
             scaling="alpha_r", head="default"):
    """What a learning rate is selected for. DRIFT's deflation level is part of
    the key (the other profile methods run at tau = 0 whatever the flag says),
    and so are the adapter scale and the classification head when they differ
    from the default."""
    parts = []
    if method == "drift" and tau is not None and abs(float(tau) - 0.95) > 1e-9:
        parts.append(f"tau{float(tau):g}")
    if scaling and scaling != "alpha_r":
        parts.append(scaling)
    if head and head != "default":
        parts.append(f"head-{head}")
    return (model, task, method, target or "all", int(budget_rank or 8), "+".join(parts))


def load_tuning(refresh=False):
    """{tune_key: {lr: dev score}} from every seed-1 run tagged 'tune'."""
    global _TUNE
    if _TUNE is not None and not refresh:
        return _TUNE
    import glob
    _TUNE = {}
    for p in glob.glob(os.path.join(ROOT, "runs", "results", "*tune*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        a = r.get("args", {})
        if a.get("tag") != "tune":
            continue
        # tuning runs of EVA/DRIFT made before the covariance/scaling fix
        # (no "cov" arg) must not steer the corrected runs
        if a.get("method") in NEEDS_PROFILE and a.get("cov") != "centered":
            continue
        metric = r.get("metric", "micro_f1")
        dev = r.get("result", {}).get("dev_best", {}).get(metric)
        if dev is None:
            continue
        k = tune_key(a.get("model"), a.get("task"), a.get("method"),
                     a.get("target", "all"), a.get("budget_rank", 8), a.get("tau"),
                     a.get("scaling", "alpha_r"), a.get("head", "default"))
        _TUNE.setdefault(k, {})[float(a.get("lr"))] = dev
    return _TUNE


def best_lr(scores):
    """Highest dev score; an exact tie goes to the smaller rate."""
    return max(sorted(scores), key=lambda lr: scores[lr])


def resolve_lr(model, task, method, target="all", budget_rank=8, tau=None,
               scaling="alpha_r", head="default"):
    """Best dev-set learning rate for this configuration, else the rate of its
    parent configuration (all modules, rank 8, default tau, scale and head),
    else LoRA's, else the default."""
    T = load_tuning()
    own = PARENT.get(method, method)
    keys = [tune_key(model, task, own, target, budget_rank, tau, scaling, head)]
    if own == "drift" and tau is not None and float(tau) == 0.0:
        # DRIFT at tau = 0 is EVA exactly (same profile, allocation and
        # initialisation), so it is selected by EVA's tuning
        keys = [tune_key(model, task, "eva", target, budget_rank)]
    keys.append(tune_key(model, task, own))
    if method in LORA_FAMILY:
        keys.append(tune_key(model, task, "lora"))
    for k in keys:
        if T.get(k):
            return best_lr(T[k])
    return LR[method]


def make_cmd(model, task, method, seed, lr=None, amp=None, **kw):
    c = cfg_for(task, {k: v for k, v in kw.items()
                       if k in ("epochs", "batch_size", "max_len")})
    if lr is None:
        lr = resolve_lr(model, task, method, target=kw.get("target") or "all",
                        budget_rank=kw.get("budget_rank") or 8, tau=kw.get("tau"),
                        scaling=kw.get("scaling") or "alpha_r",
                        head=kw.get("head") or "default")
    cmd = [PY, RUN, "--model", model, "--task", task, "--method", method,
           "--seed", str(seed), "--lr", str(lr),
           "--epochs", str(c["epochs"]), "--batch_size", str(c["batch_size"]),
           "--max_len", str(c["max_len"])]
    for k in ("budget_rank", "tau", "rho", "score_mode", "init_mode",
              "alloc_mode", "target", "max_train", "tag", "ref", "n_ref", "n_dom",
              "head", "scaling"):
        if k in kw and kw[k] is not None:
            cmd += ["--" + k, str(kw[k])]
    if kw.get("deterministic"):
        cmd += ["--deterministic"]
    if AMP if amp is None else amp:
        cmd += ["--amp"]
    return cmd


def cmd_args(cmd):
    """{--flag: value} of a run.py command; bare flags map to True."""
    out, toks, i = {}, cmd[2:], 0
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[toks[i]] = toks[i + 1]
            i += 2
        else:
            out[toks[i]] = True
            i += 1
    return out


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------
def plan_tune(model):
    """LR selection, one seed, dev-set decision, for every main-table method."""
    out = []
    # Each baseline grid extends one step past the value first selected, so no
    # baseline's chosen rate sits on the edge of its search range.
    grids = {"lora": [1e-4, 3e-4, 1e-3], "full": [1e-5, 3e-5, 5e-5],
             "linear": [1e-3, 5e-3, 2e-2], "bitfit": [3e-4, 1e-3, 3e-3]}
    # every other low-rank method gets its own search; 1e-3 is omitted because
    # it already diverges for plain LoRA on HoC
    for m in ("dora", "pissa", "adalora"):
        grids[m] = [1e-4, 3e-4]
    # profile-initialised methods get two lower rates as well: EVA's principal
    # directions carry RoBERTa's high-variance outlier features and diverge at
    # 1e-4 and above, so its grid must reach down to where it can train. DRIFT
    # gets the identical grid so the comparison stays matched.
    for m in ("eva", "eva_white", "drift"):
        grids[m] = [1e-5, 3e-5, 1e-4, 3e-4]
    for task in TASKS:
        for method, lrs in grids.items():
            for lr in lrs:
                out.append(make_cmd(model, task, method, 1, lr=lr, tag="tune"))
    return out


def plan_main(model, seeds=(1, 2, 3), methods=None):
    methods = methods or ["drift", "lora", "eva", "eva_white", "adalora", "full",
                          "bitfit", "linear", "pissa", "dora"]
    out = []
    # seed-major ordering: a complete 1-seed table exists as early as possible
    for seed in seeds:
        for task in TASKS:
            for method in methods:
                out.append(make_cmd(model, task, method, seed))
    return out


SWEEP_METHODS = ["drift", "lora", "eva", "eva_white"]


def plan_ladder(model_list=None, seeds=(1, 2, 3), task="chemprot",
                lr_from="roberta-base"):
    """The small backbones are not tuned separately: each method uses the rate
    it was tuned to on the primary backbone for the same task."""
    models = model_list or LADDER
    # DRIFT_LADDER_LR (JSON {method: lr}) pins the rates when the tuning results
    # are not available in this session, e.g. a ladder-only run
    pinned = json.loads(os.environ.get("DRIFT_LADDER_LR", "{}"))
    out = []
    for seed in seeds:
        for model in models:
            for method in SWEEP_METHODS:
                lr = pinned.get(method) or resolve_lr(lr_from, task, method)
                out.append(make_cmd(model, task, method, seed, lr=lr))
    return out


def plan_ladder_lr(model_list=None, seeds=(1, 2, 3), task="chemprot",
                   lr_from="roberta-base"):
    """Control for the ladder: EVA at the learning rate the other low-rank
    methods use there (LoRA's), instead of the lower rate EVA was tuned to on the
    primary backbone. Tagged so it never replaces the tuned-rate runs in the
    tables."""
    models = model_list or LADDER
    pinned = json.loads(os.environ.get("DRIFT_LADDER_LR", "{}"))
    lr = pinned.get("lora") or resolve_lr(lr_from, task, "lora")
    return [make_cmd(model, task, "eva", seed, lr=lr, tag="ladderLR")
            for seed in seeds for model in models]


def pinned_lr(model, task, method, target="all"):
    """The tuned rate, pinned from the local results when the tuning runs are not
    restorable in a remote session (DRIFT_PINNED_LR, JSON {task: {method: lr}})."""
    pinned = json.loads(os.environ.get("DRIFT_PINNED_LR", "{}"))
    hit = pinned.get(task, {}).get(PARENT.get(method, method)) if target == "all" else None
    return hit or resolve_lr(model, task, method, target=target)


def plan_placement_x(model, seeds=(1, 2, 3), tasks=("rct20k", "hoc")):
    """The adapter-placement ablation of plan_ablation, on the other two tasks.
    LoRA runs at the rate tuned for each placement; DRIFT inherits its own."""
    return [make_cmd(model, task, method, seed,
                     lr=pinned_lr(model, task, method, target=tgt), target=tgt)
            for seed in seeds for task in tasks
            for method in ("lora", "drift") for tgt in ("attn", "ffn")]


def plan_seeds45(model, seeds=(4, 5)):
    """Two further seeds for the four methods the analysis turns on. Tagged, so the
    main table keeps the three seeds every method has."""
    return [make_cmd(model, task, method, seed, lr=pinned_lr(model, task, method),
                     tag="seeds45")
            for seed in seeds for task in TASKS
            for method in ("lora", "eva", "eva_white", "drift")]


DECODER = "HuggingFaceTB/SmolLM2-360M"
DECODER_METHODS = ("lora", "eva", "eva_white", "drift")


def plan_decoder_tune(model, task="chemprot"):
    """Learning rates for the decoder SLM, tuned on its own dev set with the grids
    used for RoBERTa-base (profile-initialised methods reach two rates lower)."""
    grids = {"lora": [1e-4, 3e-4, 1e-3]}
    for m in ("eva", "eva_white", "drift"):
        grids[m] = [1e-5, 3e-5, 1e-4, 3e-4]
    return [make_cmd(model, task, m, 1, lr=lr, tag="tune")
            for m, lrs in grids.items() for lr in lrs]


def plan_decoder_tune_ext(model, task="chemprot"):
    """Every method chose the top of its first grid on the decoder, so each grid
    extends past it (and the profile-initialised methods now reach LoRA's rate)."""
    grids = {"lora": [3e-3]}
    for m in ("eva", "eva_white", "drift"):
        grids[m] = [1e-3, 3e-3]
    return [make_cmd(model, task, m, 1, lr=lr, tag="tune")
            for m, lrs in grids.items() for lr in lrs]


def plan_decoder_main(model, seeds=(1, 2, 3), task="chemprot"):
    return [make_cmd(model, task, m, seed) for seed in seeds for m in DECODER_METHODS]


def plan_budget(model, seeds=(1, 2), task="chemprot"):
    out = []
    for seed in seeds:
        for r in [1, 2, 4, 8, 16]:
            for method in SWEEP_METHODS:
                out.append(make_cmd(model, task, method, seed, budget_rank=r))
    return out


def plan_ablation(model, seeds=(1, 2, 3), task="chemprot"):
    out = []
    for seed in seeds:
        # tau sweep
        for tau in [0.0, 0.5, 0.9, 0.95, 0.99]:
            out.append(make_cmd(model, task, "drift", seed, tau=tau))
        # factorised: allocation vs initialisation
        out.append(make_cmd(model, task, "drift", seed, init_mode="random",
                            tag="allocOnly"))
        out.append(make_cmd(model, task, "drift", seed, alloc_mode="uniform",
                            tag="initOnly"))
        out.append(make_cmd(model, task, "drift", seed, init_mode="rand_ortho",
                            tag="randOrtho"))
        # scoring variant
        out.append(make_cmd(model, task, "drift_abs", seed))
        # module targeting
        for tgt in ["attn", "ffn"]:
            out.append(make_cmd(model, task, "drift", seed, target=tgt))
            out.append(make_cmd(model, task, "lora", seed, target=tgt))
    return out


# --------------------------------------------------------------------------
# revision: adaptive tuning of every configuration, and the runs that use it
# --------------------------------------------------------------------------
# A grid is extended one step past whichever edge holds the best dev score,
# until the selected rate is interior or the ladder of rates ends.
LADDER_LR = {
    "full": [3e-6, 1e-5, 3e-5, 5e-5, 1e-4, 2e-4],
    "linear": [5e-4, 1e-3, 5e-3, 2e-2, 5e-2, 1e-1, 2e-1],
    "bitfit": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
}
LOWRANK_LR = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# the first grids of the main-table tuning (plan_tune)
MAIN_GRIDS = {"lora": [1e-4, 3e-4, 1e-3], "full": [1e-5, 3e-5, 5e-5],
              "linear": [1e-3, 5e-3, 2e-2], "bitfit": [3e-4, 1e-3, 3e-3],
              "dora": [1e-4, 3e-4], "pissa": [1e-4, 3e-4], "adalora": [1e-4, 3e-4],
              "eva": [1e-5, 3e-5, 1e-4, 3e-4], "eva_white": [1e-5, 3e-5, 1e-4, 3e-4],
              "drift": [1e-5, 3e-5, 1e-4, 3e-4]}
MAIN_METHODS = ["drift", "lora", "eva", "eva_white", "adalora", "full", "bitfit",
                "linear", "pissa", "dora"]
# (target, uniform rank): rank 14 on the feed-forward matrices spends 1.29M, about
# the all-module rank-8 budget; rank 4 on all modules spends 0.66M, about the
# feed-forward rank-8 budget
PLACEMENTS = [("attn", 8), ("ffn", 8), ("ffn", 14), ("all", 4)]
SWEEP_RANKS = [1, 2, 4, 16]
TAU_SWEEP = [0.5, 0.9, 0.99]          # 0.95 is DRIFT itself; 0 is EVA exactly


def _same(a, b):
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def _step(method, lr, up):
    """The next rate above (or below) lr on the method's ladder, or None."""
    lad = LADDER_LR.get(method, LOWRANK_LR)
    if up:
        nxt = [x for x in lad if x > lr and not _same(x, lr)]
        return nxt[0] if nxt else None
    nxt = [x for x in lad if x < lr and not _same(x, lr)]
    return nxt[-1] if nxt else None


def tuning_cells(model):
    """Every configuration whose learning rate is selected on its own:
    (backbone, task, method, extra run arguments, first grid)."""
    cells = []
    for task in TASKS:                          # main table: grid edges only
        for m in MAIN_METHODS:
            cells.append((model, task, m, {}, MAIN_GRIDS[m]))
    for task in TASKS:                          # adapter placement (LoRA)
        for tgt, r in PLACEMENTS:
            cells.append((model, task, "lora", {"target": tgt, "budget_rank": r},
                          [1e-4, 3e-4, 1e-3]))
    for task in TASKS:                          # generalised-eigenvector init
        cells.append((model, task, "gev", {}, [1e-4, 3e-4, 1e-3]))
    for tau in TAU_SWEEP:                       # each deflation level
        cells.append((model, "chemprot", "drift", {"tau": tau}, [1e-4, 3e-4, 1e-3]))
    for r in SWEEP_RANKS:                       # each budget of the sweep
        for m in SWEEP_METHODS:
            first = [3e-5, 1e-4, 3e-4] if m == "eva" else [1e-4, 3e-4, 1e-3]
            cells.append((model, "chemprot", m, {"budget_rank": r}, first))
    for bb in LADDER:                           # each backbone of the ladder
        for m in SWEEP_METHODS:
            cells.append((bb, "chemprot", m, {}, [1e-4, 3e-4, 1e-3]))
    return cells


def tuning_cells_rev2(model, part="all"):
    """The configurations added after the audit of the review: LoRA with rsLoRA's
    scale at every budget, the budgets of 0.5% and 2% on the other two tasks and
    the linear-head control (part 'a'), and the decoder on HoC (part 'b', by far
    the most expensive, so it runs last)."""
    cells = []
    if part in ("all", "b"):
        for m in DECODER_METHODS:               # decoder SLM on HoC (batch 8)
            first = [1e-4, 3e-4, 1e-3] if m == "eva" else [3e-4, 1e-3, 3e-3]
            cells.append((DECODER, "hoc", m, {"batch_size": DECODER_HOC_BATCH}, first))
    if part == "b":
        return cells
    for r in [1, 2, 4, 8, 16]:                  # rsLoRA scale alpha/sqrt(r)
        cells.append((model, "chemprot", "lora", {"budget_rank": r, "scaling": "rslora"},
                      [1e-4, 3e-4, 1e-3]))
    for task in ("rct20k", "hoc"):              # 0.5% and 2% budgets elsewhere
        for r in (4, 16):
            for m in SWEEP_METHODS:
                first = [3e-5, 1e-4, 3e-4] if m == "eva" else [1e-4, 3e-4, 1e-3]
                cells.append((model, task, m, {"budget_rank": r}, first))
    for m in LINHEAD_METHODS:                   # linear classification head
        first = {"eva": [3e-5, 1e-4, 3e-4], "bitfit": [3e-4, 1e-3, 3e-3]}.get(
            m, [1e-4, 3e-4, 1e-3])
        cells.append((model, "chemprot", m, {"head": "linear"}, first))
    return cells


DECODER_HOC_BATCH = 8          # 512-token documents: the decoder fits a T4 at 8
LINHEAD_METHODS = ("lora", "eva", "eva_white", "drift", "bitfit")


def tuning_status(model, cells=None):
    """[(cell, {lr: dev}, rates still to run)] for every tuning cell."""
    T = load_tuning(refresh=True)
    out = []
    for cell in (cells if cells is not None else tuning_cells(model)):
        mdl, task, m, extra, first = cell
        tried = T.get(tune_key(mdl, task, m, extra.get("target", "all"),
                               extra.get("budget_rank", 8), extra.get("tau"),
                               extra.get("scaling", "alpha_r"),
                               extra.get("head", "default")), {})
        todo = [lr for lr in first if not any(_same(lr, x) for x in tried)]
        if not todo:
            best, lrs = best_lr(tried), sorted(tried)
            nxt = None
            if _same(best, lrs[-1]):
                nxt = _step(m, best, up=True)
            elif _same(best, lrs[0]):
                nxt = _step(m, best, up=False)
            if nxt is not None and not any(_same(nxt, x) for x in tried):
                todo = [nxt]
        out.append((cell, tried, todo))
    return out


def plan_tune_rev(model, seeds=(1,)):
    """One round of adaptive tuning (seed 1, dev set): the first grid of every
    cell, then one step past the edge that holds the best score. The notebook
    calls it until it returns nothing."""
    return [make_cmd(mdl, task, m, 1, lr=lr, tag="tune", **extra)
            for (mdl, task, m, extra, _), _, todo in tuning_status(model)
            for lr in todo]


def plan_rev_core(model, seeds=(1, 2, 3)):
    """Main table, seeds 4-5, the ChemProt ablation (including the tau sweep) and
    the placement ablation on the other tasks, all at the rates now selected.
    Finished runs are skipped, so only cells whose selection moved are rerun."""
    return (plan_main(model, seeds=seeds) + plan_seeds45(model)
            + plan_ablation(model, seeds=seeds) + plan_placement_x(model, seeds=seeds))


def plan_placement_budget(model, seeds=(1, 2, 3)):
    """Budget-matched placement: feed-forward-only LoRA at the all-module budget
    and all-module LoRA at the feed-forward budget, each at its own rate."""
    return [make_cmd(model, task, "lora", seed, target=tgt, budget_rank=r)
            for seed in seeds for task in TASKS for tgt, r in (("ffn", 14), ("all", 4))]


def plan_gev(model, seeds=(1, 2, 3)):
    return [make_cmd(model, task, "gev", seed) for seed in seeds for task in TASKS]


def plan_refctl(model, seeds=(1, 2, 3)):
    """DRIFT with its reference replaced by a second general-domain corpus (news),
    by word-shuffled WikiText, or by uniformly random tokens; DRIFT's own rate."""
    cells = [("chemprot", "news"), ("chemprot", "shuffled"), ("chemprot", "random"),
             ("hoc", "news"), ("hoc", "random")]
    return [make_cmd(model, task, "drift", seed, ref=ref)
            for seed in seeds for task, ref in cells]


def plan_refctl45(model, seeds=(4, 5)):
    """Seeds 4-5 of the reference controls, so that they can be compared with
    LoRA and DRIFT over the same five seeds (the word-shuffled reference's gain
    over LoRA on ChemProt rests on three nearly identical paired differences)."""
    cells = [("chemprot", "news"), ("chemprot", "shuffled"), ("chemprot", "random"),
             ("hoc", "news"), ("hoc", "random")]
    return [make_cmd(model, task, "drift", seed, ref=ref, tag="seeds45")
            for task, ref in cells for seed in seeds]


def plan_eva_units(model, seeds=(1, 2, 3)):
    """EVA with its own allocation rule (a budget of rank units, so FFN rank is as
    cheap as attention rank), at EVA's rate."""
    return [make_cmd(model, task, "eva", seed, alloc_mode="units")
            for seed in seeds for task in ("hoc", "chemprot")]


def plan_ladder_tuned(model=None, seeds=(1, 2, 3), task="chemprot"):
    """The ladder with every method at the rate tuned on that backbone."""
    return [make_cmd(bb, task, m, seed) for seed in seeds for bb in LADDER
            for m in SWEEP_METHODS]


def plan_tiny_prof(model=None, seeds=(1, 2, 3), task="chemprot"):
    """BERT-Tiny profiled from every ChemProt training sentence and every WikiText
    passage (counts above what exists take everything), at BERT-Tiny's rates:
    is the small-backbone shortfall estimation noise in the profile?"""
    return [make_cmd(LADDER[0], task, m, seed, n_ref=8192, n_dom=8192)
            for seed in seeds for m in ("eva", "eva_white", "drift")]


def plan_budget_tuned(model, seeds=(1, 2, 3), task="chemprot"):
    return [make_cmd(model, task, m, seed, budget_rank=r)
            for seed in seeds for r in SWEEP_RANKS for m in SWEEP_METHODS]


def plan_eva_lowlr(model, seeds=(1, 2, 3, 4, 5)):
    """EVA on HoC one grid step below its selected rate, five seeds: does a
    smaller global step stabilise it the way whitening does? Tagged, so it never
    enters the main table."""
    return [make_cmd(model, "hoc", "eva", seed, lr=1e-4, tag="lowlr")
            for seed in seeds]


FP32_METHODS = ("lora", "dora", "pissa", "eva", "eva_white", "drift", "gev")


def plan_fp32_hoc(model, seeds=(1, 2, 3)):
    """HoC in fp32 with deterministic kernels at the rates tuned in fp16: are the
    divergences a property of the method or of fp16 numerics? DRIFT's seed 1 is
    run twice to check that the runs are now reproducible."""
    out = [make_cmd(model, "hoc", m, seed, amp=False, deterministic=True, tag="fp32")
           for seed in seeds for m in FP32_METHODS]
    out.append(make_cmd(model, "hoc", "drift", 1, amp=False, deterministic=True,
                        tag="fp32rep"))
    return out


def _tune_round(model, cells):
    return [make_cmd(mdl, task, m, 1, lr=lr, tag="tune", **extra)
            for (mdl, task, m, extra, _), _, todo in tuning_status(model, cells)
            for lr in todo]


def plan_tune_rev2(model, seeds=(1,)):
    """One round of adaptive tuning for every configuration added after the
    audit (see tuning_cells_rev2); called until it returns nothing."""
    return _tune_round(model, tuning_cells_rev2(model))


def plan_tune_rev2a(model, seeds=(1,)):
    """... the inexpensive part: rsLoRA, budgets elsewhere, linear head."""
    return _tune_round(model, tuning_cells_rev2(model, "a"))


def plan_tune_rev2b(model, seeds=(1,)):
    """... the decoder on HoC."""
    return _tune_round(model, tuning_cells_rev2(model, "b"))


CLIN = "mtsamples"
CLIN_METHODS = ("lora", "eva", "eva_white", "drift")


def tuning_cells_clinical(model):
    """The clinical-text task: the four methods the analysis turns on."""
    return [(model, CLIN, m, {}, [3e-5, 1e-4, 3e-4] if m == "eva" else [1e-4, 3e-4, 1e-3])
            for m in CLIN_METHODS]


def plan_tune_clin(model, seeds=(1,)):
    return _tune_round(model, tuning_cells_clinical(model))


def plan_clinical(model, seeds=(1, 2, 3)):
    """Clinical notes (R3 W3): the four methods at their tuned rates, and DRIFT
    with its reference replaced by news text and by random tokens, since the
    reference-corpus choice is what clinical text might change."""
    out = [make_cmd(model, CLIN, m, seed) for seed in seeds for m in CLIN_METHODS]
    out += [make_cmd(model, CLIN, "drift", seed, ref=ref)
            for seed in seeds for ref in ("news", "random")]
    return out


def plan_eva_fp32_lr_low(model, seeds=(1, 2, 3)):
    """plan_eva_fp32_lr without EVA's selected rate, which plan_fp32_hoc runs."""
    return [make_cmd(model, "hoc", "eva", seed, lr=lr, amp=False, deterministic=True,
                     tag="fp32") for seed in seeds for lr in (1e-4, 3e-5)]


def plan_decoder_hoc(model, seeds=(1, 2, 3)):
    """The decoder SLM on HoC, where the instability lives (EIC W5)."""
    return [make_cmd(DECODER, "hoc", m, seed, batch_size=DECODER_HOC_BATCH)
            for seed in seeds for m in DECODER_METHODS]


PREDS_METHODS = ("lora", "drift", "eva", "eva_white", "bitfit", "dora", "pissa",
                 "adalora", "gev")


def plan_r2_fixes(model, seeds=(1, 2, 3)):
    """Round-2 re-review, NEW-3: the one uninformative cell of the placement table,
    feed-forward-only rank 14 on HoC, where two of three seeds collapse at the
    selected rate. Both remedies the reviewer offers: rerun it in fp32 at the same
    rate, and add the seeds needed to select its rate by the mean dev score over
    three seeds instead of one."""
    cfg = dict(target="ffn", budget_rank=14)
    sel = resolve_lr(model, "hoc", "lora", **cfg)
    out = [make_cmd(model, "hoc", "lora", s, amp=False, deterministic=True, tag="fp32", **cfg)
           for s in seeds]
    out += [make_cmd(model, "hoc", "lora", s, lr=3e-4, tag="ffn14lr", **cfg)
            for s in seeds if not _same(sel, 3e-4)]
    return out


def plan_preds(model, seeds=(1, 2, 3)):
    """Table I once more, every run storing its per-example test predictions, for
    instance-level bootstrap intervals (R1 W3): seeds 1-3 of every
    parameter-efficient method (the comparisons with LoRA) and seeds 4-5 of the
    four five-seed methods. Tagged, so the table keeps its runs; the replicates
    also measure run-to-run reproducibility. Task by task, so that a session
    that stops at its deadline leaves complete cells behind."""
    out = []
    for task in TASKS:
        out += [make_cmd(model, task, m, seed, tag="preds")
                for m in PREDS_METHODS for seed in seeds]
        out += [make_cmd(model, task, m, seed, tag="preds")
                for m in ("lora", "eva", "eva_white", "drift") for seed in (4, 5)]
    return out


def plan_rslora(model, seeds=(1, 2, 3), task="chemprot"):
    """LoRA with rsLoRA's scale alpha/sqrt(r) at every budget, tuned per rank
    (R2 W2, Q3)."""
    return [make_cmd(model, task, "lora", seed, budget_rank=r, scaling="rslora")
            for seed in seeds for r in [1, 2, 4, 8, 16]]


def plan_budget_x(model, seeds=(1, 2, 3)):
    """Budgets of 0.5% (rank 4) and 2% (rank 16) on RCT-20k and HoC, every
    (method, budget) tuned (the devil's advocate's unexamined premise)."""
    return [make_cmd(model, task, m, seed, budget_rank=r)
            for seed in seeds for task in ("rct20k", "hoc") for r in (4, 16)
            for m in SWEEP_METHODS]


def plan_fp32_rest(model, seeds=(1, 2, 3)):
    """The rest of the HoC column in deterministic fp32 (R1 W2a): full
    fine-tuning, linear probing, BitFit and AdaLoRA."""
    return [make_cmd(model, "hoc", m, seed, amp=False, deterministic=True, tag="fp32")
            for seed in seeds for m in ("adalora", "bitfit", "full", "linear")]


def plan_eva_fp32_lr(model, seeds=(1, 2, 3)):
    """EVA on HoC in fp32 at one and two grid steps below its selected rate,
    three seeds each, so its rate can be selected by the mean dev score over
    seeds rather than by seed 1 (the devil's advocate's numerics test)."""
    # 3e-4 is EVA's fp16 selection; if it still is, that run is fp32_hoc's and
    # the planner skips it here
    return [make_cmd(model, "hoc", "eva", seed, lr=lr, amp=False, deterministic=True,
                     tag="fp32") for seed in seeds for lr in (3e-4, 1e-4, 3e-5)]


def plan_linhead(model, seeds=(1, 2, 3), task="chemprot"):
    """Every compared method with a single linear classification head (R1 W4)."""
    return [make_cmd(model, task, m, seed, head="linear")
            for seed in seeds for m in LINHEAD_METHODS]


def plan_factor_x(model, seeds=(1, 2, 3)):
    """The allocation/initialisation factorisation on RCT-20k and HoC, at
    DRIFT's rate on each task."""
    out = []
    for seed in seeds:
        for task in ("rct20k", "hoc"):
            out.append(make_cmd(model, task, "drift", seed, init_mode="random",
                                tag="allocOnly"))
            out.append(make_cmd(model, task, "drift", seed, alloc_mode="uniform",
                                tag="initOnly"))
    return out


PLANS = {"tune": plan_tune, "main": plan_main, "ladder": plan_ladder,
         "ladder_lr": plan_ladder_lr, "budget": plan_budget,
         "ablation": plan_ablation, "placement_x": plan_placement_x,
         "seeds45": plan_seeds45, "decoder_tune": plan_decoder_tune,
         "decoder_tune_ext": plan_decoder_tune_ext, "decoder_main": plan_decoder_main,
         "tune_rev": plan_tune_rev, "rev_core": plan_rev_core,
         "placement_budget": plan_placement_budget, "gev": plan_gev,
         "refctl": plan_refctl, "refctl45": plan_refctl45, "eva_units": plan_eva_units,
         "ladder_tuned": plan_ladder_tuned, "tiny_prof": plan_tiny_prof,
         "budget_tuned": plan_budget_tuned, "fp32_hoc": plan_fp32_hoc,
         "eva_lowlr": plan_eva_lowlr, "tune_rev2": plan_tune_rev2,
         "tune_rev2a": plan_tune_rev2a, "tune_rev2b": plan_tune_rev2b,
         "tune_clin": plan_tune_clin, "clinical": plan_clinical,
         "eva_fp32_lr_low": plan_eva_fp32_lr_low,
         "decoder_hoc": plan_decoder_hoc, "preds": plan_preds, "rslora": plan_rslora,
         "budget_x": plan_budget_x, "fp32_rest": plan_fp32_rest,
         "eva_fp32_lr": plan_eva_fp32_lr, "linhead": plan_linhead,
         "factor_x": plan_factor_x, "r2_fixes": plan_r2_fixes}
MODEL_ONLY = ("tune", "decoder_tune", "decoder_tune_ext", "tune_rev", "tune_rev2",
              "tune_rev2a", "tune_rev2b", "tune_clin")
SEEDS_ONLY = ("ladder", "ladder_lr")


# Wall-clock caps on every child process. A stalled checkpoint download once hung
# a profiling step for 5.6 h until the platform killed the session; with a cap the
# step fails, the run that needed it fails fast, and everything else proceeds.
PROFILE_TIMEOUT_S = 30 * 60
RUN_TIMEOUT_S = 60 * 60


def _run_capped(cmd, timeout):
    try:
        return subprocess.run(cmd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"  !! timeout after {timeout/60:.0f} min: {' '.join(cmd[2:8])}",
              flush=True)
        return -9


PROFILE_DIR = os.path.join(ROOT, "runs", "profiles")
RESULT_DIR = os.path.join(ROOT, "runs", "results")


def profile_needs(cmds):
    """(model, task, ref, n_ref, n_dom, gev) of every profile the runs load."""
    need = set()
    for c in cmds:
        d = cmd_args(c)
        if d.get("--method") in NEEDS_PROFILE:
            need.add((d["--model"], d["--task"], d.get("--ref", "wikitext"),
                      int(d.get("--n_ref", 1024)), int(d.get("--n_dom", 1024)),
                      d.get("--method") == "gev"))
    return need


def ensure_profiles(cmds, deadline=0):
    """Compute any DRIFT profile a queued run will need (existing ones are kept:
    runs extending earlier ones must see the identical initialisation)."""
    import runspec
    for model, task, ref, n_ref, n_dom, gev in sorted(profile_needs(cmds)):
        key = runspec.profile_key(model, task, n_ref, n_dom, True, ref, gev)
        if os.path.exists(os.path.join(PROFILE_DIR, key + ".pt")):
            continue
        if deadline and time.time() > deadline:
            print("DEADLINE reached: skipping remaining profiles", flush=True)
            return
        cmd = [PY, PROF, "--model", model, "--task", task]
        if ref != "wikitext":
            cmd += ["--ref", ref]
        if (n_ref, n_dom) != (1024, 1024):
            cmd += ["--n_ref", str(n_ref), "--n_dom", str(n_dom)]
        if gev:
            # the GEV runs read only the generalised eigenvectors and the module list
            cmd += ["--gev", "--taus", "0.0"]
        print(f"[profile] {key}", flush=True)
        _run_capped(cmd, PROFILE_TIMEOUT_S)


def pending(cmds):
    """Drop commands whose result file already exists, and repeats of one run
    (two tuning cells can share a configuration, e.g. all-module LoRA at rank 4 is
    both a placement and a budget cell), before sharding, so the GPUs split only
    the work that is left and never run the same configuration twice."""
    import runspec
    out, seen = [], set()
    for c in cmds:
        rid = runspec.run_id(runspec.parse(c[2:]))
        if rid in seen or os.path.exists(os.path.join(RESULT_DIR, rid + ".json")):
            continue
        seen.add(rid)
        out.append(c)
    return out


def build_plan(plan, model, seeds):
    fn = PLANS[plan]
    if plan in MODEL_ONLY:
        return fn(model)
    if plan in SEEDS_ONLY:
        return fn(seeds=seeds)
    return fn(model, seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=list(PLANS))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--count", action="store_true",
                    help="print only the number of runs still to do")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1",
                    help="k/n: run every n-th command starting at k (one per GPU)")
    ap.add_argument("--deadline", type=float, default=0,
                    help="unix time after which no new run is started")
    ap.add_argument("--profiles_only", action="store_true")
    ap.add_argument("--no_profiles", action="store_true")
    a = ap.parse_args()

    seeds = tuple(int(s) for s in a.seeds.split(","))
    cmds = pending(build_plan(a.plan, a.model, seeds))
    if a.count:
        print(f"PENDING {len(cmds)}")
        return
    if a.limit:
        cmds = cmds[:a.limit]
    k, n = (int(x) for x in a.shard.split("/"))
    cmds = cmds[k::n]

    print(f"plan={a.plan} model={a.model} shard={k}/{n} runs={len(cmds)}")
    if a.dry:
        for c in cmds[:400]:
            print(" ", " ".join(c[2:]))
        return

    if not a.no_profiles:
        ensure_profiles(cmds, a.deadline)
    if a.profiles_only:
        return

    t0 = time.perf_counter()
    done = 0
    for i, c in enumerate(cmds, 1):
        if a.deadline and time.time() > a.deadline:
            print(f"\nDEADLINE reached: stopping before run {i}/{len(cmds)}; "
                  "re-run next session to continue", flush=True)
            return
        print(f"\n[{i}/{len(cmds)}] {' '.join(c[3:])}", flush=True)
        rc = _run_capped(c, RUN_TIMEOUT_S)
        if rc != 0:
            print(f"  !! exit {rc}", flush=True)
        done += 1
        el = time.perf_counter() - t0
        print(f"  elapsed {el/60:.1f} min | avg {el/done/60:.2f} min/run | "
              f"eta {(len(cmds)-done)*el/done/3600:.2f} h", flush=True)
    print("GRID COMPLETE")


if __name__ == "__main__":
    main()
