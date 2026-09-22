"""Generate the paper figures from cached profiles and result JSONs."""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import torch                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
PROFILES = os.path.join(ROOT, "runs", "profiles")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "figure.dpi": 200,
})

COL = {"q": "#3B6EA8", "k": "#6BA3D6", "v": "#C44E52", "o": "#E0885A",
       "ffn_up": "#55A868", "ffn_down": "#8172B2"}
LBL = {"q": "$W_Q$", "k": "$W_K$", "v": "$W_V$", "o": "$W_O$",
       "ffn_up": "FFN up", "ffn_down": "FFN down"}
ORDER = ["q", "k", "v", "o", "ffn_up", "ffn_down"]
TASK_PRETTY = {"chemprot": "ChemProt", "rct20k": "RCT-20k", "hoc": "HoC"}


def classify(name):
    n = name.lower()
    if "query" in n or "q_proj" in n:
        return "q"
    if "key" in n or "k_proj" in n:
        return "k"
    if "value" in n or "v_proj" in n:
        return "v"
    if "attention.output.dense" in n or "out_proj" in n or "o_proj" in n:
        return "o"
    # Llama-style decoders: gate and up projections read the same input
    if "intermediate.dense" in n or "fc1" in n or "gate_proj" in n or "up_proj" in n:
        return "ffn_up"
    return "ffn_down"


# W_Q, W_K and W_V are all applied to the same block input, so they share one
# second-moment matrix: input-side profiling sees four distinct sites per block,
# not six. Grouping the drift plot this way avoids drawing three identical lines.
SITE_OF = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "attn_out",
           "ffn_up": "ffn_in", "ffn_down": "ffn_hidden"}
SITE_ORDER = ["attn_in", "attn_out", "ffn_in", "ffn_hidden"]
SITE_LBL = {"attn_in": "block input ($W_Q,W_K,W_V$)",
            "attn_out": "attention output ($W_O$)",
            "ffn_in": "FFN input (up-proj)",
            "ffn_hidden": "FFN hidden (down-proj)"}
SITE_COL = {"attn_in": "#3B6EA8", "attn_out": "#C44E52",
            "ffn_in": "#55A868", "ffn_hidden": "#8172B2"}


def layer_of(name):
    m = re.search(r"layers?\.(\d+)\.", name)      # encoder "layer.N.", decoder "layers.N."
    return int(m.group(1)) if m else -1


def load_profile(model, task, n_ref=1024, n_dom=1024):
    # centred profiles only: the key must match profile_drift.profile_key
    key = f"{model.replace('/', '__')}__{task}__ref{n_ref}__dom{n_dom}__cen"
    p = os.path.join(PROFILES, key + ".pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=False)


# --------------------------------------------------------------------------
def fig_drift_profile(model, tasks, tau=0.95, out="fig_drift_profile.pdf"):
    """Per-layer share of target activation energy outside the reference subspace."""
    profs = [(t, load_profile(model, t)) for t in tasks]
    profs = [(t, p) for t, p in profs if p is not None]
    if not profs:
        print("no profiles for drift-profile figure")
        return
    fig, axes = plt.subplots(1, len(profs), figsize=(3.4 * len(profs), 2.0),
                             sharey=True, squeeze=False)
    for ax, (task, prof) in zip(axes[0], profs):
        key = str(tau) if str(tau) in prof["taus"] else list(prof["taus"])[0]
        per = prof["taus"][key]
        series = defaultdict(dict)
        for name, d in per.items():
            ratio = d["trace_drift"] / max(d["trace_d"], 1e-12)
            series[SITE_OF[classify(name)]][layer_of(name)] = ratio
        for kind in SITE_ORDER:
            if kind not in series:
                continue
            xs = sorted(series[kind])
            ys = [series[kind][x] for x in xs]
            ax.plot(xs, ys, marker="o", color=SITE_COL[kind], label=SITE_LBL[kind])
        ax.set_title(TASK_PRETTY.get(task, task))
        ax.set_xlabel("transformer block")
        blocks = sorted({layer_of(n) for n in per})
        ax.set_xticks(blocks)
        if len(blocks) > 8:
            ax.set_xticklabels([str(b) if b % 2 == 0 else "" for b in blocks])
    axes[0][0].set_ylabel(r"drift ratio $\rho_m$")
    axes[0][0].legend(frameon=False, loc="best", handlelength=1.4)
    fig.tight_layout()
    path = os.path.join(FIGS, out)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_rank_alloc(model, task, budget_rank=8, rho=2.0, out="fig_rank_alloc.pdf"):
    """Rank allocated per module by DRIFT vs EVA, against the uniform baseline."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import drift as drift_mod

    prof = load_profile(model, task)
    if prof is None:
        print("no profile for rank-alloc figure")
        return
    costs = prof["meta"]["costs"]
    budget = budget_rank * sum(costs.values())
    r_cap = int(rho * budget_rank)

    allocs = {}
    for label, tau in [("DRIFT ($\\tau$=0.95)", 0.95), ("EVA ($\\tau$=0)", 0.0)]:
        key = str(tau) if str(tau) in prof["taus"] else list(prof["taus"])[0]
        per = prof["taus"][key]
        spectra = {n: per[n]["evals"] for n in per}
        norms = {n: per[n]["trace_d"] for n in per}
        ranks, _ = drift_mod.allocate_ranks(spectra, costs, budget, r_min=0,
                                            r_max=r_cap, score_mode="relative",
                                            norms=norms)
        allocs[label] = ranks

    layers = sorted({layer_of(n) for n in costs})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    for ax, (label, ranks) in zip(axes, allocs.items()):
        grid = np.zeros((len(ORDER), len(layers)))
        for n, r in ranks.items():
            grid[ORDER.index(classify(n)), layers.index(layer_of(n))] = r
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=r_cap)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LBL[k] for k in ORDER])
        ax.set_xlabel("transformer block")
        ax.set_title(label)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("allocated rank")
    cb.ax.axhline(budget_rank, color="w", lw=1.0)
    path = os.path.join(FIGS, out)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    for label, ranks in allocs.items():
        v = np.array(list(ranks.values()))
        print(f"  {label}: mean {v.mean():.2f} min {v.min()} max {v.max()} "
              f"frac at cap {np.mean(v == r_cap):.2f} frac<=1 {np.mean(v <= 1):.2f}")


# --------------------------------------------------------------------------
# result-driven figures
# --------------------------------------------------------------------------
def _rows():
    return _analyze().load_all()


def _analyze():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import analyze
    return analyze


def _series(rows, keyfn, filt):
    """{x: (mean, std, n)} over seeds."""
    g = defaultdict(list)
    for r in rows:
        if not filt(r) or r["score"] is None:
            continue
        g[keyfn(r)].append(r["score"])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                len(v)) for k, v in g.items()}


METHOD_STYLE = {"drift": ("#C44E52", "o", r"\textsc{Drift}"),
                "lora": ("#3B6EA8", "s", "LoRA"),
                "eva": ("#55A868", "^", "EVA"),
                "eva_white": ("#DD8452", "D", "EVA (whitened)")}
# AdaLoRA is deliberately absent: it is not part of the budget or ladder sweeps,
# and a lone rank-8 point would read as a sweep result.


BUDGET_RANKS = (1, 2, 4, 8, 16)


def budget_series(model, task="chemprot"):
    """{method: {rank: (mean, sd, n)}}, every (method, budget) at the learning
    rate selected for it (seeds 1-3)."""
    an = _analyze()
    rows = _rows()
    mdl = model.replace("/", "__")
    out = {}
    for meth in list(METHOD_STYLE) + ["lora_rs"]:
        s = {}
        for r in BUDGET_RANKS:
            if meth == "lora_rs":     # LoRA with rsLoRA's scale alpha/sqrt(r)
                rs = an.pick(rows, mdl, task, "lora", budget_rank=r, scaling="rslora")
            else:
                rs = an.pick(rows, mdl, task, meth, budget_rank=r)
            mu, sd, n, _ = an.cell(rs)
            if n:
                s[r] = (mu, sd, n)
        out[meth] = s
    return out


def fig_budget(model, task="chemprot", out="fig_budget.pdf"):
    S = budget_series(model, task)
    if not any(S.values()):
        print("no rows for budget figure")
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    styles = dict(METHOD_STYLE)
    styles["lora_rs"] = ("#3B6EA8", "s", r"LoRA, $\alpha/\sqrt{r}$")
    for meth, (col, mk, lbl) in styles.items():
        s = S.get(meth)
        if not s:
            continue
        xs = sorted(s)
        ax.errorbar(xs, [s[x][0] * 100 for x in xs], yerr=[s[x][1] * 100 for x in xs],
                    marker=mk, color=col,
                    mfc="white" if meth in ("eva", "eva_white", "lora_rs") else col,
                    ls="--" if meth == "lora_rs" else "-",
                    label=lbl.replace("\\textsc{Drift}", "DRIFT"), capsize=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGET_RANKS)
    ax.set_xticklabels([str(r) for r in BUDGET_RANKS])
    ax.minorticks_off()
    ax.set_xlabel("uniform-equivalent rank (adapter budget)")
    ax.set_ylabel("test F1 (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGS, out)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_ladder(task="chemprot", out="fig_ladder.pdf"):
    analyze = _analyze()
    # the ladder is the BERT miniature family only: one vocabulary and one
    # pretraining recipe, so scale is the only variable
    rows = [r for r in _rows() if r["task"] == task and analyze.is_default(r)
            and r["model"].startswith("google__bert_uncased_")]
    if not rows:
        print("no rows for ladder figure")
        return
    sizes = {}
    for r in rows:
        pretty = analyze.MODEL_PRETTY.get(r["model"])
        if pretty and pretty in analyze.MODEL_PARAMS:
            sizes[r["model"]] = analyze.MODEL_PARAMS[pretty]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax = axes[0]
    for meth, (col, mk, lbl) in METHOD_STYLE.items():
        s = _series(rows, lambda r: sizes.get(r["model"]),
                    lambda r, m=meth: r["method"] == m and r["model"] in sizes)
        s.pop(None, None)
        if not s:
            continue
        xs = sorted(s)
        ax.errorbar(xs, [s[x][0] * 100 for x in xs],
                    yerr=[s[x][1] * 100 for x in xs], marker=mk, color=col,
                    label=lbl.replace("\\textsc{Drift}", "DRIFT"), capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("backbone parameters (M)")
    ax.set_ylabel("test F1 (%)")
    ax.legend(frameon=False, loc="lower right")

    ax2 = axes[1]
    d = _series(rows, lambda r: sizes.get(r["model"]),
                lambda r: r["method"] == "drift" and r["model"] in sizes)
    l = _series(rows, lambda r: sizes.get(r["model"]),
                lambda r: r["method"] == "lora" and r["model"] in sizes)
    e = _series(rows, lambda r: sizes.get(r["model"]),
                lambda r: r["method"] == "eva" and r["model"] in sizes)
    w = _series(rows, lambda r: sizes.get(r["model"]),
                lambda r: r["method"] == "eva_white" and r["model"] in sizes)
    for ref, col, lbl in [(l, "#3B6EA8", "vs. LoRA"), (e, "#55A868", "vs. EVA"),
                          (w, "#DD8452", "vs. EVA (whitened)")]:
        xs = sorted(set(d) & set(ref) - {None})
        if not xs:
            continue
        ax2.plot(xs, [(d[x][0] - ref[x][0]) * 100 for x in xs],
                 marker="o", color=col, label=lbl)
    ax2.axhline(0, color="k", lw=0.6, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("backbone parameters (M)")
    ax2.set_ylabel("F1 gain of DRIFT (pts)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGS, out)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


OUTLIER_DIMS = (77, 588)   # RoBERTa-base (Kovaleva et al. 2021; Puccetti et al. 2022)


def outlier_stats(model, task, tau="0.95"):
    """Per input site: energy of the leading EVA (tau=0) and DRIFT direction
    relative to an average direction, and whether EVA's leading direction peaks
    on an outlier dimension of the hidden state."""
    p = load_profile(model, task)
    if p is None:
        return None
    eva, dr, dims = p["taus"]["0.0"], p["taus"][tau], p["meta"]["dims"]
    out = []
    for n in eva:
        d_in = dims[n][0]
        lam_bar = eva[n]["trace_d"] / d_in
        u = np.abs(np.asarray(eva[n]["basis"])[:, 0])
        out.append({"name": n, "layer": layer_of(n), "kind": classify(n),
                    "eva": float(eva[n]["evals"][0]) / lam_bar,
                    "drift": float(dr[n]["evals"][0]) / lam_bar,
                    "hit": d_in == 768 and int(u.argmax()) in OUTLIER_DIMS})
    return out


def gev_lead(model, task):
    """{module: leading-direction summary} of the GEV initialisation, from the
    GEV profile's JSON summary (written by profile_drift.py --gev)."""
    key = f"{model.replace('/', '__')}__{task}__ref1024__dom1024__cen__gev"
    p = os.path.join(PROFILES, key + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("lead", {}).get("gev")


def fig_outliers(model, tasks, out="fig_outliers.pdf"):
    """Energy of each site's leading initial direction, EVA vs DRIFT (first task);
    prints the summary statistics quoted in the paper for every task."""
    for task in tasks:
        s = outlier_stats(model, task)
        if s:
            e = [r["eva"] for r in s]
            d = [r["drift"] for r in s]
            print(f"{task}: modules on dims {OUTLIER_DIMS}: {sum(r['hit'] for r in s)}/{len(s)}"
                  f" | EVA median {np.median(e):.0f}x max {max(e):.0f}x"
                  f" | DRIFT median {np.median(d):.1f}x max {max(d):.1f}x")
    s = outlier_stats(model, tasks[0])
    if not s:
        print("no profile for outlier figure")
        return
    # W_Q, W_K and W_V read the same input: plot one point per input site
    seen, pts = set(), []
    for r in sorted(s, key=lambda r: (r["layer"], ORDER.index(r["kind"]))):
        site = (r["layer"], SITE_OF[r["kind"]])
        if site not in seen:
            seen.add(site)
            pts.append(r)
    x = np.arange(len(pts))
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    hit = np.array([r["hit"] for r in pts])
    ev = np.array([r["eva"] for r in pts])
    dv = np.array([r["drift"] for r in pts])
    # marker shape and fill carry the distinction, so it survives greyscale
    ax.scatter(x[hit], ev[hit], s=12, color="#C44E52", zorder=3,
               label="EVA, peak on dim 77/588")
    ax.scatter(x[~hit], ev[~hit], s=12, facecolors="none", edgecolors="#C44E52",
               linewidths=0.7, zorder=3, label="EVA, other")
    ax.scatter(x, dv, s=16, marker="^", facecolors="none", edgecolors="#3B6EA8",
               linewidths=0.8, zorder=3, label=r"DRIFT ($\tau{=}0.95$)")
    gev = gev_lead(model, tasks[0])
    if gev:
        gv = np.array([gev[r["name"]]["energy_rel"] for r in pts])
        ax.scatter(x, gv, s=14, marker="x", color="#333333", linewidths=0.7,
                   zorder=3, label="GEV")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_yscale("log")
    per = len(SITE_ORDER)
    ax.set_xticks(np.arange(0, len(pts), per) + (per - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(len(pts) // per)])
    ax.set_xlabel("transformer block (4 input sites each)")
    ax.set_ylabel("energy / average")
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2)
    fig.tight_layout()
    p = os.path.join(FIGS, out)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


TAUS = (0.0, 0.5, 0.9, 0.95, 0.99)


def tau_series(model, task="chemprot", value="score"):
    """{'fixed': {tau: cell}, 'tuned': {tau: cell}}: every deflation level at
    DRIFT's rate, and every level at the rate selected for it (tau = 0 is EVA
    exactly and takes EVA's rate). value='dev' gives the best dev scores."""
    an = _analyze()
    rows = _rows()
    mdl = model.replace("/", "__")
    d_lr = an.lr_for(mdl, task, "drift")
    out = {"fixed": {}, "tuned": {}}
    for t in TAUS:
        for kind, lr in (("fixed", d_lr), ("tuned", "tuned")):
            mu, sd, n, _ = an.cell(an.pick(rows, mdl, task, "drift", lr=lr, tau=t), value)
            if n:
                out[kind][t] = (mu, sd, n)
    return out


def fig_tau(model, task="chemprot", out="fig_tau.pdf"):
    panels = [("score", "test F1 (%)"), ("dev", "dev F1 (%)")]
    S = {v: tau_series(model, task, v) for v, _ in panels}
    if len(S["score"]["fixed"]) < 2:
        print("no rows for tau figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=False)
    for ax, (value, ylabel) in zip(axes, panels):
        for kind, col, mk, lbl, dx in (("fixed", "#C44E52", "o", "DRIFT's rate", -0.008),
                                       ("tuned", "#3B6EA8", "s", "own rate", 0.008)):
            s = S[value][kind]
            if len(s) < 2:
                continue
            xs = sorted(s)
            ax.errorbar([x + dx for x in xs], [s[x][0] * 100 for x in xs],
                        yerr=[s[x][1] * 100 for x in xs], marker=mk, color=col,
                        mfc="white" if kind == "tuned" else col, capsize=1.5,
                        markersize=3, lw=1.0, label=lbl)
        ax.axvline(0.0, color="#55A868", lw=0.8, ls=":")
        ax.set_xlabel(r"deflation $\tau$")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.2)
    fig.tight_layout()
    p = os.path.join(FIGS, out)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--tasks", default="chemprot,rct20k,hoc")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    tasks = a.tasks.split(",")
    # the paper uses the outlier, tau and budget figures; the others on request
    want = (lambda n: a.only == n if a.only else n in ("budget", "tau", "outliers"))
    if want("profile"):
        fig_drift_profile(a.model, tasks)
    if want("alloc"):
        fig_rank_alloc(a.model, tasks[0])
    if want("budget"):
        fig_budget(a.model, tasks[0])
    if want("ladder"):
        fig_ladder(tasks[0])
    if want("tau"):
        fig_tau(a.model, tasks[0])
    if want("outliers"):
        fig_outliers(a.model, tasks)


if __name__ == "__main__":
    main()
