"""Every number the Results prose quotes, computed from the result files.

Built on analyze.py (so it sees exactly what the tables see); verify_results.py
re-derives the tables independently. Prints a plain report; missing experiments
show as n/a.

    python src/paper_numbers.py > paper/numbers.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an      # noqa: E402
import grid               # noqa: E402

RB = "roberta-base"
TASKS = ["chemprot", "rct20k", "hoc"]
ROWS = an.load_all(exclude_tags=("smoke",))
ROWS_NT = [r for r in ROWS if r["tag"] != "tune"]


def c(rs, value="score"):
    return an.cell(rs, value)


def f(cell_):
    mu, sd, n, _ = cell_
    return "n/a" if not n else f"{100*mu:.1f}+-{100*sd:.1f} (n={n})"


def pt(a, b):
    d, p, n = an.paired_test(c(a)[3], c(b)[3])
    return "n/a" if n < 2 else f"{100*d:+.2f} (p={p:.3f}, n={n})"


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- run counts
section("RUN COUNTS")
allruns = [r for r in an.load_all(exclude_tags=("smoke",))]
tune = [r for r in allruns if r["tag"] == "tune"]
print("study runs (excluding smoke):", len(allruns), "| tuning runs:", len(tune))
by_model = {}
for r in allruns:
    by_model[r["model"]] = by_model.get(r["model"], 0) + 1
print("by backbone:", by_model)

# ---------------------------------------------------------------- selections moved
section("LEARNING-RATE SELECTIONS THAT MOVED (main table, first grid vs extended)")
T = grid.load_tuning(refresh=True)
for task in TASKS:
    for m in grid.MAIN_METHODS:
        d = T.get(grid.tune_key(RB, task, m), {})
        first = {lr: v for lr, v in d.items() if any(grid._same(lr, x) for x in grid.MAIN_GRIDS[m])}
        if not first:
            continue
        b0, b1 = grid.best_lr(first), grid.best_lr(d)
        ext = sorted(set(d) - set(first))
        tag = "MOVED" if not grid._same(b0, b1) else "same"
        print(f"  {task:9s} {m:10s} first-grid best {b0:g} -> final {b1:g} [{tag}] "
              f"extension tried: " + ", ".join(f"{lr:g}:{100*d[lr]:.1f}" for lr in ext))

# ---------------------------------------------------------------- RQ1
section("RQ1 MAIN TABLE")
methods = ["full", "linear", "bitfit", "lora", "dora", "pissa", "adalora", "eva",
           "eva_white", "drift", "gev"]
C = an.main_cells(ROWS_NT, RB, methods, TASKS)
st = an.main_stats(ROWS_NT, RB, methods, TASKS)
for t in TASKS:
    print(f"  {t}:")
    for m in methods:
        print(f"    {m:10s} {f(c(C[(t, m)]))}  lr={an.lr_for(RB, t, m):g}  "
              + (f"vs LoRA {st.get(f'{t}:{m}-lora', {}).get('delta', float('nan')):+.2f} "
                 f"p={st.get(f'{t}:{m}-lora', {}).get('p', float('nan')):.3f} "
                 f"holm={st.get(f'{t}:{m}-lora', {}).get('p_holm', float('nan')):.3f}"
                 if m not in ('lora', 'full', 'linear') else ""))
gains = [(v["delta"], k) for k, v in st.items() if isinstance(v, dict) and "delta" in v]
print("  largest gain over LoRA:", max(gains))
print("  smallest Holm-adjusted p:", min(v["p_holm"] for v in st.values()
                                          if isinstance(v, dict) and "p_holm" in v))
print("  uncorrected p<0.05:", sorted((k, round(v["delta"], 2), round(v["p"], 3))
                                     for k, v in st.items()
                                     if isinstance(v, dict) and v.get("p", 1) < 0.05))
print("  failed runs:", {k: v for k, v in st["failed"].items() if v})
# micro-F1 on HoC (BLURB)
for m in ("bitfit", "lora", "full"):
    rs = C[("hoc", m)]
    print(f"  HoC micro-F1 {m}: {f(c(rs, 'micro'))}")
d, p, n = an.paired_test(c(C[("hoc", "bitfit")], "micro")[3], c(C[("hoc", "lora")], "micro")[3])
print(f"  BitFit - LoRA HoC micro: {100*d:+.2f} p={p:.3f} n={n}")
for t in TASKS:
    peft = [(100 * c(C[(t, m)])[0], m) for m in methods if m not in ("full", "linear")]
    peft = [x for x in peft if x[0] == x[0]]
    if t != "hoc":
        print(f"  {t}: PEFT spread {max(peft)[0]-min(peft)[0]:.1f} ({min(peft)[1]}..{max(peft)[1]})")

section("RQ1 HoC NUMERICS")
for m in grid.FP32_METHODS:
    fp16 = C[("hoc", m)]
    fp32 = an.pick(ROWS_NT, RB, "hoc", m, tag="fp32", det=True)
    fl = an.fail_floor(ROWS_NT, RB, "hoc")
    nf32 = sum(r["dev"] < fl for r in fp32)
    nf16 = sum(r["dev"] < fl for r in fp16)
    s32 = c(fp32)
    med = f"{100*np.median(list(s32[3].values())):.1f}" if s32[2] else "n/a"
    print(f"  {m:10s} fp16 {f(c(fp16))} failed {nf16}/{len(fp16)} | fp32 {f(s32)} "
          f"median {med} failed {nf32}/{len(fp32)} | per-seed fp32 "
          + str({s: round(100 * v, 1) for s, v in s32[3].items()}))
    for r in fp32:
        h = r.get("history") or []
        if h:
            print(f"      seed {r['seed']}: best epoch {r['best_epoch']}, dev by epoch "
                  + " ".join(f"{100*e['dev']:.0f}" for e in h)
                  + f" | max grad norm {max(e['grad_norm_max'] for e in h):.2f}")
rep = an.pick(ROWS_NT, RB, "hoc", "drift", tag="fp32rep", det=True)
one = [r for r in an.pick(ROWS_NT, RB, "hoc", "drift", tag="fp32", det=True) if r["seed"] == 1]
if rep and one:
    print("  fp32 repeat of DRIFT seed 1 identical:",
          rep[0]["dev"] == one[0]["dev"] and rep[0]["score"] == one[0]["score"],
          rep[0]["dev"], one[0]["dev"])
low = an.pick(ROWS_NT, RB, "hoc", "eva", lr=1e-4, tag="lowlr")
fl = an.fail_floor(ROWS_NT, RB, "hoc")
print(f"  EVA at 1e-4 on HoC (5 seeds): {f(c(low))} failed {sum(r['dev'] < fl for r in low)}/{len(low)}"
      f" per-seed {dict((s, round(100*v,1)) for s, v in c(low)[3].items())}")
# telemetry of the fp16 failures that have it
for m in ("eva", "drift", "pissa", "dora"):
    for r in C[("hoc", m)]:
        if r.get("history") and r["dev"] < fl:
            print(f"  fp16 failure {m} seed {r['seed']}: skipped steps "
                  f"{sum(e['amp_skipped_steps'] for e in r['history'])}, dev "
                  + " ".join(f"{100*e['dev']:.0f}" for e in r["history"]))

# ---------------------------------------------------------------- RQ2
section("RQ2 CONTROLS AND GEV (test F1)")
for t in ("chemprot", "hoc", "rct20k"):
    d_lr, e_lr = an.lr_for(RB, t, "drift"), an.lr_for(RB, t, "eva")
    base = an.pick(ROWS_NT, RB, t, "drift")
    print(f"  {t}: DRIFT {f(c(base))} (lr {d_lr:g})")
    for ref in ("news", "shuffled", "random"):
        rs = an.pick(ROWS_NT, RB, t, "drift", lr=d_lr, ref=ref)
        if rs:
            print(f"    ref={ref:9s} {f(c(rs))}  vs DRIFT {pt(rs, base)}")
    g = an.pick(ROWS_NT, RB, t, "gev")
    if g:
        lora = an.pick(ROWS_NT, RB, t, "lora")
        print(f"    GEV {f(c(g))} lr {an.lr_for(RB, t, 'gev'):g} vs LoRA {pt(g, lora)} "
              f"vs DRIFT {pt(g, base)}")
    eu = an.pick(ROWS_NT, RB, t, "eva", lr=e_lr, alloc_mode="units")
    if eu:
        ev = an.pick(ROWS_NT, RB, t, "eva")
        pr = c(eu, "adapter_params")
        print(f"    EVA rank-unit {f(c(eu))} params {pr[0]/1e6:.2f}M vs EVA budget-matched "
              f"{pt(eu, ev)}; failed {sum(r['dev'] < an.fail_floor(ROWS_NT, RB, t) for r in eu)}/{len(eu)}")

section("RQ2 REFERENCE CONTROLS OVER FIVE SEEDS (seeds 4-5 added)")
S5 = an.SEED_TAGS
for t, refs in (("chemprot", ("news", "shuffled", "random")), ("hoc", ("news", "random"))):
    d_lr = an.lr_for(RB, t, "drift")
    lora5 = an.pick(ROWS_NT, RB, t, "lora", tags=S5)
    drift5 = an.pick(ROWS_NT, RB, t, "drift", tags=S5)
    fl5 = an.fail_floor(ROWS_NT, RB, t)
    comps = []
    for ref in refs:
        rs = an.pick(ROWS_NT, RB, t, "drift", lr=d_lr, ref=ref, tags=S5)
        d, p, n = an.paired_test(c(rs)[3], c(lora5)[3])
        comps.append((ref, p))
        print(f"  {t:8s} ref={ref:9s} {f(c(rs))} seeds {sorted(r['seed'] for r in rs)} "
              f"failed {sum(r['dev'] < fl5 for r in rs)}/{len(rs)} | vs LoRA {pt(rs, lora5)}"
              f" | vs DRIFT {pt(rs, drift5)}")
    adj = an.holm([p if p == p else 1.0 for _, p in comps])
    print(f"    Holm over this task's reference-vs-LoRA comparisons: "
          + ", ".join(f"{r} {a:.3f}" for (r, _), a in zip(comps, adj)))

section("RQ2 PROFILE SUMMARIES (leading directions)")
prof = os.path.join(an.ROOT, "runs", "profiles")
for fn in sorted(os.listdir(prof)):
    if not fn.startswith("roberta-base__") or not fn.endswith(".json"):
        continue
    s = json.load(open(os.path.join(prof, fn), encoding="utf-8"))
    if "lead" not in s:
        continue
    print(" ", fn, "| n_ref", s.get("n_ref_actual"), "| drift ratio",
          {k: round(v, 4) for k, v in s.get("drift_ratio", {}).items()})
    for key, lead in s["lead"].items():
        hid = {n: v for n, v in lead.items() if "attention.self" in n or "intermediate.dense" in n}
        hits = sum(v["argmax"] in (77, 588) for v in hid.values())
        er = np.array([v["energy_rel"] for v in lead.values()])
        w588 = None
        print(f"    [{key}] {hits}/{len(hid)} hidden-state modules peak on 77/588; energy "
              f"median {np.median(er):.1f}x max {er.max():.1f}x")

section("RQ2 TAU SWEEP")
import figures  # noqa: E402
S = figures.tau_series(RB)
for kind in ("fixed", "tuned"):
    print(f"  {kind}: " + " | ".join(f"tau {t:g}: {100*v[0]:.1f}+-{100*v[1]:.1f}"
                                     for t, v in sorted(S[kind].items())))
for t in (0.5, 0.9, 0.99):
    print(f"  tuned rate tau={t}: {an.lr_for(RB, 'chemprot', 'drift', tau=t):g}")

# ---------------------------------------------------------------- RQ3
section("RQ3 ABLATION (ChemProt)")
rows = an.ablation_rows(ROWS_NT, RB, "chemprot")
lora = rows[0][1]
drift = [x for x in rows if x and x[0] == r"\method{} (full)"][0][1]
for item in rows:
    if item is None:
        continue
    name, rs = item
    print(f"  {name:44s} {f(c(rs))} vs LoRA {pt(rs, lora)} vs DRIFT {pt(rs, drift)}")

section("RQ3 PLACEMENT")
for t in TASKS:
    cells = {}
    for item in an.PLACEMENT_ROWS:
        if item is None:
            continue
        name, m, cfg = item
        rs = an.placement_runs(ROWS_NT, RB, t, m, cfg)
        cells[(m, cfg["target"], cfg["budget_rank"])] = rs
        lr = an.lr_for(RB, t, m, target=cfg["target"], budget_rank=cfg["budget_rank"]) \
            if m == "lora" else an.lr_for(RB, t, "drift")
        print(f"  {t:9s} {name:30s} r={cfg['budget_rank']:2d} {f(c(rs))} lr={lr:g} "
              f"params {c(rs, 'adapter_params')[0]/1e6 if rs else float('nan'):.2f}M")
    L = lambda tgt, r: cells.get(("lora", tgt, r), [])   # noqa: E731
    print(f"    ffn8 - all8 {pt(L('ffn', 8), L('all', 8))} | ffn14 - all8 {pt(L('ffn', 14), L('all', 8))}"
          f" | all4 - ffn8 {pt(L('all', 4), L('ffn', 8))} | all4 - all8 {pt(L('all', 4), L('all', 8))}"
          f" | ffn14 - ffn8 {pt(L('ffn', 14), L('ffn', 8))} | attn8 - all8 {pt(L('attn', 8), L('all', 8))}")

section("RQ3 THE FEED-FORWARD r14 HoC CELL: fp32 RERUN AND MULTI-SEED RATE")
_ff = dict(target="ffn", budget_rank=14)
_sel = an.lr_for(RB, "hoc", "lora", **_ff)
_main = an.pick(ROWS_NT, RB, "hoc", "lora", **_ff)
_fp32 = an.pick(ROWS_NT, RB, "hoc", "lora", lr=_sel, tag="fp32", det=True, **_ff)
_low = an.pick(ROWS_NT, RB, "hoc", "lora", lr=3e-4, tag="ffn14lr", **_ff)
_all8 = an.pick(ROWS_NT, RB, "hoc", "lora", target="all", budget_rank=8)
_ffn8 = an.pick(ROWS_NT, RB, "hoc", "lora", target="ffn", budget_rank=8)
_fl = an.fail_floor(ROWS_NT, RB, "hoc")
for nm, rs in (("fp16 @ selected", _main), ("fp32 @ selected", _fp32), ("fp16 @ 3e-4", _low)):
    ok = [r for r in rs if r["dev"] >= _fl]
    print(f"  {nm:16s} {f(c(rs))} trained {len(ok)}/{len(rs)} | test per seed "
          + str({r['seed']: round(100 * r['score'], 1) for r in sorted(rs, key=lambda r: r['seed'])})
          + (f" | trained mean {100*np.mean([r['score'] for r in ok]):.1f}" if ok else ""))
print(f"  selected rate {_sel:g}; mean dev over 3 seeds: "
      f"{100*np.mean([r['dev'] for r in _main]):.1f} at {_sel:g} vs "
      f"{100*np.mean([r['dev'] for r in _low]):.1f} at 0.0003 "
      f"-> a three-seed dev mean selects 0.0003")
print(f"    ffn14@3e-4 - all8 {pt(_low, _all8)} | ffn14@3e-4 - ffn8 {pt(_low, _ffn8)}"
      f" | ffn14 fp32 - all8 {pt(_fp32, _all8)}")

# ---------------------------------------------------------------- RQ4
section("RQ4 BUDGET SWEEP (ChemProt, tuned per rank)")
B = figures.budget_series(RB)


def _blr(m, r):
    """The rate tuned for one series of the budget sweep (lora_rs is LoRA with
    rsLoRA's scale, tuned separately at every rank)."""
    if m == "lora_rs":
        return an.lr_for(RB, "chemprot", "lora", budget_rank=r, scaling="rslora")
    return an.lr_for(RB, "chemprot", m, budget_rank=r)


for m, s in B.items():
    print(f"  {m:10s} " + " | ".join(f"r{r}: {100*v[0]:.1f}+-{100*v[1]:.1f} "
                                    f"(lr {_blr(m, r):g})" for r, v in sorted(s.items())))
for r in figures.BUDGET_RANKS:
    vals = {m: B[m][r][0] for m in B if r in B[m]}
    trio = [vals[m] for m in ("lora", "drift", "eva_white") if m in vals]
    if trio:
        print(f"  r{r}: spread LoRA/DRIFT/whitened {100*(max(trio)-min(trio)):.1f}; "
              f"EVA - LoRA {100*(vals.get('eva', np.nan) - vals.get('lora', np.nan)):+.1f}")
    e = an.pick(ROWS_NT, RB, "chemprot", "eva", budget_rank=r)
    l = an.pick(ROWS_NT, RB, "chemprot", "lora", budget_rank=r)
    print(f"      EVA vs LoRA at r{r}: {pt(e, l)}; EVA best epochs "
          + str(sorted(x['best_epoch'] for x in e)))

section("RQ4 LADDER (ChemProt, tuned per backbone)")
Lc = an.ladder_cells(ROWS_NT)
for mdl in an.LADDER_ORDER:
    name = an.MODEL_PRETTY[mdl]
    lora = Lc[(mdl, "lora")]
    print(f"  {name}:")
    for m in ("lora", "eva", "eva_white", "drift", "eva_rb"):
        lr = an.lr_for(mdl, "chemprot", "eva" if m == "eva_rb" else m) if m != "eva_rb" \
            else an.lr_for(RB, "chemprot", "eva")
        extra = "" if m == "lora" else f" vs LoRA {pt(Lc[(mdl, m)], lora)}"
        print(f"    {m:10s} {f(c(Lc[(mdl, m)]))} lr={lr:g}{extra}")
tiny = an.LADDER_ORDER[0]
print("  BERT-Tiny profiled from all texts:")
for m in ("eva", "eva_white", "drift"):
    big = an.pick(ROWS_NT, tiny, "chemprot", m, n_ref=8192, n_dom=8192)
    std = an.pick(ROWS_NT, tiny, "chemprot", m)
    print(f"    {m:10s} {f(c(big))} vs 1,024-text profile {f(c(std))}: {pt(big, std)}")

# ---------------------------------------------------------------- post-audit
section("fp32 HoC COLUMN (all methods) AND EVA'S fp32 RATE SELECTION")
fl = an.fail_floor(ROWS_NT, RB, "hoc")
for m in methods:
    rs = an.pick(ROWS_NT, RB, "hoc", m, tag="fp32", det=True)
    s = c(rs)
    med = f"{100*np.median(list(s[3].values())):.1f}" if s[2] else "n/a"
    print(f"  {m:10s} fp32 {f(s)} median {med} failed {sum(r['dev'] < fl for r in rs)}/{len(rs)}"
          f" | fp16 {f(c(C[('hoc', m)]))}")
sel = {}
for lr in (3e-5, 1e-4, 3e-4):
    rs = an.pick(ROWS_NT, RB, "hoc", "eva", lr=lr, tag="fp32", det=True)
    if rs:
        sel[lr] = (np.mean([r["dev"] for r in rs]), c(rs),
                   sum(r["dev"] < fl for r in rs), len(rs))
        print(f"  EVA fp32 lr={lr:g}: mean dev {100*sel[lr][0]:.1f} test {f(sel[lr][1])} "
              f"failed {sel[lr][2]}/{sel[lr][3]}")
if sel:
    best = max(sel, key=lambda k: sel[k][0])
    print(f"  -> multi-seed fp32 selection picks {best:g}: test {f(sel[best][1])}")

section("DECODER ON HoC")
for m in ("lora", "eva", "eva_white", "drift"):
    rs = an.pick(ROWS_NT, an.DECODER, "hoc", m)
    dfl = None
    lora = an.pick(ROWS_NT, an.DECODER, "hoc", "lora")
    if lora:
        dfl = float(np.median([r["dev"] for r in lora])) - an.FAIL_MARGIN
    nf = sum(r["dev"] < dfl for r in rs) if dfl is not None else "n/a"
    print(f"  {m:10s} {f(c(rs))} lr={an.lr_for(an.DECODER, 'hoc', m):g} failed {nf}/{len(rs)}"
          + ("" if m == "lora" else f" vs LoRA {pt(rs, lora)}"))

section("INSTANCE-LEVEL BOOTSTRAP (replicate of Table I with predictions)")
cij = os.path.join(an.OUT, "ci.json")
if os.path.exists(cij):
    ci = json.load(open(cij))
    for k, v in sorted(ci.items()):
        d = v.get("delta")
        print(f"  {k:24s} mean {100*v['mean']:.1f} [{100*v['ci'][0]:.1f}, {100*v['ci'][1]:.1f}]"
              + ("" if not d else f" delta {100*d[0]:+.2f} [{100*d[1]:+.2f}, {100*d[2]:+.2f}]")
              + f" rep {100*v['rep_diff']:+.2f} (n={v['n_seeds']})")
    excl = [k for k, v in ci.items() if v.get("delta") and (v["delta"][1] > 0 or v["delta"][2] < 0)]
    print("  intervals of the difference to LoRA that exclude zero:", excl)

section("rsLoRA SCALE (LoRA, ChemProt, tuned per rank)")
for r in figures.BUDGET_RANKS:
    rs = an.pick(ROWS_NT, RB, "chemprot", "lora", budget_rank=r, scaling="rslora")
    base = an.pick(ROWS_NT, RB, "chemprot", "lora", budget_rank=r)
    print(f"  r{r:<2d} alpha/sqrt(r): {f(c(rs))} lr "
          f"{an.lr_for(RB, 'chemprot', 'lora', budget_rank=r, scaling='rslora'):g} | alpha/r "
          f"{f(c(base))} | diff {pt(rs, base)}")

section("BUDGETS 0.5% AND 2% ON RCT-20k AND HoC")
for t in ("rct20k", "hoc"):
    for r in (4, 8, 16):
        lora = an.pick(ROWS_NT, RB, t, "lora", budget_rank=r)
        line = f"  {t:7s} r{r:<2d}"
        for m in ("lora", "eva", "eva_white", "drift"):
            rs = an.pick(ROWS_NT, RB, t, m, budget_rank=r)
            line += f" | {m} {f(c(rs))}" + ("" if m == "lora" else f" {pt(rs, lora)}")
        print(line)

section("LINEAR HEAD (ChemProt)")
for m in ("lora", "eva", "eva_white", "drift", "bitfit"):
    a = an.pick(ROWS_NT, RB, "chemprot", m)
    b = an.pick(ROWS_NT, RB, "chemprot", m, head="linear")
    lb = an.pick(ROWS_NT, RB, "chemprot", "lora", head="linear")
    print(f"  {m:10s} default {f(c(a))} | linear {f(c(b))} lr "
          f"{an.lr_for(RB, 'chemprot', m, head='linear'):g}"
          + ("" if m == "lora" else f" | vs linear-head LoRA {pt(b, lb)}"))

section("FACTORISATION ON RCT-20k AND HoC")
for t in ("rct20k", "hoc"):
    d_lr = an.lr_for(RB, t, "drift")
    lora = an.pick(ROWS_NT, RB, t, "lora")
    for tag, want in (("allocOnly", {"init_mode": "random"}), ("initOnly", {"alloc_mode": "uniform"})):
        rs = an.pick(ROWS_NT, RB, t, "drift", lr=d_lr, tag=tag, **want)
        print(f"  {t:7s} {tag:9s} {f(c(rs))} vs LoRA {pt(rs, lora)}")

section("CLINICAL NOTES (MTSamples)")
CL = an.CLIN
clin = an.clinical_rows(ROWS_NT, RB)
lora_c = dict(clin).get(an.PRETTY["lora"], [])
flc = an.fail_floor(ROWS_NT, RB, CL) if lora_c else None
for label, rs in clin:
    if not rs:
        print(f"  {label:36s} n/a")
        continue
    nf = sum(r["dev"] < flc for r in rs) if flc is not None else "n/a"
    print(f"  {label:36s} micro {f(c(rs))} macro {f(c(rs, 'macro_f1'))} lr {rs[0]['lr']:g} "
          f"failed {nf}/{len(rs)} n_train {rs[0].get('n_train')}"
          + ("" if label == an.PRETTY["lora"] else f" | vs LoRA {pt(rs, lora_c)}")
          + " | per seed " + " ".join(f"{100*r['score']:.1f}" for r in sorted(rs, key=lambda r: r['seed'])))
T_clin = grid.load_tuning(refresh=True)
for m in an.SWEEP:
    d = T_clin.get(grid.tune_key(RB, CL, m), {})
    if d:
        print(f"  tuning {m:10s} " + " ".join(f"{lr:g}:{100*v:.1f}" for lr, v in sorted(d.items())))
for fn in sorted(os.listdir(prof)):
    if fn.startswith(f"roberta-base__{CL}__") and fn.endswith(".json"):
        s = json.load(open(os.path.join(prof, fn), encoding="utf-8"))
        for key, lead in s.get("lead", {}).items():
            hid = {n: v for n, v in lead.items() if "attention.self" in n or "intermediate.dense" in n}
            er = np.array([v["energy_rel"] for v in lead.values()])
            print(f"  {fn} [{key}] {sum(v['argmax'] in (77, 588) for v in hid.values())}/{len(hid)} "
                  f"hidden-state modules on 77/588; energy median {np.median(er):.1f}x")
        if "drift_ratio" in s:
            print(f"    drift ratio {s['drift_ratio']}")

section("TAU-FREE DIVERGENCES")
for t in TASKS + [an.CLIN]:
    p = os.path.join(an.ROOT, "runs", "divergence", f"{RB}__{t}.json")
    if not os.path.exists(p):
        print(f"  {t}: n/a")
        continue
    d = json.load(open(p))
    sites = {}
    for n, v in d["modules"].items():
        sites[(figures.layer_of(n), figures.SITE_OF[figures.classify(n)])] = v
    for pair in ("target", "ref2", "news", "random"):
        if pair not in next(iter(sites.values())):
            continue
        vals = {k: float(np.mean([v[pair][k] for v in sites.values()]))
                for k in ("coral", "bures", "jeffreys")}
        print(f"  {t:9s} {pair:7s} " + " ".join(f"{k} {v:.4f}" for k, v in vals.items()))
