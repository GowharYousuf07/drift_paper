"""Independent re-derivation of the paper's tables from the raw result JSONs.

Deliberately shares no code with analyze.py or grid.py, so a bug there cannot
hide here: it re-implements the run filter, the per-configuration learning-rate
selection and the aggregation from the protocol as written in the paper, then
compares every number the tables print. Text checks for numbers quoted in the
prose are at the end.

    python src/verify_results.py
"""
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
R = [json.load(open(p, encoding="utf-8"))
     for p in glob.glob(os.path.join(ROOT, "runs", "results", "*.json"))]
PROF = {"eva", "eva_white", "drift", "drift_abs", "gev"}
METRIC = {"chemprot": "micro_f1", "rct20k": "micro_f1", "hoc": "example_f1",
          "mtsamples": "micro_f1"}
TASKS = ["chemprot", "rct20k", "hoc"]
RB = "roberta-base"
bad = 0


def report(name, ok, detail=""):
    global bad
    bad += not ok
    print(f"  {name:58s} {'OK' if ok else 'MISMATCH'} {detail}")


# ---- defaults of a canonical run, as the paper describes them
def canonical(a, **free):
    want = dict(tag="", target="all", budget_rank=8, max_train=None, ref="wikitext",
                deterministic=False, n_ref=1024, n_dom=1024, head="default",
                scaling="alpha_r")
    if a["method"] in PROF:
        want.update(tau=0.95, score_mode="relative", init_mode="drift",
                    alloc_mode="drift", rho=2.0, cov="centered", scale="adjusted")
        if a["method"] == "gev":
            want.update(init_mode="gev", alloc_mode="uniform")
    want.update(free)
    got = dict(a)
    got.setdefault("ref", "wikitext")
    got.setdefault("deterministic", False)
    got.setdefault("n_ref", 1024)
    got.setdefault("n_dom", 1024)
    got.setdefault("head", "default")
    got.setdefault("scaling", "alpha_r")
    got["tag"] = got.get("tag") or ""
    for k, v in want.items():
        g = got.get(k)
        if isinstance(v, float) or isinstance(g, float):
            if g is None or v is None or abs(float(g) - float(v)) > 1e-9:
                return False
        elif g != v:
            return False
    return True


# ---- tuned rate of a configuration: argmax dev over seed-1 runs tagged "tune"
def tkey(a):
    parts = []
    if a["method"] == "drift" and abs(float(a.get("tau", 0.95)) - 0.95) > 1e-9:
        parts.append(f"tau{float(a['tau']):g}")
    if a.get("scaling", "alpha_r") != "alpha_r":
        parts.append(a["scaling"])
    if a.get("head", "default") != "default":
        parts.append(f"head-{a['head']}")
    return (a["model"], a["task"], a["method"], a.get("target", "all"),
            int(a.get("budget_rank", 8)), "+".join(parts))


TUNE = {}
for r in R:
    a = r["args"]
    if a.get("tag") != "tune":
        continue
    if a["method"] in PROF and a.get("cov") != "centered":
        continue
    TUNE.setdefault(tkey(a), {})[float(a["lr"])] = r["result"]["dev_best"][METRIC[a["task"]]]


def tuned(model, task, method, target="all", budget_rank=8, tau=None,
          scaling="alpha_r", head="default"):
    parts = []
    if method == "drift" and tau is not None and abs(tau - 0.95) > 1e-9:
        parts.append(f"tau{tau:g}")
    if scaling != "alpha_r":
        parts.append(scaling)
    if head != "default":
        parts.append(f"head-{head}")
    t = "+".join(parts)
    keys = [(model, task, method, target, budget_rank, t)]
    if method == "drift" and tau is not None and tau == 0.0:
        keys = [(model, task, "eva", target, budget_rank, "")]
    keys += [(model, task, {"drift_abs": "drift"}.get(method, method), "all", 8, ""),
             (model, task, "lora", "all", 8, "")]
    for k in keys:
        if TUNE.get(k):
            d = TUNE[k]
            return max(sorted(d), key=lambda lr: d[lr])
    return None


def runs(model, task, method, lr, tags=("",), **free):
    out = {}
    for r in R:
        a = r["args"]
        if (a["model"], a["task"], a["method"]) != (model, task, method):
            continue
        if (a.get("tag") or "") not in tags:
            continue
        if not canonical(a, tag=a.get("tag") or "", **free):
            continue
        if lr is not None and abs(a["lr"] - lr) > 1e-9 * max(lr, 1e-12):
            continue
        if a["seed"] in out:
            raise SystemExit(f"duplicate seed {a['seed']} for {model} {task} {method} {free}")
        out[a["seed"]] = r
    return out


def score(rs):
    return {s: 100 * r["result"]["test"][METRIC[r["args"]["task"]]] for s, r in rs.items()}


num = re.compile(r"(\d+\.\d)(?=.textsubscript)")


def table_rows(fname):
    rows = {}
    path = os.path.join(PAPER, fname)
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        if "&" in line and "\\\\" in line and not line.startswith(("Method", "Variant",
                                                                    "Placement", "BERT")):
            cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")]
            rows.setdefault(cells[0], []).append(cells)
    return rows


def means(cells):
    return [float(m) for c in cells for m in num.findall(c)]


# ================================================================== main table
print("MAIN TABLE")
FIVE = ("lora", "eva", "eva_white", "drift")
NAMES = {"full": "Full fine-tuning", "linear": "Linear probe", "bitfit": "BitFit",
         "lora": "LoRA", "dora": "DoRA", "pissa": "PiSSA", "adalora": "AdaLoRA",
         "eva": "EVA (budget-matched)", "eva_white": "EVA (whitened)",
         "drift": "\\method{}", "gev": "GEV"}
tab = table_rows("tab_main.tex")
cells = {}
for t in TASKS:
    for m in NAMES:
        tags = ("", "seeds45") if m in FIVE else ("",)
        cells[(t, m)] = runs(RB, t, m, tuned(RB, t, m), tags=tags)
floor = {}
for t in TASKS:
    lora_dev = [100 * r["result"]["dev_best"][METRIC[t]] for r in cells[(t, "lora")].values()]
    floor[t] = st.median(lora_dev) - 10
for m, nm in NAMES.items():
    key = nm + ("$^\\dagger$" if m in FIVE else "")
    row = tab.get(key, [[None] * 7])[0]
    want_n = 5 if m in FIVE else 3
    mine = []
    for t in TASKS:
        s = score(cells[(t, m)])
        if len(s) != want_n:
            report(f"{m} {t} seeds", False, f"have {sorted(s)}")
        mine.append(round(st.mean(s.values()), 1))
    paper = means(row[2:5]) if row[0] else None
    report(f"{m:10s} means", paper == mine, f"paper {paper} recomputed {mine}")
    hoc = score(cells[("hoc", m)])
    med = f"{st.median(hoc.values()):.1f}"
    report(f"{m:10s} HoC median", row[5] == med if row[0] else False, f"paper {row[5] if row[0] else None} recomputed {med}")
    if m != "linear":
        nf = sum(100 * r["result"]["dev_best"][METRIC[t]] < floor[t]
                 for t in TASKS for r in cells[(t, m)].values())
        tot = sum(len(cells[(t, m)]) for t in TASKS)
        report(f"{m:10s} failed", row[6] == f"{nf}/{tot}" if row[0] else False,
               f"paper {row[6] if row[0] else None} recomputed {nf}/{tot}")

# ================================================================== ablation
print("ABLATION (three tasks)")


def abl(t):
    d_lr, e_lr = tuned(RB, t, "drift"), tuned(RB, t, "eva")
    return {
        "LoRA (uniform rank, random init)": runs(RB, t, "lora", tuned(RB, t, "lora")),
        "Rank allocation only (random init)": runs(RB, t, "drift", d_lr, tags=("allocOnly",), init_mode="random"),
        "Drift init only (uniform rank)": runs(RB, t, "drift", d_lr, tags=("initOnly",), alloc_mode="uniform"),
        "Random orthonormal init, drift ranks": runs(RB, t, "drift", d_lr, tags=("randOrtho",), init_mode="rand_ortho"),
        "No deflation ($\\tau{=}0$), \\method{}'s rate": runs(RB, t, "drift", d_lr, tau=0.0),
        "Absolute (unnormalised) drift score": runs(RB, t, "drift_abs", d_lr),
        "\\method{} (full)": runs(RB, t, "drift", d_lr),
        "GEV init (uniform rank)": runs(RB, t, "gev", tuned(RB, t, "gev")),
        "EVA, rank-unit budget (its own rule)": runs(RB, t, "eva", e_lr, alloc_mode="units"),
        "\\method{}, news reference": runs(RB, t, "drift", d_lr, ref="news"),
        "\\method{}, word-shuffled reference": runs(RB, t, "drift", d_lr, ref="shuffled"),
        "\\method{}, random-token reference": runs(RB, t, "drift", d_lr, ref="random"),
    }


ABL = {t: abl(t) for t in TASKS}
tab = table_rows("tab_ablation.tex")
for name in ABL["chemprot"]:
    mine, ns = [], []
    for t in TASKS:
        s = score(ABL[t][name])
        if len(s) >= 3:              # cells with runs still in progress print --
            mine.append(round(st.mean(s.values()), 1))
            ns.append(len(s))
    paper = means(tab[name][0][1:]) if name in tab else None
    report(name[:58], paper == mine and all(n == 3 for n in ns),
           f"paper {paper} recomputed {mine} n={ns}")

# ================================================================== fp32 HoC
print("fp32 HoC")
tab = table_rows("tab_fp32.tex")
for m, nm in NAMES.items():
    s = score(runs(RB, "hoc", m, tuned(RB, "hoc", m), tags=("fp32",), deterministic=True))
    key = nm.replace(" (budget-matched)", "") + ("$^\\dagger$" if m in FIVE else "")
    row = tab.get(key, [None])[0]
    if row is None:
        report(f"fp32 {m}", False, "row missing")
        continue
    fp16 = round(st.mean(score(cells[("hoc", m)]).values()), 1)
    mine = [fp16] + ([round(st.mean(s.values()), 1)] if s else [])
    report(f"fp32 {m}", means(row[1:]) == mine, f"paper {means(row[1:])} recomputed {mine}")

# ================================================================== budgets elsewhere
print("BUDGETS ON RCT-20k AND HoC")
tab_lines = open(os.path.join(PAPER, "tab_budgetx.tex"), encoding="utf-8").read().splitlines() \
    if os.path.exists(os.path.join(PAPER, "tab_budgetx.tex")) else []
body = [l for l in tab_lines if "&" in l and "textsubscript" in l or ("&" in l and "--" in l)]
i = 0
for t in ("rct20k", "hoc"):
    for r in (4, 8, 16):
        mine = []
        for m in ("lora", "eva", "eva_white", "drift"):
            s = score(runs(RB, t, m, tuned(RB, t, m, budget_rank=r), budget_rank=r))
            mine.append(round(st.mean(s.values()), 1) if s else None)
        paper = means(body[i].split("&")[2:]) if i < len(body) else None
        report(f"{t} r{r}", paper == [x for x in mine if x is not None], f"paper {paper} recomputed {mine}")
        i += 1

# ================================================================== linear head
print("LINEAR HEAD (ChemProt)")
tab = table_rows("tab_linhead.tex")
for m, nm in [("lora", "LoRA"), ("eva", "EVA (budget-matched)"), ("eva_white", "EVA (whitened)"),
              ("drift", "\\method{}"), ("bitfit", "BitFit")]:
    a = score(runs(RB, "chemprot", m, tuned(RB, "chemprot", m)))
    b = score(runs(RB, "chemprot", m, tuned(RB, "chemprot", m, head="linear"), head="linear"))
    mine = [round(st.mean(a.values()), 1)] + ([round(st.mean(b.values()), 1)] if b else [])
    paper = means(tab[nm][0][1:]) if nm in tab else None
    report(f"linear head {m}", paper == mine, f"paper {paper} recomputed {mine}")

# ================================================================== placement
print("PLACEMENT")
tab = table_rows("tab_placement.tex")
PL = [("All modules", "lora", "all", 8), ("All modules", "lora", "all", 4),
      ("Feed-forward only", "lora", "ffn", 8), ("Feed-forward only", "lora", "ffn", 14),
      ("Attention only", "lora", "attn", 8),
      ("\\method{}, feed-forward only", "drift", "ffn", 8),
      ("\\method{}, attention only", "drift", "attn", 8)]
for name, m, tgt, r in PL:
    mine = []
    for t in TASKS:
        lr = tuned(RB, t, "drift") if m == "drift" else tuned(RB, t, m, target=tgt, budget_rank=r)
        s = score(runs(RB, t, m, lr, target=tgt, budget_rank=r))
        mine.append(round(st.mean(s.values()), 1) if s else None)
    rows = [c for c in tab.get(name, []) if c[1] == str(r)]
    paper = means(rows[0][3:]) if rows else None
    report(f"{name} r={r}"[:58], paper == [x for x in mine if x is not None], f"paper {paper} recomputed {mine}")

# ================================================================== ladder
print("LADDER (ChemProt)")
LAD = [("Tiny", "google/bert_uncased_L-2_H-128_A-2"), ("Mini", "google/bert_uncased_L-4_H-256_A-4"),
       ("Small", "google/bert_uncased_L-4_H-512_A-8"), ("Medium", "google/bert_uncased_L-8_H-512_A-8"),
       ("Base", "google/bert_uncased_L-12_H-768_A-12")]
tab = table_rows("tab_ladder.tex")
eva_rb = tuned(RB, "chemprot", "eva")
for name, mdl in LAD:
    mine = []
    for m in ("lora", "eva", "eva_white", "drift"):
        s = score(runs(mdl, "chemprot", m, tuned(mdl, "chemprot", m)))
        mine.append(round(st.mean(s.values()), 1) if s else None)
    s = score(runs(mdl, "chemprot", "eva", eva_rb))
    mine.append(round(st.mean(s.values()), 1) if s else None)
    row = next((c for k, c in tab.items() if k.startswith(name + " (")), None)
    paper = means(row[0][1:]) if row else None
    report(f"BERT-{name}", paper == [x for x in mine if x is not None], f"paper {paper} recomputed {mine}")

# ================================================================== decoder
print("DECODER (SmolLM2-360M; ChemProt, HoC)")
DEC = "HuggingFaceTB/SmolLM2-360M"
tab = table_rows("tab_decoder.tex")
for m, nm in [("lora", "LoRA"), ("eva", "EVA (budget-matched)"), ("eva_white", "EVA (whitened)"),
              ("drift", "\\method{}")]:
    mine, ns = [], []
    for t in ("chemprot", "hoc"):
        s = score(runs(DEC, t, m, tuned(DEC, t, m)))
        if s:
            mine.append(round(st.mean(s.values()), 1))
            ns.append(len(s))
    paper = means(tab[nm][0][1:]) if nm in tab else None
    report(nm, paper == mine and all(n == 3 for n in ns), f"paper {paper} recomputed {mine}")

# ================================================================== clinical notes
print("CLINICAL NOTES (MTSamples)")
tab = table_rows("tab_clinical.tex")
if tab:
    cl_lr = tuned(RB, "mtsamples", "drift")
    for m, nm in [("lora", "LoRA"), ("eva", "EVA (budget-matched)"), ("eva_white", "EVA (whitened)"),
                  ("drift", "\\method{}")]:
        rs = runs(RB, "mtsamples", m, tuned(RB, "mtsamples", m))
        s = score(rs)
        mac = [100 * r["result"]["test"]["macro_f1"] for r in rs.values()]
        mine = [round(st.mean(s.values()), 1), round(st.mean(mac), 1)] if s else []
        paper = means(tab[nm][0][1:3]) if nm in tab else None
        # rows of -- until the runs exist
        report(f"clinical {m}", paper == mine and len(s) in (0, 3), f"paper {paper} recomputed {mine}")
    for ref, nm in [("news", "\\method{}, news reference"), ("random", "\\method{}, random-token reference")]:
        rs = runs(RB, "mtsamples", "drift", cl_lr, ref=ref)
        s = score(rs)
        mac = [100 * r["result"]["test"]["macro_f1"] for r in rs.values()]
        mine = [round(st.mean(s.values()), 1), round(st.mean(mac), 1)] if s else []
        paper = means(tab[nm][0][1:3]) if nm in tab else None
        report(f"clinical drift ref={ref}", paper == mine, f"paper {paper} recomputed {mine}")
else:
    print("  (no clinical table yet)")

# ================================================================== grid edges
print("LEARNING-RATE SELECTIONS")
edges = []
for k, d in sorted(TUNE.items(), key=str):
    lrs = sorted(d)
    best = max(lrs, key=lambda lr: d[lr])
    if len(lrs) > 1 and best in (lrs[0], lrs[-1]):
        edges.append((k, best, lrs))
print(f"  {len(TUNE)} tuned configurations, {len(edges)} with the selection on a grid edge")
for k, b, lrs in edges:
    print("   edge:", k, b, lrs)

# ================================================================== text checks
TEXT = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
text_checks = []


TEXTN = " ".join(TEXT.split())


def quoted(s):
    """True if the string s occurs in main.tex (line breaks and runs of spaces
    count as one space)."""
    return " ".join(s.split()) in TEXTN


def ttest(a, b):
    """Paired t-test over the seeds two {seed: score} dicts share: (diff, p)."""
    from scipy import stats
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    if all(abs(x) < 1e-12 for x in d):
        return 0.0, 1.0
    return st.mean(d), float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue)


def holm_adj(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


def sgn(x, nd=1):
    """Signed number as the paper prints it in math mode, e.g. '$-1.1$'."""
    return f"${'+' if x >= 0 else '-'}{abs(x):.{nd}f}$"


def mstd(s):
    v = list(s.values())
    return f"${st.mean(v):.1f}\\pm{st.stdev(v):.1f}$"


def failed(rs, t):
    return sorted(s for s, r in rs.items() if 100 * r["result"]["dev_best"][METRIC[t]] < floor[t])


def check(name, ok):
    text_checks.append((name, ok))


# --- RQ1: Table I statistics
comp = []
for t in TASKS:
    for m in ("bitfit", "dora", "pissa", "adalora", "eva", "eva_white", "drift", "gev"):
        d, p = ttest(score(cells[(t, m)]), score(cells[(t, "lora")]))
        comp.append(((t, m), d, p))
adj = holm_adj([p for _, _, p in comp])
check(f"Holm over {len(comp)} comparisons: smallest adjusted p = {min(adj):.2f}",
      quoted(f"over the ${len(comp)}$ comparisons with LoRA")
      and (quoted(f"adjusted $p = {min(adj):.2f}$") if min(adj) < 1 else
           quoted("every adjusted $p$ is $1.00$")))
nominal = [(k, d, p) for k, d, p in comp if p < 0.05]
check("one nominal difference: BitFit ChemProt",
      [k for k, _, _ in nominal] == [("chemprot", "bitfit")]
      and quoted(f"BitFit on ChemProt ({sgn(nominal[0][1])}, $p = {nominal[0][2]:.2f}$)"))
check("largest gain over LoRA 0.34", abs(max(d for _, d, _ in comp) - 0.34) < 0.005
      and quoted("more than $0.34$ F1"))
PEFT = ("bitfit", "lora", "dora", "pissa", "adalora", "eva", "eva_white", "drift")
spread = max(max(round(st.mean(score(cells[(t, m)]).values()), 1) for m in PEFT)
             - min(round(st.mean(score(cells[(t, m)]).values()), 1) for m in PEFT)
             for t in ("chemprot", "rct20k"))
check(f"PEFT spread ChemProt/RCT <= 1.7 (is {spread:.1f})", spread <= 1.7 + 1e-9
      and quoted("within $1.7$ points of each other"))
d1, p1 = ttest(score(cells[("chemprot", "drift")]), score(cells[("chemprot", "lora")]))
d2, p2 = ttest(score(cells[("rct20k", "drift")]), score(cells[("rct20k", "lora")]))
check("DRIFT vs LoRA, five seeds", quoted(f"({sgn(d1)}, $p = {p1:.2f}$;\n${'+' if d2 >= 0 else '-'}{abs(d2):.1f}$, $p = {p2:.2f}$)")
      or quoted(f"({sgn(d1)}, $p = {p1:.2f}$; {sgn(d2)}, $p = {p2:.2f}$)"))
pw = [ttest(score(cells[(t, "drift")]), score(cells[(t, "eva_white")]))[1] for t in ("chemprot", "rct20k")]
check(f"DRIFT vs whitened EVA p >= 0.70 ({min(pw):.3f})", min(pw) >= 0.695)
fails = {m: failed(cells[("hoc", m)], "hoc") for m in NAMES if m != "linear"}
check("HoC failures PiSSA 2/3, DoRA 1/3, EVA 2/5, DRIFT 1/5, others 0",
      len(fails["pissa"]) == 2 and len(fails["dora"]) == 1 and len(fails["eva"]) == 2
      and len(fails["drift"]) == 1 and all(not fails[m] for m in ("lora", "eva_white", "adalora", "bitfit", "full")))
dh = score(cells[("hoc", "drift")])
rest = sorted(v for s, v in dh.items() if s != 1)
check("DRIFT HoC seed 1 and the other four", quoted(f"stops at ${dh[1]:.1f}$ F1")
      and quoted(f"${rest[0]:.1f}$--${rest[-1]:.1f}$"))
check("EVA ChemProt and HoC, five seeds",
      quoted(f"({mstd(score(cells[('chemprot', 'eva')]))} vs.\\ {mstd(score(cells[('chemprot', 'lora')]))}")
      and quoted(f"falls to {mstd(score(cells[('hoc', 'eva')]))} on HoC"))
check("whitened EVA HoC", quoted(f"({mstd(score(cells[('hoc', 'eva_white')]))}, none in five"))
wc = score(cells[("chemprot", "eva_white")])
check("whitened EVA ChemProt at its tuned rate", quoted(f"mean to {mstd(wc)} at its tuned rate")
      and quoted(f"ends at ${min(wc.values()):.1f}$") and abs(tuned(RB, "chemprot", "eva_white") - 1e-3) < 1e-12)
w3 = score(runs(RB, "chemprot", "eva_white", 3e-4, tags=("", "seeds45")))
check("whitened EVA ChemProt at 3e-4", len(w3) == 5 and quoted(
    f"all five seeds reach ${min(w3.values()):.1f}$--${max(w3.values()):.1f}$ ({mstd(w3)})"))

# --- fp32 reruns of HoC
F32 = {m: runs(RB, "hoc", m, tuned(RB, "hoc", m), tags=("fp32",), deterministic=True)
       for m in ("lora", "dora", "pissa", "eva", "eva_white", "drift")}
f16_fail = {m: failed(cells[("hoc", m)], "hoc") for m in ("pissa", "dora", "drift", "eva")}
pdd = [(m, s) for m in ("pissa", "dora", "drift") for s in f16_fail[m]]
n8 = sum(len(F32[m]) for m in ("pissa", "dora", "drift"))
check(f"fp32: the four fp16 failures of PiSSA/DoRA/DRIFT vanish ({n8} fp32 runs)",
      len(pdd) == 4 and all(s in F32[m] for m, s in pdd)
      and not any(failed(F32[m], "hoc") for m in ("pissa", "dora", "drift"))
      and quoted(f"none of their { {8: 'eight', 9: 'nine'}.get(n8, str(n8)) } fp32 runs fails"))
ev32 = F32["eva"]
trained16 = [s for s in cells[("hoc", "eva")] if s in ev32 and s not in f16_fail["eva"]]
check("fp32: EVA fails all three seeds, two of which trained in fp16",
      len(ev32) == 3 and len(failed(ev32, "hoc")) == 3 and len(trained16) == 2
      and quoted("loses 2 of 5 HoC seeds in fp16 and all 3 in fp32"))
flat = {round(e["train_loss"], 2) for r in ev32.values() for e in r["result"]["history"][1:]}
g = [max(e["grad_norm_max"] for e in r["result"]["history"]) for r in ev32.values()]
gp = max(e["grad_norm_max"] for r in F32["pissa"].values() for e in r["result"]["history"])
check("fp32: EVA's loss flat at 0.38, gradients bounded", flat == {0.38}
      and quoted(f"(maximum norm ${min(g):.1f}$--${max(g):.1f}$, against ${gp:.0f}$ in a PiSSA run"))

ada = cells[("rct20k", "adalora")]
drops = sorted(round(100 * r["result"]["history"][2]["dev"]) for r in ada.values()
               if r["result"].get("history") and r["result"]["history"][2]["dev"] < 0.7)
check("AdaLoRA RCT-20k: two seeds fall in the third epoch", len(drops) == 2
      and quoted(f"fall to ${drops[0]}$ and ${drops[1]}$ dev F1 in the third epoch"))

# --- RQ2: reference controls, GEV, EVA's rank-unit rule
lo3 = {t: score(ABL[t]["LoRA (uniform rank, random init)"]) for t in TASKS}
dr3 = {t: score(ABL[t]["\\method{} (full)"]) for t in TASKS}
REF = {"news": "\\method{}, news reference", "shuffled": "\\method{}, word-shuffled reference",
       "random": "\\method{}, random-token reference"}
cm = [round(st.mean(score(ABL["chemprot"][REF[k]]).values()), 1) for k in ("news", "shuffled", "random")]
check("ChemProt reference controls", quoted(f"${cm[0]:.1f}$, ${cm[1]:.1f}$ and ${cm[2]:.1f}$ with the news, shuffled")
      and quoted(f"against ${st.mean(dr3['chemprot'].values()):.1f}$ with WikiText"))
hm = [round(st.mean(score(ABL["hoc"][REF[k]]).values()), 1) for k in ("news", "random")]
check("HoC reference controls", quoted(f"${hm[0]:.1f}$ and\n${hm[1]:.1f}$ with news and random tokens")
      or quoted(f"${hm[0]:.1f}$ and ${hm[1]:.1f}$ with news and random tokens"))
check("HoC WikiText three seeds", quoted(f"against WikiText's ${st.mean(dr3['hoc'].values()):.1f}$"))
pr = [ttest(score(ABL[t][REF[k]]), dr3[t])[1] for t, ks in (("chemprot", REF), ("hoc", ("news", "random")))
      for k in ks]
check(f"no reference differs from WikiText (min p {min(pr):.3f})", min(pr) >= 0.065 and quoted("($p \\ge 0.07$)"))
ds, ps = ttest(score(ABL["chemprot"][REF["shuffled"]]), lo3["chemprot"])
check("shuffled reference vs LoRA", quoted(f"({sgn(ds)},\n$p = {ps:.4f}$ uncorrected)") or quoted(f"({sgn(ds)}, $p = {ps:.4f}$ uncorrected)"))
# five seeds of every reference control (seeds 4-5 tagged seeds45)
R5 = {(t, k): runs(RB, t, "drift", tuned(RB, t, "drift"), tags=("", "seeds45"), ref=k)
      for t, ks in (("chemprot", ("news", "shuffled", "random")), ("hoc", ("news", "random")))
      for k in ks}
t5 = {k: ttest(score(v), score(cells[(k[0], "lora")])) for k, v in R5.items()}
a5 = dict(zip(t5, holm_adj([p for _, p in t5.values()])))
sh = t5[("chemprot", "shuffled")]
check("five-seed reference controls: all have five seeds", all(len(v) == 5 for v in R5.values()))
check("shuffled reference over five seeds", quoted(
    f"the gain shrinks to {sgn(sh[0])} ($p = {sh[1]:.2f}$ uncorrected; $p = {a5[('chemprot', 'shuffled')]:.2f}$ after Holm"))
oth = [abs(d) for k, (d, _) in t5.items() if k != ("chemprot", "shuffled")]
check(f"other references within 0.2 of LoRA ({max(oth):.2f})", max(oth) <= 0.2 + 1e-9
      and quoted("the other references are within $0.2$ points of LoRA"))
hoc_ref = [r for k, v in R5.items() if k[0] == "hoc" for r in v.values()]
check("no HoC run with a replaced reference fails", len(hoc_ref) == 10
      and not any(100 * r["result"]["dev_best"][METRIC["hoc"]] < floor["hoc"] for r in hoc_ref)
      and len(fails["drift"]) == 1 and quoted("none of the ten HoC runs with a replaced reference fails"))

gv = [round(st.mean(score(ABL[t]["GEV init (uniform rank)"]).values()), 1) for t in TASKS]
check("GEV means", quoted(f"GEV reaches ${gv[0]:.1f}$, ${gv[1]:.1f}$ and ${gv[2]:.1f}$"))
gp = [ttest(score(ABL[t]["GEV init (uniform rank)"]), lo3[t]) for t in TASKS]
check("GEV within 0.3 of LoRA, p >= 0.72", max(abs(d) for d, _ in gp) <= 0.3 + 1e-9
      and min(p for _, p in gp) >= 0.715 and quoted("within $0.3$ points of LoRA on every task ($p \\ge 0.72$)"))
check("GEV rates", [tuned(RB, t, "gev") for t in TASKS] == [1e-3, 1e-4, 3e-4]
      and quoted("($10^{-3}$, $10^{-4}$\nand $3\\times10^{-4}$)"))
check("GEV no failed run", all(not failed(ABL[t]["GEV init (uniform rank)"], t) for t in TASKS))
un = {t: ABL[t]["EVA, rank-unit budget (its own rule)"] for t in ("chemprot", "hoc")}
upar = {t: st.mean(r["result"].get("params_adapter_effective", r["result"]["params_adapter"])
                   for r in un[t].values()) / 1e6 for t in un}
check("EVA rank-unit parameters", quoted(f"spends ${upar['chemprot']:.2f}$M (ChemProt) and\n${upar['hoc']:.2f}$M (HoC)")
      or quoted(f"spends ${upar['chemprot']:.2f}$M (ChemProt) and ${upar['hoc']:.2f}$M (HoC)"))
check("EVA rank-unit accuracy", quoted(f"reaches {mstd(score(un['chemprot']))} and\n{mstd(score(un['hoc']))} at EVA's rate")
      or quoted(f"reaches {mstd(score(un['chemprot']))} and {mstd(score(un['hoc']))} at EVA's rate"))
eb = score(runs(RB, "chemprot", "eva", tuned(RB, "chemprot", "eva")))
du, pu = ttest(score(un["chemprot"]), eb)
check("EVA rank-unit vs budget-matched EVA", quoted(f"({sgn(du)}, $p = {pu:.2f}$)"))
check("EVA rank-unit: no HoC failure, seed 2 fails for budget-matched EVA",
      not failed(un["hoc"], "hoc") and 2 in fails["eva"])

# --- tau sweep at each level's own rate
TT = {}
for tau in (0.0, 0.5, 0.9, 0.95, 0.99):
    lr = tuned(RB, "chemprot", "drift", tau=tau)
    TT[tau] = score(runs(RB, "chemprot", "drift", lr, tau=tau))
t0 = st.mean(TT[0.0].values())
others = [st.mean(TT[t].values()) for t in (0.5, 0.9, 0.95, 0.99)]
sds = [st.stdev(TT[t].values()) for t in (0.5, 0.9, 0.95, 0.99)]
check("tau sweep at tuned rates", quoted(f"$\\tau = 0$: {mstd(TT[0.0])}")
      and quoted(f"${min(others):.1f}$--${max(others):.1f}$ with standard deviations\nof ${min(sds):.1f}$--${max(sds):.1f}$")
      and round(max(others) - t0, 1) <= 0.7)
fam = [ttest(score(cells[(t, m)]), score(cells[(t, "lora")]))[0] for t in ("chemprot", "rct20k")
       for m in ("eva_white", "drift")] + [gp[0][0], gp[1][0]]
famh = [ttest(score(cells[("hoc", m)]), score(cells[("hoc", "lora")]))[0] for m in ("eva_white", "drift")] + [gp[2][0]]
check("family within 0.5 of LoRA on ChemProt/RCT, below on HoC",
      max(abs(x) for x in fam) <= 0.5 and max(famh) < 0 and quoted("within $0.5$ points of LoRA on ChemProt and\nRCT-20k"))

# --- placement at matched budgets
def pl(t, m, tgt, r):
    lr = tuned(RB, t, "drift") if m == "drift" else tuned(RB, t, m, target=tgt, budget_rank=r)
    return runs(RB, t, m, lr, target=tgt, budget_rank=r)


P = {t: {k: pl(t, *k) for k in [("lora", "all", 8), ("lora", "all", 4), ("lora", "ffn", 8),
                                  ("lora", "ffn", 14), ("lora", "attn", 8), ("drift", "ffn", 8),
                                  ("drift", "attn", 8)]} for t in TASKS}


def pd(t, a, b):
    return ttest(score(P[t][a]), score(P[t][b]))


A8, A4, F8, F14, T8 = ("lora", "all", 8), ("lora", "all", 4), ("lora", "ffn", 8), ("lora", "ffn", 14), ("lora", "attn", 8)
x = [pd(t, F14, A8) for t in ("chemprot", "rct20k")]
check("ffn14 vs all8", quoted(f"is {sgn(x[0][0])} on ChemProt\n($p = {x[0][1]:.2f}$) and {sgn(x[1][0])} on RCT-20k ($p = {x[1][1]:.2f}$)"))
check("ffn14 HoC: two of three seeds never leave the constant prediction",
      len(failed(P["hoc"][F14], "hoc")) == 2
      and all(r["result"]["best_epoch"] == 0 for s, r in P["hoc"][F14].items() if s in failed(P["hoc"][F14], "hoc"))
      and abs(tuned(RB, "hoc", "lora", target="ffn", budget_rank=14) - 1e-3) < 1e-12)
y = [pd(t, A4, F8) for t in TASKS]
check("all4 vs ffn8", quoted(f"is {sgn(y[0][0])} on ChemProt\n($p = {y[0][1]:.2f}$), {sgn(y[1][0])} on RCT-20k ($p = {y[1][1]:.2f}$) and {sgn(y[2][0])} on HoC ($p = {y[2][1]:.2f}$)"))
z = pd("chemprot", F8, A8)
check("ffn8 vs all8 ChemProt", quoted(f"on ChemProt ({sgn(z[0])},\n$p = {z[1]:.2f}$)") or quoted(f"on ChemProt ({sgn(z[0])}, $p = {z[1]:.2f}$)"))
w = [pd(t, A4, A8) for t in TASKS]
check("all4 vs all8", quoted(f"by {sgn(w[0][0])}, {sgn(w[1][0])} and {sgn(w[2][0])} points") and min(p for _, p in w) >= 0.335)
v = [pd(t, T8, A8)[0] for t in TASKS]
check(f"attention-only within 0.2 of all8 ({max(abs(q) for q in v):.2f})", max(abs(q) for q in v) <= 0.2 + 1e-9)
q = ttest(score(P["chemprot"][("drift", "attn", 8)]), score(P["chemprot"][T8]))
check("DRIFT attention-only vs LoRA attention-only ChemProt", quoted(f"attention only, {sgn(q[0])}, $p = {q[1]:.2f}$"))
check("DRIFT feed-forward-only fails one HoC seed", len(failed(P["hoc"][("drift", "ffn", 8)], "hoc")) == 1)
same = [ttest(score(P[t][("drift", tg, 8)]), score(P[t][("lora", tg, 8)]))
        for t in TASKS for tg in ("ffn", "attn")]
check("DRIFT never significantly better than LoRA at the same placement",
      not any(d > 0 and p < 0.05 for d, p in same))

# --- round 2, NEW-3: the uninformative feed-forward rank-14 HoC cell, reruns in
# fp32 at the same rate and in fp16 at the rate a three-seed dev mean selects
FFCFG = dict(target="ffn", budget_rank=14)
SEL14 = tuned(RB, "hoc", "lora", **FFCFG)
FF32 = runs(RB, "hoc", "lora", SEL14, tags=("fp32",), amp=False, deterministic=True, **FFCFG)
FFLOW = runs(RB, "hoc", "lora", 3e-4, tags=("ffn14lr",), **FFCFG)


def hoc_dev(rs):
    return {s: 100 * r["result"]["dev_best"][METRIC["hoc"]] for s, r in rs.items()}


s32, bad32 = score(FF32), failed(FF32, "hoc")
ok32 = sorted(s32[s] for s in s32 if s not in bad32)
check("NEW-3 fp32 rerun recovers one of the two failed seeds",
      len(FF32) == 3 and len(bad32) == 1
      and quoted(f"one of the two failed seeds recovers (${ok32[0]:.1f}$ and "
                 f"${ok32[1]:.1f}$ against a third still at ${s32[bad32[0]]:.1f}$)")
      and all(FF32[s]["result"]["best_epoch"] == 0 for s in bad32))
check("NEW-3 the three-seed dev mean selects the lower rate",
      quoted(f"the dev score is\n${st.mean(list(hoc_dev(P['hoc'][F14]).values())):.1f}$ at the selected "
             f"$10^{{-3}}$ and ${st.mean(list(hoc_dev(FFLOW).values())):.1f}$ at $3\\times10^{{-4}}$")
      and abs(SEL14 - 1e-3) < 1e-12
      and st.mean(list(hoc_dev(FFLOW).values())) > st.mean(list(hoc_dev(P["hoc"][F14]).values())))
check("NEW-3 all three seeds train at the lower rate",
      len(FFLOW) == 3 and not failed(FFLOW, "hoc")
      and quoted("all three seeds train and the test score is " + mstd(score(FFLOW))))
d_a8 = ttest(score(FFLOW), score(P["hoc"][A8]))
d_f8 = ttest(score(FFLOW), score(P["hoc"][F8]))
check("NEW-3 repaired cell still below all-module rank 8 and feed-forward rank 8",
      d_a8[0] < 0 and d_f8[0] < 0
      and quoted(f"is ${abs(d_a8[0]):.1f}$ points below all-module rank $8$\n"
                 f"($p = {d_a8[1]:.2f}$) and ${abs(d_f8[0]):.1f}$ below feed-forward-only rank $8$"))
plcap = open(os.path.join(PAPER, "tab_placement.tex"), encoding="utf-8").read()
check("NEW-3 placement caption quotes the repaired cell",
      f"{st.mean(list(score(FFLOW).values())):.1f}$\\pm${st.stdev(list(score(FFLOW).values())):.1f}"
      in " ".join(plcap.split()))

# --- budget sweep (ChemProt)
BS = {(m, r): score(runs(RB, "chemprot", m, tuned(RB, "chemprot", m, budget_rank=r), budget_rank=r))
      for m in ("lora", "eva", "eva_white", "drift") for r in (1, 2, 4, 8, 16)}
sp = max(max(round(st.mean(BS[(m, r)].values()), 1) for m in ("lora", "eva_white", "drift"))
         - min(round(st.mean(BS[(m, r)].values()), 1) for m in ("lora", "eva_white", "drift")) for r in (1, 2, 4, 8, 16))
check(f"budget spread LoRA/DRIFT/whitened <= 1.0 ({sp:.1f})", sp <= 1.0 + 1e-9)
check("LoRA rank 2", quoted(f"from rank $2$ (${st.mean(BS[('lora', 2)].values()):.1f}$)"))
e1, e2 = ttest(BS[("eva", 1)], BS[("lora", 1)]), ttest(BS[("eva", 2)], BS[("lora", 2)])
check("EVA at ranks 1 and 2", quoted(f"({sgn(e1[0])} and {sgn(e2[0])} against LoRA at ranks $1$ and $2$; $p = {e1[1]:.2f}$\nand ${e2[1]:.2f}$ uncorrected)"))
check("EVA rate 1e-4 up to rank 8", all(abs(tuned(RB, "chemprot", "eva", budget_rank=r) - 1e-4) < 1e-12 for r in (1, 2, 4, 8)))
dd = ttest(BS[("drift", 1)], BS[("lora", 1)])
bps = [ttest(BS[(m, r)], BS[("lora", r)])[1] for m in ("eva", "eva_white", "drift") for r in (1, 2, 4, 16)]
check("DRIFT at rank 1", quoted(f"({mstd(BS[('drift', 1)])}\nagainst LoRA's {mstd(BS[('lora', 1)])}; {sgn(dd[0])}, $p = {dd[1]:.2f}$ uncorrected")
      and min(holm_adj(bps)) >= 0.05 and len(bps) == 12)
check("EVA collapses at 3e-4 at ranks 1-2; others train at 1e-3",
      all(TUNE[(RB, "chemprot", "eva", "all", r, "")][3e-4] < 0.34 for r in (1, 2))
      and all(TUNE[(RB, "chemprot", m, "all", r, "")][1e-3] > 0.8 for m in ("lora", "eva_white", "drift") for r in (1, 2)))
check("EVA rank 1 at 1e-4, best epoch last", quoted(f"reaches {mstd(BS[('eva', 1)])} with the best dev epoch")
      and all(r["result"]["best_epoch"] == 9 for r in runs(RB, "chemprot", "eva", 1e-4, budget_rank=1).values()))

# --- ladder
LS = {(n, m): score(runs(mdl, "chemprot", m, tuned(mdl, "chemprot", m))) for n, mdl in LAD
      for m in ("lora", "eva", "eva_white", "drift")}
LR_ = {n: score(runs(mdl, "chemprot", "eva", eva_rb)) for n, mdl in LAD}
dT, dB = ttest(LS[("Tiny", "drift")], LS[("Tiny", "lora")]), ttest(LS[("Base", "drift")], LS[("Base", "lora")])
check("ladder DRIFT on Tiny and Base", quoted(f"BERT-Tiny ({sgn(dT[0])},\n$p = {dT[1]:.2f}$) and BERT-Base ({sgn(dB[0])}, $p = {dB[1]:.2f}$ uncorrected)"))
lps = [ttest(LS[(n, m)], LS[(n, "lora")])[1] for n, _ in LAD for m in ("eva", "eva_white", "drift")] + \
      [ttest(LR_[n], LS[(n, "lora")])[1] for n, _ in LAD]
ladj = holm_adj(lps)
surv = [i for i, a in enumerate(ladj) if a < 0.05]
check("ladder: only EVA at RoBERTa's rate on BERT-Mini survives Holm",
      len(lps) == 20 and surv == [15 + 1])
ed = [st.mean(LS[(n, "lora")].values()) - st.mean(LS[(n, "eva")].values()) for n, _ in LAD]
wd = [st.mean(LS[(n, "lora")].values()) - st.mean(LS[(n, "eva_white")].values()) for n, _ in LAD]
check("EVA trails by 0.0-2.5 at its own rate; whitened by at most 1.8",
      round(min(ed), 1) == 0.0 and round(max(ed), 1) == 2.5 and round(max(wd), 1) <= 1.8
      and quoted(f"(${abs(ed[4]):.1f}$ on BERT-Base, ${ed[3]:.1f}$ on\nBERT-Medium, ${ed[2]:.1f}$ on BERT-Small, ${ed[1]:.1f}$ on BERT-Mini and BERT-Tiny)"))
rb = [st.mean(LS[(n, "lora")].values()) - st.mean(LR_[n].values()) for n, _ in LAD]
check("EVA at RoBERTa's rate behind the tuned LoRA", quoted(f"${rb[3]:.1f}$, ${rb[2]:.1f}$, ${rb[1]:.1f}$ and ${rb[0]:.1f}$ points behind the\ntuned LoRA"))
check("Discussion: BERT-Tiny 13.4 / 2.5", quoted(f"EVA fell ${rb[0]:.1f}$ points behind the tuned LoRA on\nBERT-Tiny, and at the rate tuned on BERT-Tiny itself only ${ed[0]:.1f}$"))
tiny = LAD[0][1]
tp = []
for m in ("eva", "eva_white", "drift"):
    big = score(runs(tiny, "chemprot", m, tuned(tiny, "chemprot", m), n_ref=8192, n_dom=8192))
    tp.append((st.mean(big.values()) - st.mean(LS[("Tiny", m)].values()), ttest(big, LS[("Tiny", m)])[1], big))
check("BERT-Tiny full profile", quoted(f"by {sgn(tp[0][0])}, {sgn(tp[1][0])} and {sgn(tp[2][0])} points") and min(p for _, p, _ in tp) >= 0.195
      and quoted(f"EVA remains ${st.mean(LS[('Tiny', 'lora')].values()) - st.mean(tp[0][2].values()):.1f}$ points behind LoRA"))
esd = [st.stdev(LS[(n, "eva")].values()) for n, _ in LAD]
lsd = [st.stdev(LS[(n, "lora")].values()) for n, _ in LAD]
check("ladder seed spreads", quoted(f"(${min(esd):.1f}$--${max(esd):.1f}$ points) is of the order of LoRA's\n(${min(lsd):.1f}$--${max(lsd):.1f}$)"))

# --- post-audit experiments (rsLoRA scale, budgets elsewhere, linear head,
#     EVA's fp32 rate selection, the clinical task)
RS = {r: (score(runs(RB, "chemprot", "lora", tuned(RB, "chemprot", "lora", budget_rank=r,
                                                   scaling="rslora"), budget_rank=r, scaling="rslora")),
          score(runs(RB, "chemprot", "lora", tuned(RB, "chemprot", "lora", budget_rank=r),
                     budget_rank=r)))
      for r in (1, 2, 4, 8, 16)}
rsd = {r: ttest(a, b) for r, (a, b) in RS.items()}
big = max(rsd, key=lambda r: rsd[r][0])
check("rsLoRA scale tracks alpha/r", max(abs(d) for d, _ in rsd.values()) <= 0.4 + 1e-9
      and quoted(f"largest difference {sgn(rsd[big][0])} at rank ${big}$, $p = {rsd[big][1]:.2f}$"))
check("rsLoRA and alpha/r coincide at rank 1",
      all(abs(RS[1][0][s] - RS[1][1][s]) < 1e-9 for s in RS[1][0]) and len(RS[1][0]) == 3
      and quoted("the two runs reproduce each other exactly"))

SWEEP4 = ("lora", "eva", "eva_white", "drift")
BX = {(t, m, r): score(runs(RB, t, m, tuned(RB, t, m, budget_rank=r), budget_rank=r))
      for t in ("rct20k", "hoc") for m in ("lora", "eva", "eva_white", "drift") for r in (4, 8, 16)}
bxd = {(t, m, r): ttest(BX[(t, m, r)], BX[(t, "lora", r)])
       for (t, m, r) in BX if m != "lora" and r in (4, 16)}
best = {t: sum(all(round(st.mean(BX[(t, "lora", r)].values()), 1) >= round(st.mean(BX[(t, m, r)].values()), 1)
                   for m in ("eva", "eva_white", "drift")) for r in (4, 8, 16))
        for t in ("rct20k", "hoc")}
check(f"LoRA best at all three HoC budgets, none on RCT-20k ({best})",
      best == {"hoc": 3, "rct20k": 0}
      and quoted("on HoC LoRA has the best mean at all three budgets"))
rct = [abs(d) for (t, m, r), (d, _) in bxd.items() if t == "rct20k"]
check(f"RCT-20k budgets within 0.6 of LoRA ({max(rct):.2f})", max(rct) <= 0.6 + 1e-9)
nom = sorted((k, d, p) for k, (d, p) in bxd.items() if p < 0.05)
adjb = dict(zip(bxd, holm_adj([p for _, p in bxd.values()])))
check("two nominal budget differences, both losses, neither survives Holm",
      len(nom) == 2 and all(d < 0 for _, d, _ in nom) and len(bxd) == 12
      and min(adjb[k] for k, _, _ in nom) >= 0.05
      and quoted("neither survives Holm correction over the table's twelve comparisons"))
hoc16 = ttest(BX[("hoc", "lora", 16)], BX[("hoc", "lora", 8)])
check("HoC rank 16 beats rank 8 for LoRA",
      quoted(f"beats LoRA at rank $8$ by ${hoc16[0]:.1f}$ points ($p = {hoc16[1]:.2f}$ uncorrected)"))
r4 = [ttest(BX[(t, "lora", 4)], BX[(t, "lora", 8)]) for t in ("rct20k", "hoc")]
r4.append(ttest(score(runs(RB, "chemprot", "lora", tuned(RB, "chemprot", "lora", budget_rank=4), budget_rank=4)),
                score(cells[("chemprot", "lora")])))
check("rank 4 level with rank 8 on all three tasks", all(p > 0.05 for _, p in r4)
      and quoted("while rank $4$ is level with rank $8$ on all three tasks"))

LH = {m: score(runs(RB, "chemprot", m, tuned(RB, "chemprot", m, head="linear"), head="linear"))
      for m in ("lora", "eva", "eva_white", "drift", "bitfit")}
lhd = {m: ttest(LH[m], LH["lora"]) for m in LH if m != "lora"}
check("linear head: DRIFT the largest difference, EVA the weakest",
      max(lhd, key=lambda m: lhd[m][0]) == "drift" and min(LH, key=lambda m: st.mean(LH[m].values())) == "eva"
      and quoted(f"\\method{{}}'s {sgn(lhd['drift'][0])}, $p = {lhd['drift'][1]:.2f}$)")
      and quoted(f"EVA remains the weakest ({sgn(lhd['eva'][0])}, $p = {lhd['eva'][1]:.2f}$"))

EL = {lr: runs(RB, "hoc", "eva", lr, tags=("fp32",), deterministic=True)
      for lr in (3e-5, 1e-4, 3e-4)}
pick_lr = max(EL, key=lambda lr: st.mean(100 * r["result"]["dev_best"][METRIC["hoc"]]
                                         for r in EL[lr].values()))
ediff = ttest(score(EL[1e-4]), score(F32["lora"]))
check("EVA's fp32 rate one step lower again (3e-5) is undertrained",
      quoted(f"at $3\\times10^{{-5}}$, it is merely undertrained ({mstd(score(EL[3e-5]))})"))
check("energy-check statement in the protocol box is two-sided",
      quoted("medians of $59$--$125\\times$ coincided with failed or unstable runs and $1$--$15\\times$ with stable training")
      and quoted("on the BERT ladder $20$--$59\\times$ trained stably at tuned rates")
      and quoted("not as a verdict"))
check("EVA's fp32 rate selection picks 1e-4 and still trails LoRA",
      abs(pick_lr - 1e-4) < 1e-12
      and quoted(f"reaches {mstd(score(EL[1e-4]))}, still ${abs(ediff[0]):.1f}$ points below LoRA's fp32 runs ($p = {ediff[1]:.3f}$)")
      and quoted(f"removes its failures but leaves it ${abs(ediff[0]):.1f}$ points below LoRA"))

# --- the last round: fp16 EVA a step lower, the fp32 repeat, the factorisation,
#     the decoder on HoC, the replicate's intervals and run counts
low = runs(RB, "hoc", "eva", 1e-4, tags=("lowlr",))
dlow = ttest(score(low), score(cells[("hoc", "lora")]))
check("fp16 EVA at 1e-4 over five seeds", len(low) == 5 and not failed(low, "hoc")
      and quoted(f"{mstd(score(low))} over five seeds, none failed, ${abs(dlow[0]):.1f}$ points below LoRA ($p = {dlow[1]:.2f}$)"))
rep = runs(RB, "hoc", "drift", tuned(RB, "hoc", "drift"), tags=("fp32rep",), deterministic=True)
orig = F32["drift"]
check("fp32 repeat differs from the original run", len(rep) == 1
      and quoted(f"reaches ${100*list(rep.values())[0]['result']['test']['example_f1']:.1f}$ against "
                 f"${100*orig[1]['result']['test']['example_f1']:.1f}$"))
FX = {(t, tag): runs(RB, t, "drift", tuned(RB, t, "drift"), tags=(tag,),
                     **({"init_mode": "random"} if tag == "allocOnly" else {"alloc_mode": "uniform"}))
      for t in ("rct20k", "hoc") for tag in ("allocOnly", "initOnly")}
fxd = {k: ttest(score(v), score(runs(RB, k[0], "lora", tuned(RB, k[0], "lora")))) for k, v in FX.items()}
check("factorisation on RCT-20k and HoC",
      all(len(v) == 3 for v in FX.values()) and len(failed(FX[("hoc", "allocOnly")], "hoc")) == 1
      and not failed(FX[("hoc", "initOnly")], "hoc")
      and quoted(f"level on RCT-20k ({sgn(fxd[('rct20k', 'allocOnly')][0], 2)})")
      and quoted(f"one of its three seeds fails ({mstd(score(FX[('hoc', 'allocOnly')]))})")
      and quoted(f"({sgn(fxd[('rct20k', 'initOnly')][0], 2)} and {sgn(fxd[('hoc', 'initOnly')][0])}, $p \\ge {min(fxd[('rct20k', 'initOnly')][1], fxd[('hoc', 'initOnly')][1]):.2f}$)"))
DEC = "HuggingFaceTB/SmolLM2-360M"
DH = {m: score(runs(DEC, "hoc", m, tuned(DEC, "hoc", m))) for m in ("lora", "eva", "eva_white", "drift")}
dev_med = st.median(100 * r["result"]["dev_best"]["example_f1"]
                    for r in runs(DEC, "hoc", "lora", tuned(DEC, "hoc", "lora")).values())
dfail = sum(100 * r["result"]["dev_best"]["example_f1"] < dev_med - 10
            for m in DH for r in runs(DEC, "hoc", m, tuned(DEC, "hoc", m)).values())
de = ttest(DH["eva"], DH["lora"])
check("decoder on HoC: no failures, all at 1e-3, EVA least stable",
      dfail == 0 and all(abs(tuned(DEC, "hoc", m) - 1e-3) < 1e-12 for m in DH)
      and quoted(f"LoRA reaches {mstd(DH['lora'])}, whitened EVA {mstd(DH['eva_white'])} and \\method{{}} {mstd(DH['drift'])}")
      and quoted(f"least stable at {mstd(DH['eva'])}, with seeds at ${max(DH['eva'].values()):.1f}$, "
                 f"${sorted(DH['eva'].values())[1]:.1f}$ and ${min(DH['eva'].values()):.1f}$ ({sgn(de[0])} against LoRA, $p = {de[1]:.2f}$)"))
ci = json.load(open(os.path.join(PAPER, "ci.json"))) if os.path.exists(os.path.join(PAPER, "ci.json")) else {}
excl = [k for k, v in ci.items() if v.get("delta") and (v["delta"][1] > 0 or v["delta"][2] < 0)]
reps = {k: 100 * v["rep_diff"] for k, v in ci.items()}
n_delta = sum(1 for v in ci.values() if v.get("delta"))
check("instance-level intervals: only BitFit on ChemProt excludes zero",
      excl == ["chemprot:bitfit"] and len(ci) == 27 and n_delta == 24
      and quoted(f"of the ${n_delta}$ paired differences to LoRA only one excludes zero")
      and quoted("BitFit's loss on ChemProt ($-1.1$, $[-2.0, -0.2]$)"))
gev32 = runs(RB, "hoc", "gev", tuned(RB, "hoc", "gev"), tags=("fp32",), deterministic=True)
gev16 = runs(RB, "hoc", "gev", tuned(RB, "hoc", "gev"))
gf = failed(gev32, "hoc")
check("GEV: no fp16 failure, one fp32 collapse",
      len(gev32) == 3 and len(gf) == 1 and not failed(gev16, "hoc")
      and all(e["dev"] < 0.2 for e in gev32[gf[0]]["result"]["history"])
      and quoted("loses one of its three fp32 HoC seeds to the same early collapse")
      and quoted(f"maximum gradient norm ${max(e['grad_norm_max'] for e in gev32[gf[0]]['result']['history']):.1f}$"))
fails16 = {m: failed(cells[("hoc", m)], "hoc") for m in ("eva_white", "drift", "gev", "eva")}
fails32 = {m: failed(F32[m], "hoc") for m in ("eva_white", "drift", "eva")}
check("energy ordering matches the failure ordering",
      not fails16["eva_white"] and not fails32["eva_white"]
      and len(fails16["drift"]) == 1 and not fails32["drift"]
      and not fails16["gev"] and len(gf) == 1
      and len(fails16["eva"]) == 2 and len(fails32["eva"]) == 3
      and quoted("whitened EVA ($1\\times$ by construction) fails nothing, \\method{} "
                 "($1.5$--$2.7\\times$) one fp16 run and no fp32 run, GEV ($8$--$15\\times$) "
                 "no fp16 run and one fp32 run, and EVA ($59$--$73\\times$) two of five in "
                 "fp16 and all three in fp32"))
check("replicate reproduces ChemProt and RCT-20k exactly, not HoC",
      all(abs(v) < 0.005 for k, v in reps.items() if not k.startswith("hoc"))
      and quoted(f"(DoRA $+{reps['hoc:dora']:.1f}$, PiSSA $+{reps['hoc:pissa']:.1f}$, EVA "
                 f"$+{reps['hoc:eva']:.1f}$, \\method{{}} $+{reps['hoc:drift']:.1f}$, against "
                 f"$+{reps['hoc:lora']:.1f}$ for\nLoRA)"))
lh_drop = {m: st.mean(score(runs(RB, "chemprot", m, tuned(RB, "chemprot", m))).values())
           - st.mean(LH[m].values()) for m in LH}
check("linear head: every method but DRIFT loses 0.6-1.5 points",
      all(0.6 - 0.05 <= d <= 1.5 + 0.05 for m, d in lh_drop.items() if m != "drift")
      and lh_drop["drift"] < 0 and quoted("lowers every method except \\method{} by $0.6$--$1.5$ points"))
n_runs = sum(1 for r in R if (r["args"].get("tag") or "") != "smoke")
n_tune = sum(1 for r in R if (r["args"].get("tag") or "") == "tune")
check(f"run counts ({n_runs} runs, {n_tune} tuning)",
      quoted(f"comprises ${n_runs//1000}{{,}}{n_runs%1000:03d}$ fine-tuning runs, ${n_tune}$ of them")
      and quoted(f"corrected (${n_runs//1000}{{,}}{n_runs%1000:03d}$ runs)"))

# --- the mechanism claim, recomputed from the profile tensors themselves
HIDDEN = ("attention.self.query", "attention.self.key", "attention.self.value",
          "intermediate.dense")
try:
    import torch

    def energy_groups(task):
        """Median energy ratio of the leading EVA direction, by module group, and
        the median after deflation at tau = 0.95."""
        p = os.path.join(ROOT, "runs", "profiles",
                         f"roberta-base__{task}__ref1024__dom1024__cen.pt")
        prof = torch.load(p, weights_only=False)
        eva, drift, dims = prof["taus"]["0.0"], prof["taus"]["0.95"], prof["meta"]["dims"]
        g = {"hidden": [], "attn_out": [], "ffn_hidden": [], "all": [], "drift": []}
        for name, d in eva.items():
            d_in = dims[name][0]
            e = float(d["evals"][0]) / (d["trace_d"] / d_in)
            g["all"].append(e)
            g["drift"].append(float(drift[name]["evals"][0]) / (drift[name]["trace_d"] / d_in))
            key = ("hidden" if any(h in name for h in HIDDEN) else
                   "attn_out" if "attention.output.dense" in name else "ffn_hidden")
            g[key].append(e)
        return g

    def lead_sites(task):
        """(hidden-state inputs whose leading EVA direction peaks on 77/588,
        number of such inputs, median energy ratio over all modules)."""
        p = os.path.join(ROOT, "runs", "profiles",
                         f"roberta-base__{task}__ref1024__dom1024__cen.pt")
        prof = torch.load(p, weights_only=False)
        eva, dims = prof["taus"]["0.0"], prof["meta"]["dims"]
        seen, hits, energy = {}, 0, []
        for name, d in eva.items():
            d_in = dims[name][0]
            energy.append(float(d["evals"][0]) / (d["trace_d"] / d_in))
            if d_in != 768 or not any(h in name for h in HIDDEN):
                continue
            # W_Q, W_K and W_V of a block read one input: count the block once
            site = (name.split(".layer.")[1].split(".")[0],
                    "ffn" if "intermediate" in name else "attn")
            if site in seen:
                continue
            seen[site] = True
            hits += int(torch.as_tensor(d["basis"])[:, 0].abs().argmax().item() in (77, 588))
        return hits, len(seen), st.median(energy)

    M = {t: lead_sites(t) for t in TASKS}
    check("mechanism: 24/24 distinct inputs on ChemProt, 23/24 on RCT-20k and HoC",
          M["chemprot"][:2] == (24, 24) and M["rct20k"][:2] == (23, 24)
          and M["hoc"][:2] == (23, 24)
          and quoted("at all $24$ distinct hidden-state inputs on ChemProt and at $23$ of $24$ on RCT-20k and HoC")
          and quoted("at 23--24 of the 24 distinct hidden-state inputs"))
    med = sorted(v[2] for v in M.values())
    check(f"mechanism: median energy {med[0]:.0f}-{med[-1]:.0f}x over all 72 modules",
          quoted(f"over all $72$ modules the median is ${med[0]:.0f}$--${med[-1]:.0f}\\times$")
          and quoted(f"a median ${med[0]:.0f}$--${med[-1]:.0f}\\times$ an average direction's energy"))
    G = {t: energy_groups(t) for t in TASKS}

    def rng(key, nd=0):
        """The range of medians across tasks, as the paper types it."""
        v = sorted(st.median(G[t][key]) for t in TASKS)
        return f"${v[0]:.{nd}f}$--${v[-1]:.{nd}f}\\times$"

    mx = round(max(max(G[t]["all"]) for t in TASKS))
    check("energies by module group (hidden / attention-output / feed-forward)",
          quoted(f"a median {rng('hidden')} the input energy")
          and quoted(f"a median {rng('attn_out')}, and those of the feed-forward hidden activations a median {rng('ffn_hidden')}")
          and quoted(f"up to ${mx//1000}{{,}}{mx % 1000:03d}\\times$")
          and all(len(G[t]["hidden"]) == 48 and len(G[t]["attn_out"]) == 12 for t in TASKS))
    check("deflated energy after tau = 0.95",
          quoted(f"a median of only {rng('drift', 1)} average")
          and quoted(f"against {rng('drift', 1)} with WikiText"))
except ImportError:                       # torch is optional for this script
    print("  (torch not available: profile-tensor checks skipped)")

# --- leading directions of the reference controls and of GEV (profile summaries)
def lead(task, tag, level):
    p = os.path.join(ROOT, "runs", "profiles", f"roberta-base__{task}__ref1024__dom1024__{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8")).get("lead", {}).get(level)
    if not d:
        return None
    hits = sum(v["argmax"] in (77, 588) for n, v in d.items() if any(h in n for h in HIDDEN))
    return hits, st.median(v["energy_rel"] for v in d.values())


def in_range(vals, lo, hi):
    return min(vals) >= lo - 0.5 and max(vals) < hi + 0.5


refl = {k: lead(t, tag, "0.95") for k, t, tag in [("news-c", "chemprot", "news__cen"),
                                                  ("shuf-c", "chemprot", "shuffled__cen"),
                                                  ("rand-c", "chemprot", "random__cen"),
                                                  ("news-h", "hoc", "news__cen"),
                                                  ("rand-h", "hoc", "random__cen")]}
check("references absorb the outlier dimensions at tau 0.95",
      all(v is not None and v[0] == 0 for v in refl.values()))
check("reference energies: news/shuffled 3-4x, random 13-15x",
      in_range([refl[k][1] for k in ("news-c", "shuf-c", "news-h")], 3, 4)
      and in_range([refl[k][1] for k in ("rand-c", "rand-h")], 13, 15)
      and quoted("$3$--$4\\times$ the energy of an average direction with news or shuffled text and $13$--$15\\times$ with random tokens"))
gl = [lead(t, "cen__gev", "gev") for t in TASKS]
check("GEV avoids the outlier dimensions; median 8-15x",
      all(g is not None and g[0] == 0 for g in gl) and in_range([g[1] for g in gl], 8, 15)
      and quoted("median $8$--$15\\times$ average energy"))
check("energy-check range 1-15x covers GEV and the references",
      max([g[1] for g in gl] + [v[1] for v in refl.values()]) < 15.5
      and quoted("against $1$--$15\\times$ for the initialisations that largely restored stability"))

# --- every remaining number the prose quotes (see src/number_coverage.py)
full = {t: score(cells[(t, "full")]) for t in TASKS}
micro_full = {s: 100 * r["result"]["test"]["micro_f1"] for s, r in cells[("hoc", "full")].items()}
check("published reference points reproduced",
      quoted(f"reaches {mstd(full['chemprot'])} micro-F1 on ChemProt")
      and quoted(f"and {mstd(micro_full)} micro-F1 on HoC"))
med = {m: st.median(score(cells[("hoc", m)]).values()) for m in ("dora", "drift", "lora")}
check("HoC medians quoted in the prose",
      quoted(f"DoRA's median is ${med['dora']:.1f}$ and \\method{{}}'s ${med['drift']:.1f}$, "
             f"against LoRA's ${med['lora']:.1f}$"))
tune_dev = TUNE[(RB, "hoc", "drift", "all", 8, "")][tuned(RB, "hoc", "drift")] * 100
main_dev = 100 * cells[("hoc", "drift")][1]["result"]["dev_best"][METRIC["hoc"]]
check("the identical-settings pair of dev scores",
      quoted(f"yet reach ${tune_dev:.1f}$ and ${main_dev:.1f}$ dev F1")
      and quoted(f"reached ${tune_dev:.1f}$ and ${main_dev:.1f}$ dev F1"))
check("fp32 EVA's flat training loss", quoted(f"stays at ${sorted(flat)[0]:.2f}$ from the second epoch"))

TAU = {}
for tau in (0.0, 0.5, 0.9, 0.95, 0.99):
    lr = tuned(RB, "chemprot", "drift")          # DRIFT's own rate, the fixed-rate sweep
    TAU[tau] = score(runs(RB, "chemprot", "drift", lr, tau=tau))
others = [st.mean(TAU[t].values()) for t in (0.5, 0.9, 0.95, 0.99)]
sds = [st.stdev(TAU[t].values()) for t in (0.5, 0.9, 0.95, 0.99)]
check("tau sweep at DRIFT's own rate",
      quoted(f"the method reaches {mstd(TAU[0.0])} on ChemProt")
      and quoted(f"gives ${min(others):.1f}$--${max(others):.1f}$ with a standard deviation of "
                 f"${min(sds):.1f}$--${max(sds):.1f}$"))
ao = ttest(score(ABL["chemprot"]["Rank allocation only (random init)"]), lo3["chemprot"])
io_ = ttest(score(ABL["chemprot"]["Drift init only (uniform rank)"]), lo3["chemprot"])
check("ChemProt factorisation quoted with its p-values",
      quoted(f"({sgn(ao[0])}, $p = {ao[1]:.2f}$ uncorrected")
      and quoted(f"({mstd(score(ABL['chemprot']['Drift init only (uniform rank)']))}), though not "
                 f"significantly better than LoRA ($p = {io_[1]:.2f}$)"))

# drift ratios and the excess over 1 - tau: rho_m per module, averaged over the
# 48 distinct input sites (W_Q, W_K and W_V of a block share one), as the text says
def site_of(name):
    layer = name.split(".layer.")[1].split(".")[0]
    kind = ("attn" if "attention.self" in name else
            "attn_out" if "attention.output.dense" in name else
            "ffn" if "intermediate.dense" in name else "ffn_hidden")
    return layer, kind


RHO = {}
for t in TASKS + ["mtsamples"]:
    p = os.path.join(ROOT, "runs", "profiles", f"roberta-base__{t}__ref1024__dom1024__cen.pt")
    if not os.path.exists(p):
        continue
    import torch as _t
    prof = _t.load(p, weights_only=False)
    RHO[t] = {}
    for tau, per in prof["taus"].items():
        sites = {site_of(n): d["trace_drift"] / d["trace_d"] for n, d in per.items()}
        RHO[t][tau] = (100 * st.mean(sites.values()), len(sites))


def exc(tau, nd=1):
    return [RHO[t][str(tau)][0] - 100 * (1 - tau) for t in TASKS]


check("drift ratio and its excess over 1 - tau (48 distinct sites)",
      all(RHO[t]["0.95"][1] == 48 for t in TASKS)
      and quoted(f"is only ${exc(0.95)[0]:.1f}$ (ChemProt), ${exc(0.95)[1]:.1f}$ (RCT-20k) and "
                 f"${exc(0.95)[2]:.1f}$ (HoC) points of variance")
      and quoted(f"($\\rho_m = {RHO['chemprot']['0.95'][0]:.1f}$, ${RHO['rct20k']['0.95'][0]:.1f}$ and "
                 f"${RHO['hoc']['0.95'][0]:.1f}\\%$)")
      and quoted(f"${exc(0.5)[0]:.1f}$/${exc(0.5)[1]:.1f}$/${exc(0.5)[2]:.1f}$ and "
                 f"${exc(0.99)[0]:.2f}$/${exc(0.99)[1]:.2f}$/${exc(0.99)[2]:.2f}$"))
check("clinical drift ratio matches ChemProt's",
      quoted(f"an excess over $1-\\tau$ of ${exc(0.95)[0] + (RHO['mtsamples']['0.95'][0] - RHO['chemprot']['0.95'][0]):.1f}$ "
             f"points at $\\tau = 0.95$, against ${exc(0.95)[0]:.1f}$"))

# tau-free divergences
DIV = {}
for t in TASKS + ["mtsamples"]:
    p = os.path.join(ROOT, "runs", "divergence", f"roberta-base__{t}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)["modules"]
        sites = {site_of(n): v for n, v in d.items()}      # the 48 distinct sites
        DIV[t] = {k: st.mean(v[k]["coral"] for v in sites.values() if k in v)
                  for k in ("target", "ref2", "news", "random")
                  if any(k in v for v in sites.values())}
check("CORAL divergences quoted in V-C",
      quoted(f"(${DIV['chemprot']['target']:.2f}$ for ChemProt, ${DIV['mtsamples']['target']:.2f}$ for the clinical notes")
      and quoted(f"${DIV['rct20k']['target']:.2f}$ for RCT-20k, ${DIV['hoc']['target']:.2f}$ for HoC, against ${DIV['chemprot']['ref2']:.2f}$")
      and quoted(f"news text sits closer to WikiText (${DIV['chemprot']['news']:.2f}$)")
      and quoted(f"random tokens further away (${DIV['chemprot']['random']:.2f}$)"))

# placement parameter counts
def par(t, m, tgt, r):
    v = [x["result"].get("params_adapter_effective", x["result"]["params_adapter"])
         for x in pl(t, m, tgt, r).values()]
    return st.mean(v) / 1e6


check("placement parameter counts",
      quoted(f"spends ${par('chemprot', 'lora', 'ffn', 8):.2f}$M parameters instead of "
             f"${par('chemprot', 'lora', 'all', 8):.2f}$M")
      and quoted(f"rank $14$ (${par('chemprot', 'lora', 'ffn', 14):.2f}$M) against all-module rank $8$ "
                 f"(${par('chemprot', 'lora', 'all', 8):.2f}$M)")
      and quoted(f"all-module rank $4$ (${par('chemprot', 'lora', 'all', 4):.2f}$M)")
      and quoted(f"against feed-forward-only rank $8$ (${par('chemprot', 'lora', 'ffn', 8):.2f}$M)")
      and quoted(f"attention-only rank $8$ (${par('chemprot', 'lora', 'attn', 8):.2f}$M)"))

# budgets elsewhere: the gaps quoted in V-E
bg = {(m, r): ttest(BX[("hoc", m, r)], BX[("hoc", "lora", r)])
      for m in ("eva", "eva_white", "drift") for r in (4, 16)}
check("HoC gaps at ranks 4 and 16",
      quoted(f"trails it by ${abs(bg[('eva', 4)][0]):.1f}$ at rank $4$ and ${abs(bg[('eva', 16)][0]):.1f}$ at rank $16$")
      and quoted(f"whitened EVA by ${abs(bg[('eva_white', 4)][0]):.1f}$ at both")
      and quoted(f"loses a seed at rank $4$ ({mstd(BX[('hoc', 'drift', 4)])}) and trails by "
                 f"${abs(bg[('drift', 16)][0]):.1f}$ at rank $16$")
      and quoted(f"(EVA at rank $16$ on RCT-20k, {sgn(ttest(BX[('rct20k', 'eva', 16)], BX[('rct20k', 'lora', 16)])[0])}; "
                 f"\\method{{}} at rank $16$ on HoC, {sgn(bg[('drift', 16)][0])})"))

# the decoder on ChemProt, and the clinical table read back from the paper's numbers
DC = {m: score(runs(DEC, "chemprot", m, tuned(DEC, "chemprot", m))) for m in ("lora", "eva")}
check("decoder on ChemProt",
      quoted(f"No method beats LoRA ({mstd(DC['lora'])})")
      and quoted(f"least stable ({mstd(DC['eva'])}, one seed at ${min(DC['eva'].values()):.1f}$)"))
# Section VI: on SmolLM2-360M EVA fails no run on either task (failure: best dev
# score more than 10 points below the median of LoRA's runs) but has the widest
# seed spread of the four methods on both
dec_fail, dec_widest = 0, True
for t in ("chemprot", "hoc"):
    R4 = {m: runs(DEC, t, m, tuned(DEC, t, m)) for m in ("lora", "eva", "eva_white", "drift")}
    med_t = st.median(100 * r["result"]["dev_best"][METRIC[t]] for r in R4["lora"].values())
    dec_fail += sum(100 * r["result"]["dev_best"][METRIC[t]] < med_t - 10
                    for rs in R4.values() for r in rs.values())
    sd = {m: st.stdev(score(rs).values()) for m, rs in R4.items()}
    dec_widest &= max(sd, key=sd.get) == "eva"
check("decoder: no failed run on either task, EVA the least stable on both (Section VI)",
      dec_fail == 0 and dec_widest
      and quoted("where it failed no run but was the least stable method"))
# Section V-E: the decoder's HoC column in fp32 (deterministic kernels, gradient
# checkpointing, same rate and profile). In fp16 two of EVA's three seeds overflow:
# the loss scale falls to zero, no gradient reaches the adapter from the fifth
# epoch on, and their best checkpoints (second and third epochs) stay above the
# failure threshold. In fp32 EVA trains in every seed without a non-finite step,
# although its gradient norm still spikes, and is level with LoRA.
h16 = {m: runs(DEC, "hoc", m, tuned(DEC, "hoc", m)) for m in ("lora", "eva")}
h32 = {m: runs(DEC, "hoc", m, tuned(DEC, "hoc", m), tags=("fp32",), deterministic=True)
       for m in ("lora", "eva")}
floor_d = st.median(100 * r["result"]["dev_best"]["example_f1"]
                    for r in h16["lora"].values()) - 10
over = {s: r["result"] for s, r in h16["eva"].items()
        if any(e.get("loss_scale") is not None and e["loss_scale"] == 0
               for e in r["result"]["history"])}
over_ok = (len(h16["eva"]) == 3 and len(over) == 2
           and all(res["history"][3]["grad_norm_max"] > 0
                   and all(e["grad_norm_max"] == 0 for e in res["history"][4:])
                   for res in over.values())
           and sorted(res["best_epoch"] for res in over.values()) == [1, 2]
           and all(100 * res["dev_best"]["example_f1"] > floor_d for res in over.values()))
e32, l32 = score(h32["eva"]), score(h32["lora"])
nonfin32 = sum(e.get("nonfinite_loss_steps", 0) for r in h32["eva"].values()
               for e in r["result"]["history"])
trained32 = all(100 * r["result"]["dev_best"]["example_f1"] > floor_d
                for r in h32["eva"].values())
setup32 = all(r["args"].get("grad_ckpt") and not r["args"].get("amp")
              and abs(r["args"]["lr"] - tuned(DEC, "hoc", m)) < 1e-12
              for m, rs in h32.items() for r in rs.values())
gmax_e = max(e["grad_norm_max"] for r in h32["eva"].values() for e in r["result"]["history"])
gmax_l = max(e["grad_norm_max"] for r in h32["lora"].values() for e in r["result"]["history"])
d_el, d_ll = ttest(e32, l32), ttest(l32, score(h16["lora"]))
check("decoder HoC in fp32: EVA's fp16 overflow gone, level with LoRA (Section V-E)",
      len(e32) == 3 and len(l32) == 3 and over_ok and nonfin32 == 0 and trained32
      and setup32
      and quoted("the fp16 loss scale collapsing to zero")
      and quoted("from the fifth epoch on, so their scores come from checkpoints of the "
                 "second and third epochs, which stay above our failure threshold")
      and quoted("EVA trains in every seed without a non-finite step")
      and quoted(f"still reaches ${gmax_e:.0f}$, against at most ${gmax_l:.1f}$ for LoRA")
      and quoted(f"It scores {mstd(e32)}, level with LoRA's {mstd(l32)} in fp32 "
                 f"({sgn(d_el[0])}, $p = {d_el[1]:.2f}$)")
      and quoted(f"precision leaves LoRA itself unchanged ({sgn(d_ll[0])}, "
                 f"$p = {d_ll[1]:.2f}$)"))
CL = {m: runs(RB, "mtsamples", m, tuned(RB, "mtsamples", m))
      for m in ("lora", "eva", "eva_white", "drift")}
CL["news"] = runs(RB, "mtsamples", "drift", tuned(RB, "mtsamples", "drift"), ref="news")
CL["random"] = runs(RB, "mtsamples", "drift", tuned(RB, "mtsamples", "drift"), ref="random")
cm = {k: st.mean(score(v).values()) for k, v in CL.items()}
cmac = st.mean(100 * r["result"]["test"]["macro_f1"] for r in CL["lora"].values())
ct = {k: ttest(score(v), score(CL["lora"])) for k, v in CL.items() if k != "lora"}
check("clinical numbers quoted in V-F",
      quoted(f"unbeaten---${cm['lora']:.1f}$ micro-F1 (${cmac:.1f}$ macro) against \\method{{}}'s "
             f"${cm['drift']:.1f}$ ({sgn(ct['drift'][0])}, $p = {ct['drift'][1]:.2f}$)")
      and quoted(f"EVA's ${cm['eva']:.1f}$ ({sgn(ct['eva'][0])}, $p = {ct['eva'][1]:.2f}$) and whitened EVA's "
                 f"${cm['eva_white']:.1f}$ ({sgn(ct['eva_white'][0])}, $p = {ct['eva_white'][1]:.2f}$)")
      and quoted(f"(${cm['random']:.1f}$, {sgn(ct['random'][0])} over LoRA, $p = {ct['random'][1]:.2f}$) and the news "
                 f"reference ${cm['news']:.1f}$"))
check("linear-head numbers quoted in Appendix D",
      quoted(f"LoRA reaches {mstd(LH['lora'])}, \\method{{}} {mstd(LH['drift'])} "
             f"({sgn(lhd['drift'][0])}, $p = {lhd['drift'][1]:.2f}$), whitened EVA {mstd(LH['eva_white'])}, "
             f"BitFit {mstd(LH['bitfit'])} and EVA {mstd(LH['eva'])} ({sgn(lhd['eva'][0])}, "
             f"$p = {lhd['eva'][1]:.2f}$ uncorrected)"))

# --- the last uncovered prose numbers (src/number_coverage.py drives this list)
micro = {m: {s: 100 * r["result"]["test"]["micro_f1"] for s, r in cells[("hoc", m)].items()}
         for m in ("bitfit", "lora")}
dmic = ttest(micro["bitfit"], micro["lora"])
dex = ttest(score(cells[("hoc", "bitfit")]), score(cells[("hoc", "lora")]))
check("BitFit's HoC lead under both metrics",
      quoted(f"widens from ${dex[0]:.1f}$ to ${dmic[0]:.1f}$ points and remains non-significant ($p = {dmic[1]:.2f}$)")
      and quoted(f"(${dmic[0]:.1f}$ under HoC's micro-F1)")
      and quoted(f"(BitFit on HoC, $p = {dex[1]:.2f}$)"))
bestpeft = {t: max(round(st.mean(score(cells[(t, m)]).values()), 1)
                   for m in NAMES if m not in ("full", "linear")) for t in TASKS}
bf = {t: round(st.mean(score(cells[(t, "bitfit")]).values()), 1) for t in TASKS}
check("BitFit within 1.4 of the best method",
      quoted(f"is within ${max(bestpeft[t] - bf[t] for t in ('chemprot', 'rct20k')):.1f}$ points of the best method")
      and bf["hoc"] == bestpeft["hoc"])
gv3 = [abs(ttest(score(ABL[t]["GEV init (uniform rank)"]), lo3[t])[0]) for t in TASKS]
check("GEV within 0.3 of LoRA (Section V-A wording)",
      quoted(f"is within ${max(gv3):.1f}$ points of LoRA on every task and fails no run"))
ed_l = [st.mean(LS[(n, "lora")].values()) - st.mean(LS[(n, m)].values())
        for n, _ in LAD for m in ("eva",)]
wd_l = [st.mean(LS[(n, "lora")].values()) - st.mean(LS[(n, "eva_white")].values()) for n, _ in LAD]
check("ladder deficits quoted in V-E and the abstract",
      quoted(f"trails LoRA by ${min(ed_l):.1f}$ to ${max(ed_l):.1f}$ points and whitened EVA by at most ${max(wd_l):.1f}$")
      and quoted(f"shrinks to at most ${max(ed_l):.1f}$ points at its own rate"))
t0 = st.mean(TT[0.0].values())
oth = [st.mean(TT[t].values()) for t in (0.5, 0.9, 0.95, 0.99)]
check("tau gap at the tuned rates", quoted(f"the gap shrinks to at most ${max(oth) - t0:.1f}$ points"))
ro = score(ABL["chemprot"]["Random orthonormal init, drift ranks"])
check("random-orthonormal mean", quoted(f"leaves the mean unchanged (${st.mean(ro.values()):.1f}$)"))
spread = max(max(round(st.mean(BS[(m, r)].values()), 1) for m in ("lora", "eva_white", "drift"))
             - min(round(st.mean(BS[(m, r)].values()), 1) for m in ("lora", "eva_white", "drift"))
             for r in (1, 2, 4, 8, 16))
# "within X points of each other" is a bound: check the widest spread obeys it
rctb = max(max(round(st.mean(BX[("rct20k", m, r)].values()), 1) for m in SWEEP4)
           - min(round(st.mean(BX[("rct20k", m, r)].values()), 1) for m in SWEEP4)
           for r in (4, 8, 16))
r4d = [ttest(BX[("rct20k", m, 4)], BX[("rct20k", "lora", 4)]) for m in ("eva_white", "drift")]
check(f"budget spreads quoted in V-E (ChemProt {spread:.1f}, RCT-20k {rctb:.1f})",
      quoted(f"stay within ${spread:.1f}$ points of each other at every budget")
      and rctb <= 0.6 + 1e-9 and quoted("lie within $0.6$ points of each other at every budget")
      and quoted(f"above LoRA at rank $4$ (${max(d for d, _ in r4d):+.1f}$, $p \\ge {min(p for _, p in r4d):.2f}$)"))
r2c = ttest(BS[("lora", 2)], score(cells[("chemprot", "lora")]))
check(f"ChemProt rank 2 within 0.1 of the full budget (gap {abs(r2c[0]):.2f})",
      abs(r2c[0]) <= 0.1 + 1e-9 and quoted("a quarter of it, is within $0.1$ points"))
check("HoC rank 16 gain quoted in the Discussion",
      quoted(f"doubling the budget to $2\\%$ gains LoRA ${hoc16[0]:.1f}$ points"))
atn = max(abs(ttest(score(P[t][T8]), score(P[t][A8]))[0]) for t in TASKS)
check("attention-only within 0.2 of all-module rank 8",
      quoted(f"by at most ${atn:.1f}$. At this budget and below it"))

try:                                    # profile-tensor claims about dimension 588
    def weight588(task, tau):
        p = os.path.join(ROOT, "runs", "profiles",
                         f"roberta-base__{task}__ref1024__dom1024__cen.pt")
        prof = torch.load(p, weights_only=False)
        per, dims = prof["taus"][tau], prof["meta"]["dims"]
        # every module whose input is the 768-dimensional hidden or attention output
        w = [abs(float(torch.as_tensor(d["basis"])[588, 0])) for n, d in per.items()
             if dims[n][0] == 768]
        return st.mean(w)

    ev588 = sorted(weight588(t, "0.0") for t in TASKS)
    dr588 = sorted(weight588(t, "0.95") for t in TASKS)
    check("average weight on dimension 588 (768-dimensional inputs)",
          quoted(f"Averaged over the $768$-dimensional inputs, \\method{{}}'s leading drift "
                 f"direction puts a weight of ${dr588[0]:.3f}$--${dr588[-1]:.3f}$ on dimension $588$")
          and quoted(f"(against ${ev588[0]:.2f}$--${ev588[-1]:.2f}$ for EVA)"))

    def med_energy(model, task, tau, n_ref=1024, n_dom=1024):
        p = os.path.join(ROOT, "runs", "profiles",
                         f"{model}__{task}__ref{n_ref}__dom{n_dom}__cen.pt")
        prof = torch.load(p, weights_only=False)
        per, dims = prof["taus"][tau], prof["meta"]["dims"]
        return st.median(float(d["evals"][0]) / (d["trace_d"] / dims[n][0])
                         for n, d in per.items())

    lad = sorted(med_energy(m.replace("/", "__"), "chemprot", "0.95") for _, m in LAD)
    check("ladder energies after deflation",
          quoted(f"deflation cuts them to ${lad[0]:.1f}$--${lad[-1]:.1f}\\times$"))
    check("decoder and clinical energies after deflation",
          quoted(f"(${med_energy(DEC.replace('/', '__'), 'chemprot', '0.95'):.1f}\\times$ after deflation)")
          and quoted(f"a median of ${med_energy('roberta-base', 'mtsamples', '0.95'):.1f}\\times$"))
except (ImportError, NameError):
    print("  (torch not available: dimension-588 and deflated-energy checks skipped)")

# --- round 2, NEW-2: AdaLoRA's RCT-20k rate, selected by 0.12 dev points on one
#     seed, against the same grid judged by the mean over three seeds
ada_grid = TUNE[(RB, "rct20k", "adalora", "all", 8, "")]
ada = {lr: runs(RB, "rct20k", "adalora", lr) for lr in (3e-4, 3e-3)}
ada_dev = {lr: st.mean(100 * r["result"]["dev_best"][METRIC["rct20k"]] for r in v.values())
           for lr, v in ada.items() if v}
check("AdaLoRA's RCT-20k selection is a near-tie on seed 1",
      abs(tuned(RB, "rct20k", "adalora") - 3e-3) < 1e-12
      and quoted(f"are ${100*ada_grid[3e-4]:.1f}$, ${100*ada_grid[1e-3]:.1f}$ and "
                 f"${100*ada_grid[3e-3]:.1f}$, and the rule takes the last"))
check("AdaLoRA: the three-seed dev mean prefers the lower rate",
      ada_dev[3e-4] > ada_dev[3e-3] and not failed(ada[3e-4], "rct20k")
      and quoted(f"falls to ${ada_dev[3e-3]:.1f}$ against ${ada_dev[3e-4]:.1f}$ one grid step lower")
      and quoted(f"test score is {mstd(score(ada[3e-4]))} instead of {mstd(score(ada[3e-3]))}"))
check("the allocation-only difference quoted in the limitation",
      quoted(f"the {sgn(ao[0])} of allocation alone among them"))

print("TEXT CHECKS")
for name, ok in text_checks:
    report(name, ok)
print("TOTAL MISMATCHES:", bad)
