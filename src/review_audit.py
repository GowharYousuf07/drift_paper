"""Check every item of the round-1 review package against the revised sources.

Each entry names a demand from peer_review.docx and the evidence that must be
present in the manuscript, the generated tables, the bibliography or the response
letter. Run after any edit:

    python src/review_audit.py > paper/review_audit.txt
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def load(name):
    p = os.path.join(PAPER, name)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return " ".join(f.read().split())


MAIN = load("main.tex")
RESP = load("response.tex")
RESP2 = load("response2.tex")
BIB = load("refs.bib")
TAB_MAIN = load("tab_main.tex")
TAB_LR = load("tab_lr.tex")
TAB_ABL = load("tab_ablation.tex")
TAB_PLACE = load("tab_placement.tex")
TAB_FP32 = load("tab_fp32.tex")
TAB_CLIN = load("tab_clinical.tex")
TAB_DEC = load("tab_decoder.tex")
TAB_CI = load("tab_ci.tex")
TAB_LIN = load("tab_linhead.tex")
TAB_DIV = load("tab_div.tex")
TAB_BX = load("tab_budgetx.tex")
TAB_SEEDS = load("tab_seeds.tex")

# (id, demand, [(text, evidence), ...]) - every evidence string must be present
ITEMS = [
    # ---------------------------------------------------------------- EIC
    ("EIC W1", "Commit to the evaluation-paper framing; DRIFT as instrument; "
               "'near-optimally' becomes a statement about the allocation objective", [
        (MAIN, "This is an evaluation study"),
        (MAIN, "useful as an instrument even where it is not useful as a method"),
        (MAIN, "allocates the budget near-optimally \\emph{for captured drift}"),
        (MAIN, "Section~\\ref{sec:ablation} finds that it does not")]),
    ("EIC W2", "Budget-matched placement study on all three tasks, own subsection and table", [
        (MAIN, "\\subsection{RQ3: where should the adapter go?}"),
        (MAIN, "\\noindent\\textbf{At matched budgets.}"),
        (TAB_PLACE, "Feed-forward only & 14"),
        (TAB_PLACE, "All modules & 4")]),
    ("EIC W3", "Abstract: stability wording, 0.34 (1.0 micro-F1), 23-24 of 24 distinct inputs", [
        (MAIN, "largely restore stability (whitening most"),
        (MAIN, "$0.34$ F1 ($1.0$ under HoC's micro-F1)"),
        (MAIN, "23--24 of the 24 distinct hidden-state inputs")]),
    ("EIC W4", "Code/data statement; per-seed supplement; author block (authors')", [
        (MAIN, "\\noindent\\textbf{Availability.}"),
        (MAIN, "\\appendices"),
        (MAIN, "\\section{Per-seed results}"),
        (RESP, "The author block is handled by the authors")]),
    ("EIC W5", "Extend the decoder experiment to HoC", [
        (MAIN, "On HoC, where the"),
        (TAB_DEC, "HoC")]),
    ("EIC D1", "Abstract within venue guidance (<= 250 words)", [(MAIN, "\\begin{abstract}")]),
    ("EIC D2", "Revisit the clinical framing", [
        (MAIN, "\\subsection{Clinical notes}"),
        (MAIN, "\\noindent\\textbf{Clinical text and first-order shift.}")]),
    ("EIC D3", "Protocol as a boxed checklist", [(MAIN, "A protocol for evaluating adapter allocation")]),
    ("EIC D4", "Cite MiLoRA", [(MAIN, "MiLoRA~\\cite{wang2025milora}"), (BIB, "wang2025milora")]),
    # ---------------------------------------------------------------- R1
    ("R1 W1", "No-edge rule for every method; table of selected rates; mark edges", [
        (MAIN, "extended by one half-decade step past whichever edge holds the best dev score"),
        (MAIN, "\\section{Learning-rate selection}"),
        (TAB_LR, "no selection lies on an edge of its final grid"),
        (TAB_LR, "RoBERTa-base, HoC, DoRA")]),
    ("R1 W2a", "Rerun HoC in fp32", [
        (MAIN, "Rerunning the column in fp32 at the same rates"),
        (TAB_FP32, "fp32")]),
    ("R1 W2b", "Log loss-scale events and gradient norms; define divergence operationally", [
        (MAIN, "the number of optimiser steps fp16's loss scaler skipped and the final loss scale"),
        (MAIN, "counted as \\emph{failed} when its best dev score")]),
    ("R1 W2c", "Diverged/N column and median next to mean", [
        (TAB_MAIN, "HoC med."), (TAB_MAIN, "Failed")]),
    ("R1 W2d", "Drop or footnote the Mean column", [(TAB_MAIN, "Method & Adapter params & ChemProt & RCT-20k & HoC & HoC med. & Failed")]),
    ("R1 W3", "Per-seed values, instance-level bootstrap CIs, five-seed means, multiplicity, "
              "suggestive narration", [
        (TAB_SEEDS, "95\\% CI"),
        (TAB_CI, "hierarchical bootstrap"),
        (TAB_MAIN, "$^\\dagger$five seeds"),
        (MAIN, "correcting with Holm's"),
        (MAIN, "like every nominal result in this paper, as suggestive only")]),
    ("R1 W4", "State what is trained and counted; alpha, dropout, B-init, head LR; linear-head control", [
        (MAIN, "also trains the backbone's classification head"),
        (MAIN, "excluded from the budget and from every parameter count we"),
        (MAIN, "$\\alpha = 16$, no adapter dropout"),
        (MAIN, "$A$ drawn Kaiming-uniform and $B = 0$"),
        (MAIN, "\\section{A linear classification head}"),
        (TAB_LIN, "Linear head")]),
    ("R1 W5", "Secondary experiments tuned: ladder, budget sweep, tau sweep", [
        (MAIN, "every backbone of the ladder"),
        (MAIN, "now with every method tuned on every backbone"),
        (MAIN, "with the learning rate tuned at every budget"),
        (MAIN, "At the rate selected for each level")]),
    ("R1 W6", "State whether tau = 0.95 was fixed a priori", [
        (MAIN, "$\\tau = 0.95$ was fixed before any experiment")]),
    ("R1 W7", "Reconcile the rank-1 probe with the budget sweep", [
        (MAIN, "a three-epoch probe at that rate had not yet left the constant prediction"),
        (MAIN, "it trains late rather than never")]),
    ("R1 W8", "Eq. (1): factor N", [(MAIN, "\\sigma^2 d^{\\mathrm{out}} N \\, \\tr")]),
    ("R1 repro", "alpha, dropout, head, sampling, k at tau, versions, seed semantics", [
        (MAIN, "first $1{,}024$ training texts"),
        (MAIN, "drawn with a fixed seed from the $3{,}227$ such passages"),
        (MAIN, "$k_m = 421$--$604$"),
        (MAIN, "PyTorch~2.10"),
        (MAIN, "A seed fixes the")]),
    ("R1 m1", "Say n-1 and that SD at n=3 is noisy", [(MAIN, "with $n-1$; at $n = 3$ it is itself noisy")]),
    ("R1 m2", "Tie-bolding rule", [(TAB_MAIN, "ties included")]),
    ("R1 m3", "Fig. 2 shows dev as well as test", [(MAIN, "test F1 left, best dev F1 right")]),
    # ---------------------------------------------------------------- R2
    ("R2 W1", "Cite He et al. and Hu et al. 7.1; reposition RQ3", [
        (MAIN, "\\cite[Sec.~7.1]{hu2022lora}"),
        (MAIN, "He et al.~\\cite{he2022unified}"),
        (MAIN, "Our placement results test that finding"),
        (BIB, "he2022unified")]),
    ("R2 W2", "Cite rsLoRA and LoRA+; say whether Fig. 3 changes under alpha/sqrt(r)", [
        (MAIN, "rsLoRA argues that"),
        (MAIN, "LoRA+ gives $A$ and $B$"),
        (MAIN, "LoRA with $\\alpha/\\sqrt{r}$, tuned at every rank, stays within $0.4$"),
        (BIB, "kalajdzievski2023rslora"), (BIB, "hayou2024loraplus")]),
    ("R2 W3", "Rogue-dimension literature; whitening as the known remedy", [
        (BIB, "timkey2021rogue"), (BIB, "luo2021positional"),
        (BIB, "bondarenko2021quant"), (BIB, "dettmers2022int8"),
        (MAIN, "the adapter-space analogue of the standardisation remedy")]),
    ("R2 W5", "Label EVA (budget-matched); run EVA's own rank-unit rule", [
        (MAIN, "We call this variant \\emph{EVA (budget-matched)}"),
        (TAB_MAIN, "EVA (budget-matched)"),
        (MAIN, "EVA's own rank-unit rule, under which feed-forward rank"),
        (TAB_ABL, "EVA, rank-unit budget (its own rule)")]),
    ("R2 W6", "Proposition 2 as assumption; greedy bound with proof sketch", [
        (MAIN, "(A) is a modelling assumption, not a derived result"),
        (MAIN, "\\emph{Proof sketch.}")]),
    ("R2 W7", "Qualify the 'first' claim", [
        (MAIN, "to our knowledge \\method{} is the first to \\emph{allocate rank} by a contrast")]),
    ("R2 det", "SLM range; P^G_m notation; k at tau = 0.95", [
        (MAIN, "from a few million to a few hundred million"),
        (MAIN, "(below we drop the index $m$ where a single module is meant)"),
        (MAIN, "leaving at least $164$ residual dimensions per module")]),
    ("R2 min", "BioCreative VI; EVA venue; Frechet encoding", [
        (BIB, "{BioCreative VI}"), (BIB, "Explained Variance Adaptation"),
        (MAIN, "Fr\\'echet")]),
    # ---------------------------------------------------------------- R3
    ("R3 W1", "One family; run the generalized-eigenvector variant", [
        (MAIN, "\\subsection{One contrast, three approximations}"),
        (MAIN, "the criterion of common spatial patterns"),
        (MAIN, "We also run the exact criterion, \\textbf{GEV}"),
        (TAB_ABL, "GEV init (uniform rank)"),
        (BIB, "koles1990csp")]),
    ("R3 W2", "Report the excess over 1-tau and a tau-free divergence", [
        (MAIN, "the excess over $1-\\tau$ is only $2.5$"),
        (MAIN, "\\section{$\\tau$-free divergences}"),
        (TAB_DIV, "CORAL")]),
    ("R3 W3", "Clinical applicability paragraph (and a clinical task)", [
        (MAIN, "MTSamples~\\cite{mtsamples}"),
        (TAB_CLIN, "Micro-F1"),
        (MAIN, "test clinical style rather than clinical deployment")]),
    ("R3 W4", "Boxed procedure with a threshold (two-sided after round 2) and its cost", [
        (MAIN, "medians of $59$--$125\\times$ coincided with failed runs"),
        (MAIN, "costs only the target half of the profiling pass ($10$--$63$\\,s here)")]),
    ("R3 W5", "Alternative general corpus and a trivial-reference control", [
        (TAB_ABL, "\\method{}, news reference"),
        (TAB_ABL, "\\method{}, word-shuffled reference"),
        (TAB_ABL, "\\method{}, random-token reference"),
        (MAIN, "A reference made of random tokens is thus as good as real text")]),
    ("R3 det", "First-order shift; inference cost first; participation ratio; privacy", [
        (MAIN, "a mean shift between domains"),
        (MAIN, "Inference cost does not separate the methods we compare"),
        (MAIN, "spectral counterpart of a small participation ratio"),
        (MAIN, "privacy-positive")]),
    ("R3 min", "Fig. 1 markers; Fig. 3 axis as ranks", [
        (MAIN, "Open triangles: \\method{} after deflation; crosses: GEV"),
        (MAIN, "adapter budget (uniform-equivalent rank)")]),
    # ---------------------------------------------------------------- Devil's advocate
    ("DA M1", "Own-standard violation removed in ladder, budget sweep and tau sweep", [
        (MAIN, "now with every method tuned on every backbone"),
        (MAIN, "every (method, budget) pair at its own"),
        (MAIN, "at the rate selected for each level (open)")]),
    ("DA M2", "Placement budget-matched, with the regularisation alternative tested", [
        (MAIN, "all-module rank $4$ ($0.66$M) against feed-forward-only rank $8$"),
        (MAIN, "tests the alternative explanation that feed-forward-only LoRA")]),
    ("DA M3", "Abstract matches the data", [
        (MAIN, "none beats a tuned uniform LoRA by more than $0.34$ F1"),
        (MAIN, "($1.0$ under HoC's micro-F1)")]),
    ("DA M4", "Trivial-reference control; 'near-optimal' tied to the objective", [
        (MAIN, "what the contrast contributes is the backbone's outlier structure"),
        (MAIN, "near-optimally \\emph{for captured drift}")]),
    ("DA M5", "Mechanism scope stated where it does not explain the result", [
        (MAIN, "must therefore have another cause, which these experiments do not isolate")]),
    ("DA M6", "23-24 of 24 distinct inputs, not 45-48 of 72", [
        (MAIN, "at all $24$ distinct hidden-state inputs on ChemProt and at $23$ of $24$")]),
    ("DA counter", "The positive placement claim is withdrawn, not reframed", [
        (MAIN, "is therefore not a placement effect that survives budget matching"),
        (RESP, "we withdrew the claim")]),
    ("DA m1", "Rank-1 evidence consistent", [
        (MAIN, "at $10^{-4}$ its rank-$1$ adapter reaches $78.2\\pm2.0$")]),
    ("DA m3", "SLM range matches the 4.4M ladder member", [
        (MAIN, "from a few million to a few hundred million")]),
    ("DA m4", "Ties bolded together", [(TAB_MAIN, "ties included")]),
    ("DA m5", "Mean column removed", [(TAB_MAIN, "HoC med. & Failed")]),
    ("DA obs", "Candour preserved: the identical-seed pair, the disowned motivation, "
               "the limitations", [
        (MAIN, "yet reach $82.5$ and $66.2$ dev F1"),
        (MAIN, "as our motivating argument assumed, but the backbone's own"),
        (MAIN, "\\section{Limitations}")]),
    ("DA m2", "Symmetric narration of nominal results", [
        (MAIN, "not significant after Holm correction over the sweep's twelve comparisons"),
        (MAIN, "the largest gain over LoRA in the ladder but not significant")]),
    ("DA alt1", "Numerics: fp32, EVA's fp32 rate by multi-seed selection, fp16 one step lower", [
        (MAIN, "picks $10^{-4}$: there EVA trains in every seed"),
        (MAIN, "at $3\\times10^{-5}$, it is merely undertrained"),
        (MAIN, "One grid step lower in fp16 it neither fails nor gains")]),
    ("DA alt2", "Regularisation: all-module rank 4 vs feed-forward rank 8", [
        (MAIN, "all-module rank $4$ is $-0.8$ on ChemProt")]),
    ("DA alt3", "Estimation noise: BERT-Tiny profiled with all available text", [
        (MAIN, "four and three times the standard profile, and all the text these corpora")]),
    ("DA alt4", "The reference may be irrelevant: random-token reference", [
        (MAIN, "$13$--$15\\times$ with random tokens")]),
    ("DA stake", "PEFT-library defaults, EVA authors, generative decoder users", [
        (MAIN, "\\noindent\\textbf{What this means for defaults and for their authors.}"),
        (MAIN, "practitioners who use decoders"),
        (MAIN, "so our conclusions about EVA do not hinge on the budget unit")]),
    ("DA prem", "Does the ranking change at 0.5% or 2% on other tasks?", [
        (MAIN, "\\section{Budgets on RCT-20k and HoC}"),
        (MAIN, "At $0.5\\%$ (rank $4$) and $2\\%$ (rank $16$) on RCT-20k and HoC"),
        (TAB_BX, "Rank $4$, $8$ and $16$ spend $0.53\\%$, $1.06\\%$ and $2.1\\%$"),
        (TAB_BX, "RCT-20k & 4 &"), (TAB_BX, "HoC & 4 &")]),
    # ------------------------------------------------- round 2 (re-review)
    ("R2 NEW-1", "Protocol box: threshold stated two-sided, consistent with the ladder", [
        (MAIN, "medians of $59$--$125\\times$ coincided with failed runs and $1$--$15\\times$ with stable training"),
        (MAIN, "on the BERT ladder $20$--$59\\times$ trained stably at tuned rates"),
        (MAIN, "not as a verdict"),
        (RESP2, "adopted the suggested wording")]),
    ("R2 NEW-2", "AdaLoRA's RCT-20k rate: the grid did include the round-1 rates; "
                 "the near-tie and what it costs are reported", [
        (MAIN, "and the rule takes the last"),
        (MAIN, "selection by the mean over seeds picks the more stable rate"),
        (TAB_LR, "RoBERTa-base, RCT-20k, AdaLoRA & 3e-3 [1e-4, 1e-2]"),
        (RESP2, "One correction and one agreement")]),
    ("R2 NEW-3", "The uninformative feed-forward rank-14 HoC cell, reruns in fp32 and "
                 "under three-seed selection", [
        (MAIN, "\\subsection{RQ3: where should the adapter go?}"),
        (MAIN, "Repairing the uninformative HoC cell"),
        (MAIN, "one of the two failed seeds recovers"),
        (MAIN, "choosing the rate by the multi-seed mean takes the lower one"),
        (MAIN, "rests on that rate rather than on fp32"),
        (TAB_PLACE, "at the rate a three-seed dev mean selects"),
        (RESP2, "NEW-3")]),
    ("R2 NEW-4", "Abstract qualified: 'agree once corrected', 'no accuracy we can detect "
                 "at this power'; controls inherit their parent's rate", [
        (MAIN, "agree once multiple comparisons are corrected"),
        (MAIN, "add no accuracy we can detect at this power"),
        (MAIN, "inherit their parent configuration's rate")]),
    ("R2 NEW-5", "Author block: placeholder marked and an anonymous alternative supplied", [
        (MAIN, "AUTHOR BLOCK --- fill this in before submission"),
        (MAIN, "Anonymous alternative for a double-blind venue")]),
    ("R2 NEW-6", "Seed counts stated in the captions of Tables III and IV; five-seed "
                 "reference values given", [
        (TAB_ABL, "seeds $1$--$3$ throughout"),
        (TAB_ABL, "Over five seeds the reference rows give"),
        (TAB_PLACE, "seeds $1$--$3$; Table~\\ref{tab:main} reports five seeds")]),
    ("R2 NEW-7", "Appendix set in one column so its tables no longer take a float page each", [
        (MAIN, "\\onecolumn"),
        (MAIN, "its tables are wide, and in two-column mode each")]),
    ("R2 NEW-8", "Availability placeholder kept visible until the repository is public", [
        (MAIN, "\\NUM{will be released with the paper}"),
        (RESP2, "stays red deliberately")]),
    ("R2 letter", "Point-by-point response to the round-2 re-review", [
        (RESP2, "Response to the Round-2 Re-Review"),
        (RESP2, "NEW-1"), (RESP2, "NEW-3"), (RESP2, "NEW-7"),
        (RESP2, "Acknowledged limitations")]),
    ("Letter", "Point-by-point response covering every required and suggested item", [
        (RESP, "Revision roadmap: status"),
        (RESP, "Editor-in-Chief"), (RESP, "Reviewer 1 (Methodology)"),
        (RESP, "Reviewer 2 (Domain)"), (RESP, "Reviewer 3 (Perspective)"),
        (RESP, "Devil's advocate")]),
]


def main():
    bad = []
    print("REVIEW AUDIT - peer_review.docx against the revised sources\n" + "=" * 78)
    for ident, demand, evidence in ITEMS:
        missing = [e for src, e in evidence if " ".join(e.split()) not in src]
        status = "OK  " if not missing else "MISS"
        print(f"{status} {ident:10s} {demand}")
        for m in missing:
            print(f"         missing evidence: {m!r}")
            bad.append((ident, m))
    # the abstract must also fit the venue guidance
    import re
    a = MAIN[MAIN.index("\\begin{abstract}"):MAIN.index("\\end{abstract}")]
    a = re.sub(r"\\NUM\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\$[^$]*\$", "X", re.sub(r"\\method\{\}", "DRIFT", a))
    n = len([w for w in re.split(r"\s+", re.sub(r"\\[a-zA-Z]+", "", a)) if re.search(r"[A-Za-z0-9]", w)])
    ok = n <= 250
    print(f"{'OK  ' if ok else 'MISS'} EIC D1     abstract is {n} words (<= 250)")
    if not ok:
        bad.append(("EIC D1", f"{n} words"))
    left = MAIN.count("\\NUM{")
    print(f"\nred placeholders left in the manuscript: {left} "
          f"({'release URL only' if left == 1 else 'check them'})")
    print(f"\n{len(ITEMS) + 1} review items checked, {len(bad)} with missing evidence")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
