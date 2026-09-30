"""Check every item of the three review rounds against the revised sources.

Round 1 is peer_review.docx, round 2 ARS_re_review_round2.docx and round 3
ARS_review_round3_final.md. Each entry names a demand and the evidence that must be
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


# the paper followed by its supplementary material (the full analyses and the
# former appendices moved there when the paper was cut to ten pages)
MAIN = load("main.tex") + " " + load("supplement.tex")
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
        (MAIN, "Supplementary Material for"),
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
        (MAIN, "medians of $59$--$125\\times$ coincided with failed or unstable runs"),
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
        # round 3 (P3) replaced "privacy-positive" with a neutral statement
        (MAIN, "served on premises, without data leaving the institution")]),
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
        (TAB_BX, "Rank $4$, $8$ and $16$ spend $0.53\\%$, $1.07\\%$ and $2.14\\%$"),
        (TAB_BX, "RCT-20k & 4 &"), (TAB_BX, "HoC & 4 &")]),
    # ------------------------------------------------- round 2 (re-review)
    ("R2 NEW-1", "Protocol box: threshold stated two-sided, consistent with the ladder", [
        (MAIN, "medians of $59$--$125\\times$ coincided with failed or unstable runs and $1$--$15\\times$ with stable training"),
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
    ("R2 NEW-5", "Author block: placeholder in round 2; filled in with the authors' "
                 "details for the (single-blind) IEEE conference in round 3 (EIC-1)", [
        (MAIN, "\\IEEEauthorblockN{Gowhar Yousuf}"),
        (MAIN, "\\IEEEauthorblockN{Mohammad Ahsan Chishti}"),
        (MAIN, "\\textit{National Institute of Technology Srinagar}")]),
    ("R2 NEW-6", "Seed counts stated in the captions of Tables III and IV; five-seed "
                 "reference values given", [
        (TAB_ABL, "seeds $1$--$3$ throughout"),
        (TAB_ABL, "Over five seeds the reference rows give"),
        (TAB_PLACE, "seeds $1$--$3$; Table~\\ref{tab:main} reports five seeds")]),
    ("R2 NEW-7", "Appendix set in one column so its tables no longer take a float page each "
                 "(now the one-column supplement)", [
        (MAIN, "\\documentclass[conference,onecolumn]{IEEEtran}"),
        (MAIN, "its tables are wide, and in two-column mode each")]),
    ("R2 NEW-8", "Availability placeholder kept visible until the repository is public; "
                 "replaced by the public repository link in round 3 (EIC-2)", [
        (MAIN, "available at \\url{https://github.com/GowharYousuf07/drift_paper}"),
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
    # ------------------------------------------ round 3 (ARS_review_round3_final.md)
    ("R3 EIC-2", "Availability: repository link instead of the red placeholder; "
                 "MTSamples terms (with P2)", [
        (MAIN, "available at \\url{https://github.com/GowharYousuf07/drift_paper}"),
        (MAIN, "whose terms allow its sample reports to be shared for educational use"),
        (MAIN, "states no licence, so we redistribute no note text")]),
    ("R3 EIC-3", "Length: the paper cut to the 10-page IEEE conference limit (references "
                 "included); the full analyses and appendices in a supplement", [
        (MAIN, "\\externaldocument{supplement}"),
        (MAIN, "\\externaldocument{main}"),
        (MAIN, "whose sections, tables and figures are numbered with an S")]),
    ("R3 EIC-4", "AI-use disclosure, in the acknowledgments as IEEE requires", [
        (MAIN, "\\section*{Acknowledgment}"),
        (MAIN, "We used an AI assistant, Claude (Anthropic)")]),
    ("R3 M1", "Table IV's DRIFT rows declared as inheriting DRIFT's rate; V-D softened", [
        (MAIN, "every LoRA placement and budget of"),
        (MAIN, "the reference controls, \\method{} placed on one module type and the fp32 reruns"),
        (MAIN, "at the rate it inherits from all-module \\method{}"),
        (TAB_PLACE, "rows are not tuned: they inherit the rate of all-module")]),
    ("R3 M2", "Protocol step 2: select the rate on the mean dev score over >= 3 seeds", [
        (MAIN, "by the mean dev score over at least three seeds"),
        (MAIN, "the protocol of Fig.~\\ref{fig:protocol} recommends the multi-seed rule")]),
    ("R3 M3", "What the instance-level intervals exclude (equivalence at +-2 points)", [
        (MAIN, "every interval ends below $+1.2$ points"),
        (MAIN, "$12$ of the $16$ lie within $\\pm2$ points, which establishes equivalence at that margin"),
        (MAIN, "(one we did not fix in advance)"),
        (MAIN, "exclude gains above $1.2$ points on ChemProt and RCT-20k")]),
    ("R3 M4", "Runs that die after their best epoch: Section IV and the Table VI caption", [
        (MAIN, "it does not flag a run that trains well and then stops"),
        (TAB_DEC, "the loss scale of two of EVA's three HoC seeds collapses")]),
    ("R3 M5", "1.07% with its denominator; 'around 1%'; linear-probe '--' explained; "
              "decoder batch, length and head in Section IV", [
        (MAIN, "($1.07\\%$ of the backbone's $124.06$M)"),
        (MAIN, "matched budget of $1.07\\%$ of the backbone"),
        (MAIN, "typically around $1\\%$ of the parameters"),
        (TAB_BX, "$0.53\\%$, $1.07\\%$ and $2.14\\%$ of the backbone's parameters"),
        (TAB_MAIN, "not assessed for the linear probe (--)"),
        (TAB_FP32, "not assessed for the linear probe (--)"),
        (MAIN, "decoder keeps these lengths and epochs but needs a batch of $8$ on HoC"),
        (MAIN, "classifying from its last token with a linear head")]),
    ("R3 D1", "CORAL ordering scoped to the benchmarks; the clinical inversion named", [
        (MAIN, "On the three benchmarks they follow the same order as the drift ratio"),
        (MAIN, "are the one inversion")]),
    ("R3 D2", "Energy medians labelled by aggregation wherever they appear", [
        (MAIN, "and, over all modules, carry a median"),
        (MAIN, "the aggregate we quote from here on unless we name a module group"),
        (MAIN, "energy of their initial directions over all modules"),
        (MAIN, "over all $72$ modules, compared with $24$ of $24$ and $73\\times$ on ChemProt"),
        (MAIN, "average direction over all $224$ modules"),
        (MAIN, "Over all modules, EVA's leading directions carry a median $59\\times$"),
        (MAIN, "EVA's all-module medians were")]),
    ("R3 D3", "Citation status: TLoRA at ACL 2026, RSRA retitled, AIRA authors, "
              "SmolLM2 and massive activations at COLM", [
        (BIB, "Computational Linguistics (ACL, Volume 1: Long Papers)"),
        (BIB, "{RSRA}: Training-Free Probing of Representation Sensitivity"),
        (BIB, "Han, Sirui"),
        (BIB, "Data-Centric Training of a Fully Open Small Language Model"),
        (BIB, "Conference on Language Modeling (COLM)")]),
    ("R3 P1", "A plain practitioner take-away opens Section VIII", [
        (MAIN, "For practitioners the advice is short.")]),
    ("R3 P2", "MTSamples terms of use stated in the Availability paragraph", [
        (MAIN, "The Hugging Face release we use states no licence")]),
    ("R3 P3", "Neutral wording in the deployment/privacy sentence", [
        (MAIN, "served on premises, without data leaving the institution")]),
    ("R3 DA-1", "Verdict scoped to non-binding budgets, in the abstract and Section VIII; "
                "input-side signals only", [
        (MAIN, "so capacity rarely binds; where it does, no allocation gain survives "
               "correction, but our power is lowest there"),
        (MAIN, "This verdict holds where the budget does not bind"),
        (MAIN, "we tested only input-side, activation-derived signals"),
        (MAIN, "although with three seeds our power is lowest there")]),
    ("R3 DA-2", "Rate-selection evidence for the curvature account scoped to ChemProt", [
        (MAIN, "Elsewhere the selected rates are weak evidence"),
        (MAIN, "the ChemProt pattern that Section~\\ref{sec:outliers} reads as support")]),
    ("R3 DA-3", "Ladder: the predicted direction appears as EVA's deficit, not as a gain", [
        (MAIN, "so the ladder gives the prediction no partial support either")]),
    ("R3 DA-4", "EVA's HoC collapse: the head-interaction alternative acknowledged", [
        (MAIN, "may also act through the randomly initialised multi-label head"),
        (MAIN, "cannot separate the two")]),
    ("R3 logs", "Per-epoch logs claimed only for the runs that have them (found while "
                "checking M4)", [
        (MAIN, "$784$ of the runs also log, for each epoch"),
        (MAIN, "with its per-epoch logs, where recorded")]),
]

# Phrases the round-3 revision removed; each must be gone from the manuscript
GONE = [
    ("R3 EIC-1", "Author Name"),
    ("R3 EIC-2", "\\NUM{will be released"),
    ("R3 P3", "privacy-positive"),
    ("R3 M5", "well under $1\\%$"),
    ("R3 M5", "$1.06\\%$"),
    ("R3 M2", "as the protocol states"),
    ("R3 logs", "every run logs"),
]


def main():
    bad = []
    print("REVIEW AUDIT - review rounds 1-3 against the revised sources\n" + "=" * 78)
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
    for ident, text in GONE:
        present = " ".join(text.split()) in MAIN
        print(f"{'MISS' if present else 'OK  '} {ident:10s} removed: {text!r}")
        if present:
            bad.append((ident, f"still present: {text}"))
    left = MAIN.count("\\NUM{")
    print(f"\nred placeholders left in the manuscript: {left}"
          + (" (check them)" if left else ""))
    # decisions that belong to the authors: reported, not counted as missing
    if "Author Name" in MAIN:
        print("OPEN R3 EIC-1 author block is still the placeholder (the authors fill it in "
              "before submission)")
    print(f"\n{len(ITEMS) + 1} review items and {len(GONE)} removals checked, "
          f"{len(bad)} with missing evidence")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
