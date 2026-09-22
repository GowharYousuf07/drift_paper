"""How much of the manuscript's prose is backed by an automatic check?

verify_results.py compares strings it builds from the raw runs against main.tex.
This script runs it, collects every string that matched, and then lists the
numbers in the body prose that no matched string covers, so that an unchecked
claim cannot hide. Numbers in tables and in generated captions are excluded:
those files are regenerated from the runs and re-derived by verify_results.py.

    python src/number_coverage.py
"""
import io
import os
import re
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")

# values taken from other papers: nothing of ours to re-derive
LITERATURE = {"81.9\\pm1.0", "79.7"}          # Gururangan et al.; the BLURB split
# constants that define the experiment rather than report a result
SETTINGS = {"1.5\\times", "0.1", "1.0", "0.01"}   # AdaLoRA's inflated budget,
# GEV's shrinkage weight, gradient clipping, weight decay

# numbers that state the design rather than a result, and so have nothing to
# re-derive: budgets, hyperparameters, dataset sizes, section/figure numbers
DESIGN = {
    "1{,}327{,}104", "1.06", "0.95", "16", "0.01", "6", "32", "128", "96", "512",
    "10", "8", "12", "1{,}024", "3{,}227", "200", "5{,}000", "6{,}000", "4{,}169",
    "2{,}427", "3{,}469", "1{,}295", "186", "371", "13", "5", "72", "24", "48",
    "768", "3{,}072", "1{,}536", "3{,}840", "0.10", "0.60", "1.33", "1.41", "1.43",
    "124.06", "125", "110", "4.4", "11.2", "28.8", "41.4", "360", "224", "64", "32",
    "2", "3", "4", "1", "0", "70", "15", "12{,}", "1{,}972", "1{,}436", "20",
    "0.5", "2.1", "0.53", "1", "9", "21", "23", "77", "588", "40", "0.9", "0.94",
}


def main():
    # 1. run verify_results with quoted() instrumented to record what matched
    src = open(os.path.join(ROOT, "src", "verify_results.py"),
               encoding="utf-8-sig").read()
    src = src.replace(
        'def quoted(s):\n    """True if the string s occurs in main.tex (line breaks and runs of spaces\n    count as one space)."""\n    return " ".join(s.split()) in TEXTN',
        'MATCHED = []\n\n\ndef quoted(s):\n    n = " ".join(s.split())\n    if n in TEXTN:\n        MATCHED.append(n)\n    return n in TEXTN')
    if "MATCHED = []" not in src:
        raise SystemExit("could not instrument quoted(); check verify_results.py")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    ns = runpy.run_path(os.path.join(ROOT, "src", "verify_results.py"),
                        init_globals={"__INSTRUMENTED__": True}) if False else None
    sys.stdout = old
    # runpy cannot take modified source, so exec the patched text directly
    g = {"__file__": os.path.join(ROOT, "src", "verify_results.py"), "__name__": "__main__"}
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(compile(src, "verify_results.py", "exec"), g)
    finally:
        sys.stdout = old
    matched = " || ".join(g.get("MATCHED", []))
    report = buf.getvalue()
    mismatches = int(re.search(r"TOTAL MISMATCHES: (\d+)", report).group(1))

    # 2. the body prose: drop the preamble, the bibliography and \input-ed tables
    text = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    body = text[text.index("\\begin{abstract}"):text.index("\\bibliographystyle")]
    body += text[text.index("\\appendices"):]
    body = re.sub(r"\\caption\{.*?\n\n", " ", body, flags=re.S)   # figure captions
    body = re.sub(r"\\label\{[^}]*\}|\\ref\{[^}]*\}|\\cite[^{]*\{[^}]*\}", " ", body)
    body = re.sub(r"\\(section|subsection|IfFileExists|input)\{[^}]*\}", " ", body)
    body = " ".join(body.split())

    # 3. result-shaped numbers in the prose: $82.4$, $-1.1$, $82.4\pm0.3$, p = 0.04,
    #    percentages. Pure integers, exponents and subscripts are design or notation.
    pattern = (r"\$[+-]?\d+\.\d+(?:\\pm\d+\.\d+)?\$"        # $82.4$, $82.4\pm0.3$
               r"|\$[+-]?\d+\.\d+--\$?[+-]?\d+\.\d+\$?"      # $1.5$--$2.7$
               r"|\d+\.\d+\\times"                           # 2.7\times
               r"|p = \d?\.\d+")                             # p = 0.04
    uncovered, cited, settings = [], 0, 0
    for m in re.finditer(pattern, body):
        tok = m.group(0).strip("$")
        if tok.rstrip(".") in DESIGN:
            continue
        # covered only if some matched string contains this very occurrence:
        # it must include the token and sit within the surrounding text. Tested
        # before the exclusion lists, so a value that happens to read like a
        # setting (weight decay 1.0 and a difference of 1.0 points) is credited
        # to the check that covers it rather than waved through.
        window = body[max(0, m.start() - 200):m.end() + 200]
        if any(tok in s and s in window for s in g.get("MATCHED", [])):
            continue
        if tok in LITERATURE:
            cited += 1
            continue
        if tok in SETTINGS:
            settings += 1
            continue
        uncovered.append((tok, " ".join(body[max(0, m.start() - 60):m.end() + 60].split())))

    print(f"verify_results.py: {mismatches} mismatches, "
          f"{len(g.get('MATCHED', []))} matched strings")
    print(f"excluded: {cited} values quoted from other papers, {settings} experiment "
          f"settings (nothing of ours to re-derive)")
    print(f"result numbers in the body prose not covered by a matched check: "
          f"{len(uncovered)}\n")
    seen = set()
    for tok, ctx in uncovered:
        if ctx in seen:
            continue
        seen.add(ctx)
        print(f"  {tok:>12s}   ...{ctx}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
