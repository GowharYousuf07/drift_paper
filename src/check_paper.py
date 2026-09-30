"""Sanity checks on the paper and its supplement: citations, TODO markers, pages.

The paper targets an IEEE conference limit of ten pages including references
(the authors' choice, 2026-09-30); the full analyses and the former appendices
are in supplement.tex, which has no page limit. Build both first with
src/build_paper.py.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
PAGE_LIMIT = 10          # main paper, references included

bib = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
cited_all = set()
for doc in ("main", "supplement"):
    path = os.path.join(PAPER, f"{doc}.tex")
    if not os.path.exists(path):
        continue
    tex = open(path, encoding="utf-8").read()
    cited = {k.strip() for m in re.finditer(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}", tex)
             for k in m.group(1).split(",")}
    cited_all |= cited
    todos = [l.strip() for l in tex.splitlines() if "\\TODO{" in l or "\\NUM{" in l]
    print(f"{doc}.tex: {len(cited)} citation keys, "
          f"missing from refs.bib: {sorted(cited - defined) or 'none'}, "
          f"open TODO/red markers: {len(todos)}")
print("in refs.bib but cited in neither document:", sorted(defined - cited_all) or "none")

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
for doc in ("main", "supplement"):
    pdf = os.path.join(PAPER, f"{doc}.pdf")
    if PdfReader is None or not os.path.exists(pdf):
        continue
    text = [p.extract_text() or "" for p in PdfReader(pdf).pages]
    n = len(text)
    refs = next((i + 1 for i, t in enumerate(text) if "REFERENCES" in t), None)
    line = f"{doc}.pdf: {n} pages, references start on p{refs}"
    if doc == "main":
        line += f" ({'within' if n <= PAGE_LIMIT else 'OVER'} the {PAGE_LIMIT}-page limit)"
    print(line)
