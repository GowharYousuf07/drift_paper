"""Sanity checks on the paper: citations, TODO markers, page count."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
# The author allowed 9 pages for the analysis-first version and, for the revision
# (2026-09-15), more pages where needed; the check reports body, references and
# appendix separately and flags only a body past BODY_LIMIT.
PAGE_LIMIT = 16
BODY_LIMIT = 11

tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()

cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

print(f"citations used: {len(cited)}   bib entries: {len(defined)}")
missing = sorted(cited - defined)
unused = sorted(defined - cited)
print("cited but MISSING from refs.bib:", missing or "none")
print("in refs.bib but never cited:", unused or "none")

todos = [l.strip() for l in tex.splitlines() if "\\TODO{" in l]
print(f"\nopen TODO markers: {len(todos)}")
for t in todos:
    print("  -", t[:110])

log = os.path.join(PAPER, "main.log")
if os.path.exists(log):
    txt = open(log, encoding="utf-8", errors="ignore").read()
    und = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", txt)))
    unc = sorted(set(re.findall(r"Citation `([^']+)'[^\n]*undefined", txt)))
    print("\nundefined refs:", und or "none")
    print("undefined citations:", unc or "none")

pdf = os.path.join(PAPER, "main.pdf")
if os.path.exists(pdf):
    try:
        from pypdf import PdfReader
        pages = PdfReader(pdf).pages
        n = len(pages)
        text = [p.extract_text() or "" for p in pages]
        refs = next((i + 1 for i, t in enumerate(text) if "REFERENCES" in t), None)
        # table captions are set in small caps, so "Appendix A" in a caption also
        # extracts as "APPENDIX A"; the appendix itself follows the references
        app = next((i + 1 for i, t in enumerate(text)
                    if "APPENDIX A" in t and refs and i + 1 > refs), None)
        print(f"\npages: {n} in total; references start on p{refs}, appendix on p{app}")
        if refs:
            print(f"body: {refs} pages ({'OK' if refs <= BODY_LIMIT else f'OVER {BODY_LIMIT}'})")
        if n > PAGE_LIMIT:
            print(f"OVER the {PAGE_LIMIT}-page total")
    except ImportError:
        pass

sys.exit(1 if missing else 0)
