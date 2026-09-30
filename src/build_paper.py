"""Build the paper and its supplementary material with every cross-reference resolved.

The two documents cite each other's sections, tables and figures through the xr
package, which reads the other document's .aux file. The build therefore compiles
main -> supplement -> main -> supplement, keeping the .aux files, and then fails
if either log still reports an undefined or multiply defined reference.

    python src/build_paper.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
TECTONIC = os.path.join(ROOT, "tools", "tectonic.exe")
DOCS = ("main", "supplement")


def compile_doc(doc):
    r = subprocess.run([TECTONIC, "-X", "compile", f"{doc}.tex", "--outdir", ".",
                        "--keep-intermediates", "--keep-logs"],
                       cwd=PAPER, capture_output=True, text=True)
    if r.returncode:
        sys.stdout.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit(f"{doc}.tex failed to compile")


def main():
    for doc in DOCS + DOCS:
        compile_doc(doc)
    # xr also reads the other document's \bibcite lines, so every key cited in
    # both documents is reported as "multiply defined"; each PDF still numbers
    # its citations from its own bibliography, so only real labels count
    with open(os.path.join(PAPER, "refs.bib"), encoding="utf-8") as f:
        bibkeys = set(re.findall(r"@\w+\{([^,\s]+),", f.read()))
    bad = 0
    for doc in DOCS:
        with open(os.path.join(PAPER, f"{doc}.log"), encoding="utf-8", errors="ignore") as f:
            log = f.read()
        undefined = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
        citations = sorted(set(re.findall(r"Citation `([^']+)'[^\n]*undefined", log)))
        multiple = sorted(set(re.findall(r"Label `([^']+)' multiply defined", log)) - bibkeys)
        overfull = len(re.findall(r"Overfull \\hbox", log))
        print(f"{doc}.pdf: undefined refs {undefined or 'none'}, undefined citations "
              f"{citations or 'none'}, multiply defined {multiple or 'none'}, "
              f"overfull boxes {overfull}")
        bad += len(undefined) + len(citations) + len(multiple)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
