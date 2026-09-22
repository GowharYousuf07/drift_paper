"""Replace bare ALL_CAPS placeholders in main.tex with a \\TODO macro.

Underscores are subscript operators in LaTeX text mode, so raw placeholder names
like RESULTS_MAIN are a compile error; this rewrites them once.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper", "main.tex")

s = open(P, encoding="utf-8").read()

num_def = "\\newcommand{\\NUM}[1]{\\textcolor{red}{#1}}"
todo_def = num_def + "\n\\newcommand{\\TODO}[1]{\\textcolor{red}{[TODO: #1]}}"
if "\\TODO" not in s:
    s = s.replace(num_def, todo_def, 1)


def repl(m):
    return "\\TODO{" + m.group(0).lower().replace("_", " ") + "}"


s = re.sub(r"\b[A-Z]{4,}(?:_[A-Z]+)+\b", repl, s)
open(P, "w", encoding="utf-8").write(s)

for line in s.splitlines():
    if "\\TODO{" in line:
        print(line.strip())
