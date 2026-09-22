"""Validate the generated Kaggle notebook before anyone spends GPU quota on it.

Checks that
  1. the notebook is valid JSON with the expected structure,
  2. every embedded module decodes byte-identically to the file in src/,
  3. every code cell parses as Python (after stripping IPython ! and % magics),
  4. the phases reference plans that actually exist in grid.py.
"""
import ast
import base64
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "kaggle",
                  sys.argv[1] if len(sys.argv) > 1 else "drift_experiments.ipynb")
SRC = os.path.join(ROOT, "src")

fail = 0


def check(ok, msg):
    global fail
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fail += 1


print("notebook:", NB)
nb = json.load(open(NB, encoding="utf-8"))
check(isinstance(nb.get("cells"), list) and len(nb["cells"]) > 5,
      f"structure: {len(nb.get('cells', []))} cells")

# 2. embedded payload round-trips
payload = None
for c in nb["cells"]:
    s = "".join(c["source"])
    if "PAYLOAD = json.loads" in s:
        blob = s.split("r'''")[1].split("'''")[0]
        payload = json.loads(blob)
        break
check(payload is not None, "found embedded payload")
if payload:
    for name, b64 in sorted(payload.items()):
        want = open(os.path.join(SRC, name), "rb").read()
        got = base64.b64decode(b64)
        check(got == want, f"{name} byte-identical ({len(want)} bytes)")

# 3. code cells parse
magic = re.compile(r"^\s*[!%]")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "PAYLOAD = json.loads" in src:
        continue          # one enormous literal; JSON-checked above
    # replace magics/shell escapes with a harmless statement, preserving indent
    lines = []
    for ln in src.splitlines():
        if magic.match(ln):
            lines.append(" " * (len(ln) - len(ln.lstrip())) + "pass")
        else:
            lines.append(ln)
    try:
        ast.parse("\n".join(lines))
        check(True, f"cell {i} parses")
    except SyntaxError as e:
        check(False, f"cell {i} SyntaxError: {e}")

# 4. referenced plans exist
grid = open(os.path.join(SRC, "grid.py"), encoding="utf-8").read()
plans = set(re.findall(r'"(\w+)":\s*plan_\w+', grid))
nb_text = "".join("".join(c["source"]) for c in nb["cells"])
used = set(re.findall(r"--plan\s+(\w+)", nb_text)) | \
    set(re.findall(r"run_plan\('(\w+)'", nb_text)) | \
    set(re.findall(r"tune_until_done\('(\w+)'\)", nb_text)) | \
    set(re.findall(r"\('(\w+)', '[\d,]+'\)", nb_text))
# plans listed for run_all as [(p, seeds) for p in ['a', 'b', ...]]
for lst in re.findall(r"for p in \[([^\]]*)\]", nb_text):
    used |= set(re.findall(r"'(\w+)'", lst))
check((bool(used) or "SMOKE" in nb_text or "STABILITY" in nb_text) and used <= plans,
      f"plans used {sorted(used)} all defined {sorted(plans)}")

print(f"\n{'ALL CHECKS PASSED' if not fail else str(fail) + ' CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
