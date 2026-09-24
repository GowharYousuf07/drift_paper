"""Generate a self-contained Kaggle/Colab notebook that reproduces the whole study.

The notebook embeds every source file as base64 so it is a single upload with no
external repository. Run:

    python src/make_kaggle.py
"""
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT_DIR = os.path.join(ROOT, "kaggle")

MODULES = ["common.py", "data.py", "drift.py", "peft_methods.py", "engine.py",
           "profile_drift.py", "run.py", "runspec.py", "grid.py", "analyze.py",
           "download_data.py", "divergence.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


ALL_PHASES = ("tune", "main", "ablation", "budget", "ladder")
# hours after which no new run starts; lower it when less GPU quota is left
DEADLINE_HOURS = 10.5
# which parts of the revision a notebook runs (see revision_cells)
REVISION_BLOCKS = ("divergence", "tune_rev", "stage2", "dependent", "clinical",
                   "independent", "decoder", "decoder_fp32")
DIV_TASKS = ("chemprot", "rct20k", "hoc", "mtsamples")


def revision_cells(blocks):
    """The revision as composable blocks, each resumable (finished runs are
    skipped), so that sessions on different accounts can split the work:
      divergence  tau-free divergences, in the background (CPU-bound)
      tune_rev    adaptive learning-rate selection for the original configurations
      quick       short runs that use those selections (seeds 4-5 of the reference
                  controls, the factorisation elsewhere, fp16 EVA one step lower)
      stage2      runs that use those selections (incl. fp32 HoC)
      dependent   post-audit runs that need stage 1's selections (Table I replicate
                  with predictions, the rest of the fp32 column, factorisation)
      clinical    the clinical-notes task
      independent post-audit runs with their own tuning (rsLoRA, budgets
                  elsewhere, linear head) and EVA's fp32 rates
      decoder     the decoder on HoC (slowest; last)
      decoder_fp32 EVA and LoRA on the decoder's HoC in fp32 (needs the decoder
                  profile of the kernel that ran 'decoder' attached); _eva and
                  _lora run one method each, so two accounts can share them"""
    cells = []
    cells.append(code(
        "def pending_count(plan, seeds='1,2,3', model=MODEL):\n"
        "    r = subprocess.run([sys.executable, 'src/grid.py', '--plan', plan, '--model',\n"
        "                        model, '--seeds', seeds, '--count'],\n"
        "                       capture_output=True, text=True)\n"
        "    for line in r.stdout.splitlines():\n"
        "        if line.startswith('PENDING'):\n"
        "            return int(line.split()[1])\n"
        "    print(r.stdout[-2000:], r.stderr[-2000:])\n"
        "    return -1\n"
        "\n"
        "def tune_until_done(plan):\n"
        "    for rnd in range(1, 8):\n"
        "        n = pending_count(plan, seeds='1')\n"
        "        print(f'[{plan}] round {rnd}: {n} runs', flush=True)\n"
        "        if n <= 0 or time.time() > DEADLINE:\n"
        "            break\n"
        "        run_plan(plan, seeds='1')\n"
        "\n"
        "def run_all(plans):\n"
        "    for plan, seeds in plans:\n"
        "        n = pending_count(plan, seeds=seeds)\n"
        "        print(f'[{plan}] {n} runs to do', flush=True)\n"
        "        if n == 0:\n"
        "            continue\n"
        "        run_plan(plan, seeds=seeds)\n"
        "div_proc = None\n"
    ))
    if "divergence" in blocks:
        tasks = " ".join(DIV_TASKS)
        cells.append(md(
            "## Background: tau-free divergences (CPU-bound)\n"
            "CORAL, Bures and log-det divergences between each task's activation "
            "covariances and the reference's, against a reference-vs-reference "
            "floor. Runs beside the training jobs; finished tasks are skipped."
        ))
        cells.append(code(
            f"DIV_TASKS = {list(DIV_TASKS)!r}\n"
            "if not all(os.path.exists(f'runs/divergence/roberta-base__{t}.json')\n"
            "           for t in DIV_TASKS):\n"
            f"    script = ('for t in {tasks}; do extra=\"\"; '\n"
            "              '[ \"$t\" = chemprot ] && extra=--extra_refs; '\n"
            "              f'{sys.executable} src/divergence.py --task $t $extra; done')\n"
            "    div_proc = subprocess.Popen(['bash', '-c', script],\n"
            "                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),\n"
            "                                stdout=open('logs/divergence.log', 'w'),\n"
            "                                stderr=subprocess.STDOUT)\n"
            "print('divergence job', 'started' if div_proc else 'already complete')\n"
        ))
    if "tune_rev" in blocks:
        cells.append(md(
            "## Learning-rate selection for every original configuration\n"
            "Seed 1, dev set: the first grid of every configuration, then one step "
            "past whichever edge holds the best score, until every selection is "
            "interior."
        ))
        cells.append(code(
            "tune_until_done('tune_rev')\n"
            "import importlib\n"
            "sys.path.insert(0, 'src')\n"
            "import grid\n"
            "importlib.reload(grid)\n"
            "for (mdl, task, m, extra, first), tried, todo in grid.tuning_status(MODEL):\n"
            "    best = grid.best_lr(tried) if tried else None\n"
            "    print(f\"{mdl.split('/')[-1][:24]:24s} {task:9s} {m:9s} {str(extra):36s} \"\n"
            "          + ' '.join(f'{lr:g}:{100*d:.1f}' for lr, d in sorted(tried.items()))\n"
            "          + f'  -> {best}' + (f'  (still to run: {todo})' if todo else ''))\n"
        ))
    if "quick" in blocks:
        cells.append(md(
            "## Short runs first\n"
            "Seeds 4-5 of the reference controls, the allocation/initialisation "
            "factorisation on RCT-20k and HoC, and fp16 EVA one grid step lower on HoC."
        ))
        cells.append(code(
            "run_all([('r2_fixes', '1,2,3'), ('refctl45', '4,5'), ('factor_x', '1,2,3'), ('eva_lowlr', '1,2,3,4,5')])\n"
        ))
    if "stage2" in blocks:
        cells.append(md("## Runs at the selected rates (including fp32 HoC)"))
        cells.append(code(
            "run_all([(p, '1,2,3') for p in ['rev_core', 'placement_budget', 'refctl',\n"
            "         'eva_units', 'gev', 'ladder_tuned', 'tiny_prof', 'budget_tuned',\n"
            "         'fp32_hoc']] + [('eva_lowlr', '1,2,3,4,5')])\n"
        ))
    if "dependent" in blocks:
        cells.append(md(
            "## Post-audit runs that use the selections above\n"
            "Table I replicated with stored predictions (bootstrap intervals), EVA's "
            "fp32 rates, the rest of the fp32 HoC column, and the factorisation on "
            "RCT-20k and HoC."
        ))
        cells.append(code(
            "run_all([(p, '1,2,3') for p in ['factor_x', 'preds', 'eva_fp32_lr',\n"
            "                                'fp32_rest']])\n"
        ))
    if "clinical" in blocks:
        cells.append(md(
            "## Clinical notes (MTSamples specialties)\n"
            "LoRA, EVA, whitened EVA and DRIFT tuned on the clinical dev set, three "
            "seeds each, and DRIFT with news and random-token references."
        ))
        cells.append(code(
            "tune_until_done('tune_clin')\n"
            "run_all([('clinical', '1,2,3')])\n"
        ))
    if "independent" in blocks:
        cells.append(md(
            "## Post-audit runs with their own tuning\n"
            "EVA's fp32 rates on HoC below its selected one; rsLoRA's scale at every "
            "budget, the 0.5% and 2% budgets on RCT-20k and HoC, and the linear head, "
            "each tuned on its own."
        ))
        cells.append(code(
            "run_all([('eva_fp32_lr_low', '1,2,3')])\n"
            "tune_until_done('tune_rev2a')\n"
            "run_all([(p, '1,2,3') for p in ['rslora', 'budget_x', 'linhead']])\n"
        ))
    if "decoder" in blocks:
        cells.append(md("## The decoder SLM on HoC (about 35 min per run)"))
        cells.append(code(
            "tune_until_done('tune_rev2b')\n"
            "run_all([('decoder_hoc', '1,2,3')])\n"
        ))
    for blk in ("decoder_fp32", "decoder_fp32_eva", "decoder_fp32_lora"):
        if blk not in blocks:
            continue
        who = {"decoder_fp32": "EVA and LoRA", "decoder_fp32_eva": "EVA",
               "decoder_fp32_lora": "LoRA"}[blk]
        cells.append(md(
            "## The decoder on HoC in fp32\n"
            f"{who} at the fp16-tuned rate, deterministic kernels, gradient "
            "checkpointing so fp32 fits a T4 (about two hours per run, so the "
            "per-run cap is raised). EVA reuses the attached kernel's decoder "
            "profile, so its initial directions match its fp16 runs."
        ))
        cells.append(code(
            "os.environ['DRIFT_RUN_TIMEOUT_H'] = '5'\n"
            "prof = 'runs/profiles/HuggingFaceTB__SmolLM2-360M__hoc__ref1024__dom1024__cen.pt'\n"
            "print('decoder HoC profile present:', os.path.exists(prof))\n"
            f"run_all([('{blk}', '1,2,3')])\n"
        ))
    cells.append(code(
        "if div_proc is not None:\n"
        "    try:\n"
        "        if div_proc.poll() is None and time.time() < DEADLINE:\n"
        "            div_proc.wait(timeout=max(60, DEADLINE - time.time()))\n"
        "    except subprocess.TimeoutExpired:\n"
        "        print('divergence job still running at the deadline')\n"
        "    print(open('logs/divergence.log', errors='ignore').read()[-2000:])\n"
        "snapshot('revision')\n"
    ))
    return cells


RESULT_KEEP = ("test", "dev_best", "best_epoch", "train_time_s", "peak_mem_bytes",
               "params_total", "params_trainable", "params_head", "params_adapter",
               "params_adapter_effective", "steps")


def _results_blob():
    """Every local result JSON, zipped and base64-encoded, so a notebook starts
    from the complete set of finished runs (including the tuning runs, which no
    earlier kernel output holds in full). Kaggle caps a notebook at 1 MB, so each
    record keeps only what the planner and the tables read (arguments, scores,
    parameter counts); the full records stay local, and ingestion never
    overwrites an existing file."""
    import glob
    import zlib
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", "results", "*.json")))
    records = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        slim = {k: r[k] for k in ("id", "args", "metric", "num_labels", "n_train",
                                  "spent_params") if k in r}
        slim["result"] = {k: r["result"][k] for k in RESULT_KEEP if k in r["result"]}
        slim["stripped"] = True
        records[os.path.basename(p)] = slim
    # one compressed document, so the redundancy across records is exploited
    packed = zlib.compress(json.dumps(records).encode("utf-8"), 9)
    return base64.b64encode(packed).decode("ascii"), len(paths)


def build(out_name, phases=ALL_PHASES, profile_phase=True, env=None,
          restore_profiles=False, embed_results=False):
    """Write one notebook. A follow-up notebook can skip profiling and run only
    some phases; finished runs are restored from earlier outputs and skipped.
    With restore_profiles, profile tensors in an attached kernel's output are
    copied in, so follow-up runs reuse the exact profiles of the original ones.
    With embed_results, every local result JSON travels inside the notebook."""
    os.makedirs(OUT_DIR, exist_ok=True)

    payload = {}
    for m in MODULES:
        with open(os.path.join(SRC, m), "rb") as f:
            payload[m] = base64.b64encode(f.read()).decode("ascii")
    blob = json.dumps(payload)

    cells = []

    cells.append(md(
        "# DRIFT: Domain-Residual Rank Allocation for Parameter-Efficient Adaptation\n"
        "\n"
        "Reproduces every experiment in the paper. Works on **Kaggle** and "
        "**Google Colab**.\n"
        "\n"
        "**Kaggle:** in the right-hand panel set **Accelerator = GPU T4 x2** and "
        "**Internet = On**, then use *Save Version -> Save & Run All (Commit)* so it "
        "runs in the background. Each session stops starting new runs after "
        "`DEADLINE_HOURS`, so it always finishes inside Kaggle's 12-hour limit and "
        "saves `drift_results.zip`. To continue in a later session, upload that zip "
        "as a Kaggle Dataset (or a new version of it), attach it with *Add Input*, "
        "and run again: finished runs are restored and skipped.\n"
        "\n"
        "**Colab:** *Runtime -> Change runtime type -> T4 GPU*, then *Run all*. "
        "Results are written straight to Google Drive (`MyDrive/drift_results/`), "
        "so a disconnect loses at most the run in progress: just *Run all* again.\n"
    ))

    cells.append(code(
        "import os, sys, subprocess, time\n"
        "SESSION_START = time.time()\n"
        "ON_KAGGLE = os.path.exists('/kaggle/working')\n"
        "ON_COLAB = 'google.colab' in sys.modules\n"
        "WORK = '/kaggle/working' if ON_KAGGLE else '/content/drift'\n"
        "os.makedirs(WORK, exist_ok=True)\n"
        "os.chdir(WORK)\n"
        "print('platform:', 'kaggle' if ON_KAGGLE else 'colab' if ON_COLAB else 'other',\n"
        "      '| work dir:', WORK)\n"
        "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',\n"
        "                      '--format=csv'], capture_output=True, text=True).stdout)\n"
        "import torch\n"
        "NGPU = torch.cuda.device_count()\n"
        "print('torch', torch.__version__, '| GPUs:', NGPU)\n"
        "assert NGPU > 0, 'No GPU: enable a GPU accelerator/runtime first.'\n"
    ))

    cells.append(md(
        "The harness implements every PEFT method itself, so the only requirements "
        "are torch, transformers, pandas/pyarrow, scikit-learn and scipy -- all "
        "preinstalled on Kaggle and Colab. We only install what is genuinely "
        "missing, rather than upgrading packages and risking the environment."
    ))
    cells.append(code(
        "import importlib\n"
        "need = []\n"
        "for mod, pkg in [('torch','torch'), ('transformers','transformers'),\n"
        "                 ('pandas','pandas'), ('pyarrow','pyarrow'),\n"
        "                 ('sklearn','scikit-learn'), ('scipy','scipy')]:\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "    except ImportError:\n"
        "        need.append(pkg)\n"
        "print('missing:', need or 'nothing')\n"
        "if need:\n"
        "    subprocess.run([sys.executable,'-m','pip','install','-q',*need], check=True)\n"
        "import transformers\n"
        "print('transformers', transformers.__version__)\n"
    ))

    cells.append(md("## Write the source tree"))
    cells.append(code(
        "import base64, json\n"
        "os.makedirs(os.path.join(WORK, 'src'), exist_ok=True)\n"
        "PAYLOAD = json.loads(r'''" + blob + "''')\n"
        "for name, b64 in PAYLOAD.items():\n"
        "    with open(os.path.join(WORK, 'src', name), 'wb') as f:\n"
        "        f.write(base64.b64decode(b64))\n"
        "print('wrote', len(PAYLOAD), 'modules:', sorted(PAYLOAD))\n"
    ))

    cells.append(md("## Fetch datasets"))
    cells.append(code(
        "subprocess.run([sys.executable, 'src/download_data.py'], cwd=WORK, check=True)\n"
    ))

    cells.append(md(
        "## Restore results from earlier sessions\n"
        "Kaggle: any attached `drift_results.zip` (a previous version's output) is "
        "unpacked. Colab: `runs/results` is linked to Google Drive, so earlier "
        "results are already there."
    ))
    cells.append(code(
        "import glob, zipfile, shutil\n"
        "os.makedirs('runs/profiles', exist_ok=True)\n"
        "if ON_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    DRIVE = '/content/drive/MyDrive/drift_results'\n"
        "    os.makedirs(os.path.join(DRIVE, 'results'), exist_ok=True)\n"
        "    if os.path.isdir('runs/results') and not os.path.islink('runs/results'):\n"
        "        for p in glob.glob('runs/results/*.json'):\n"
        "            shutil.copy(p, os.path.join(DRIVE, 'results'))\n"
        "        shutil.rmtree('runs/results')\n"
        "    if not os.path.exists('runs/results'):\n"
        "        os.symlink(os.path.join(DRIVE, 'results'), 'runs/results')\n"
        "os.makedirs('runs/results', exist_ok=True)\n"
        "restored = 0\n"
        "for z in sorted(glob.glob('/kaggle/input/**/drift_results.zip', recursive=True)):\n"
        "    with zipfile.ZipFile(z) as zf:\n"
        "        for n in zf.namelist():\n"
        "            if n.startswith('results/') and n.endswith('.json'):\n"
        "                dst = os.path.join('runs', n)\n"
        "                if not os.path.exists(dst):\n"
        "                    with zf.open(n) as s, open(dst, 'wb') as d:\n"
        "                        d.write(s.read())\n"
        "                    restored += 1\n"
        "    print('restored from', z)\n"
        "# a zip uploaded as a Kaggle Dataset arrives already unpacked\n"
        "for p in glob.glob('/kaggle/input/**/results/*.json', recursive=True):\n"
        "    dst = os.path.join('runs/results', os.path.basename(p))\n"
        "    if not os.path.exists(dst):\n"
        "        shutil.copy(p, dst)\n"
        "        restored += 1\n"
        "print('restored', restored, 'new results;',\n"
        "      len(glob.glob('runs/results/*.json')), 'results present in total')\n"
    ))
    if embed_results:
        rblob, nres = _results_blob()
        cells.append(code(
            f"# the {nres} results finished before this notebook was built (slim\n"
            "# records: arguments, scores and parameter counts)\n"
            "import zlib\n"
            "RECORDS = json.loads(zlib.decompress(base64.b64decode('" + rblob + "')))\n"
            "added = 0\n"
            "for fname, rec in RECORDS.items():\n"
            "    dst = os.path.join('runs', 'results', fname)\n"
            "    if not os.path.exists(dst):\n"
            "        with open(dst, 'w') as d:\n"
            "            json.dump(rec, d)\n"
            "        added += 1\n"
            "print('embedded results added:', added, '| total',\n"
            "      len(glob.glob('runs/results/*.json')))\n"
        ))
    if restore_profiles:
        cells.append(code(
            "# reuse the attached kernel's profiles: identical initialisations to the\n"
            "# runs being extended (profile_drift.py skips profiles that exist)\n"
            "copied = 0\n"
            "for p in glob.glob('/kaggle/input/**/runs/profiles/*.pt', recursive=True) + \\\n"
            "         glob.glob('/kaggle/input/**/runs/profiles/*.json', recursive=True):\n"
            "    dst = os.path.join('runs/profiles', os.path.basename(p))\n"
            "    if not os.path.exists(dst):\n"
            "        shutil.copy(p, dst)\n"
            "        copied += 1\n"
            "print('copied', copied, 'profile files:',\n"
            "      sorted(os.path.basename(p) for p in glob.glob('runs/profiles/*.pt')))\n"
        ))

    cells.append(md(
        "## Configure\n"
        "\n"
        "`DRIFT_AMP=1` turns on fp16 autocast, a large speed-up on T4/P100, applied "
        "identically to every method so comparisons stay matched. Runs are split "
        "across all visible GPUs (two on Kaggle's T4 x2). No new run starts after "
        "`DEADLINE_HOURS`; the margin leaves room for the last run to finish."
    ))
    cells.append(code(
        "os.environ['DRIFT_AMP'] = '1'\n"
        "# fail a stalled checkpoint download instead of hanging on it\n"
        "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'\n"
        "os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'\n"
        + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in (env or {}).items()) +
        "MODEL = 'roberta-base'\n"
        "DECODER = 'HuggingFaceTB/SmolLM2-360M'   # decoder SLM for the generality check\n"
        f"DEADLINE_HOURS = {DEADLINE_HOURS}          # Kaggle kills a session at 12 h\n"
        "DEADLINE = SESSION_START + DEADLINE_HOURS * 3600\n"
        # each plan computes the profiles its runs need before sharding; this is
        # instant when they already exist, so it is never switched off
        "NEED_PROFILES = True\n"
        "os.makedirs('logs', exist_ok=True)\n"
        "\n"
        "def snapshot(label=''):\n"
        "    \"\"\"Zip the result JSONs and profile summaries (not the large .pt\n"
        "    profile tensors, which are recomputed cheaply).\"\"\"\n"
        "    paths = sorted(glob.glob('runs/results/*.json')) + \\\n"
        "            sorted(glob.glob('runs/profiles/*.json')) + \\\n"
        "            sorted(glob.glob('runs/divergence/*.json'))\n"
        "    out = os.path.join(WORK, 'drift_results.zip')\n"
        "    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:\n"
        "        for p in paths:\n"
        "            z.write(p, os.path.relpath(p, 'runs'))\n"
        "    if ON_COLAB:\n"
        "        shutil.copy(out, DRIVE)\n"
        "    nres = sum(1 for p in paths if 'results' in p)\n"
        "    print(f'[snapshot {label}] {nres} results -> {out}', flush=True)\n"
        "\n"
        "def run_plan(plan, seeds='1,2,3', model=MODEL):\n"
        "    \"\"\"Run one experiment plan, one worker per GPU, then snapshot.\"\"\"\n"
        "    if time.time() > DEADLINE:\n"
        "        print(f'[{plan}] skipped: session deadline reached. Start a new '\n"
        "              'session to continue.')\n"
        "        return\n"
        "    base = [sys.executable, 'src/grid.py', '--plan', plan,\n"
        "            '--model', model, '--seeds', seeds]\n"
        "    if NEED_PROFILES:\n"
        "        subprocess.run(base + ['--profiles_only', '--deadline', str(DEADLINE)],\n"
        "                       check=False)\n"
        "    procs, logs = [], []\n"
        "    for g in range(NGPU):\n"
        "        log = f'logs/{plan}_gpu{g}.log'\n"
        "        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))\n"
        "        procs.append(subprocess.Popen(\n"
        "            base + ['--shard', f'{g}/{NGPU}', '--deadline', str(DEADLINE),\n"
        "                    '--no_profiles'],\n"
        "            env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT))\n"
        "        logs.append(log)\n"
        "    t0 = last_snap = last_print = time.time()\n"
        "    while any(p.poll() is None for p in procs):\n"
        "        # poll often, so a plan with nothing left to do costs seconds, not minutes\n"
        "        time.sleep(5)\n"
        "        if time.time() - last_print < 120:\n"
        "            continue\n"
        "        last_print = time.time()\n"
        "        done = sum(open(l, errors='ignore').read().count('\\nDONE ') for l in logs)\n"
        "        print(f'[{plan}] {(time.time()-t0)/60:.0f} min, {done} runs finished '\n"
        "              f'this session', flush=True)\n"
        "        # a session stopped from outside (quota, time limit) keeps what the\n"
        "        # last snapshot holds, so snapshot during long plans too\n"
        "        if time.time() - last_snap > 900:\n"
        "            snapshot(plan + ' (partial)')\n"
        "            last_snap = time.time()\n"
        "    for l in logs:\n"
        "        txt = open(l, errors='ignore').read()\n"
        "        print(f'--- tail of {l} ---')\n"
        "        print(txt[-1500:])\n"
        "        if 'Traceback' in txt:\n"
        "            print(f'!! errors in {l}: search it for Traceback')\n"
        "    snapshot(plan)\n"
    ))

    if profile_phase:
        cells.append(md(
            "## Phase 0 - DRIFT profiles\n"
            "Two forward-only passes per (model, task). Cheap; also produces the "
            "profiling-cost numbers reported in the paper."
        ))
        cells.append(code(
            "for task in ['chemprot', 'rct20k', 'hoc']:\n"
            "    subprocess.run([sys.executable, 'src/profile_drift.py', '--model', MODEL,\n"
            "                    '--task', task], check=False)\n"
            "snapshot('profiles')\n"
        ))

    if "smoke" in phases:
        cells.append(md(
            "## Smoke test\n"
            "Exercises the covariance profiling and every adapter code path on "
            "short runs, and checks that DRIFT at tau=0 reproduces EVA exactly."
        ))
        cells.append(code(
            "import json, glob\n"
            "def sh(*a):\n"
            "    r = subprocess.run([sys.executable, *a], capture_output=True, text=True)\n"
            "    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]\n"
            "    print('$', ' '.join(a[:6]), '->', 'OK' if r.returncode == 0 else f'EXIT {r.returncode}')\n"
            "    print('   ' + '\\n   '.join(tail))\n"
            "    return r.returncode\n"
            "fails = 0\n"
            "fails += sh('src/profile_drift.py', '--model', MODEL, '--task', 'chemprot') != 0\n"
            "for method, extra in [('drift', []), ('eva', []), ('drift', ['--tau', '0']),\n"
            "                      ('lora', []), ('drift', ['--init_mode', 'rand_ortho'])]:\n"
            "    fails += sh('src/run.py', '--model', MODEL, '--task', 'chemprot',\n"
            "                '--method', method, '--epochs', '1', '--max_train', '600',\n"
            "                '--amp', '--tag', 'smoke', *extra) != 0\n"
            "res = {}\n"
            "for p in glob.glob('runs/results/*smoke*.json'):\n"
            "    r = json.load(open(p)); a = r['args']\n"
            "    res[(a['method'], a['tau'], a['init_mode'])] = r['result']['dev_best']['micro_f1']\n"
            "    print(a['method'], 'tau', a['tau'], a['init_mode'], 'cov', a.get('cov'),\n"
            "          'scale', a.get('scale'), 'dev', round(r['result']['dev_best']['micro_f1'], 4),\n"
            "          'ranks', r['rank_hist'])\n"
            "same = res.get(('drift', 0.0, 'drift')) == res.get(('eva', 0.95, 'drift'))\n"
            "print('DRIFT(tau=0) == EVA:', same)\n"
            "for p in glob.glob('runs/profiles/*.json'):\n"
            "    print(open(p).read())\n"
            "print('SMOKE', 'PASSED' if fails == 0 and same else f'FAILED ({fails} failures, same={same})')\n"
            "snapshot('smoke')\n"
        ))

    if "stability" in phases:
        cells.append(md(
            "## Stability check\n"
            "Re-runs, on full data for a few epochs, the exact configurations that "
            "collapsed to a constant prediction before EVA was brought in line with "
            "its reference implementation (centred covariance, fixed scaling)."
        ))
        cells.append(code(
            "import json, glob\n"
            "MAJORITY_DEV = {'chemprot': 0.3346, 'hoc': 0.1183}   # constant-prediction dev scores\n"
            "for task in ['chemprot', 'hoc']:\n"
            "    subprocess.run([sys.executable, 'src/profile_drift.py', '--model', MODEL,\n"
            "                    '--task', task], check=False)\n"
            "# EVA at the two configurations that collapsed at lr 3e-4, over lower rates\n"
            "jobs = [(t, 'eva', r, lr) for lr in ('1e-4', '3e-5', '1e-5')\n"
            "        for t, r in (('chemprot', '1'), ('hoc', '8'))]\n"
            "procs = []\n"
            "for i, (task, method, r, lr) in enumerate(jobs):\n"
            "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % NGPU))\n"
            "    cmd = [sys.executable, 'src/run.py', '--model', MODEL, '--task', task,\n"
            "           '--method', method, '--budget_rank', r, '--seed', '1', '--lr', lr,\n"
            "           '--epochs', '3', '--amp', '--tag', 'stab']\n"
            "    if task == 'hoc':\n"
            "        cmd += ['--batch_size', '16', '--max_len', '512']\n"
            "    procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,\n"
            "                                  stderr=open(f'logs/stab_{i}.log', 'w')))\n"
            "    if len(procs) == NGPU:\n"
            "        for p in procs: p.wait()\n"
            "        procs = []\n"
            "for p in procs: p.wait()\n"
            "ok = True\n"
            "for p in sorted(glob.glob('runs/results/*stab*.json')):\n"
            "    r = json.load(open(p)); a = r['args']; m = r['metric']\n"
            "    dev = r['result']['dev_best'][m]\n"
            "    collapsed = dev <= MAJORITY_DEV[a['task']] + 0.01\n"
            "    ok &= not collapsed\n"
            "    print(f\"{a['task']:9s} {a['method']:6s} r{a['budget_rank']:<3d} lr={a['lr']:g} dev={dev:.4f} \"\n"
            "          f\"best_epoch={r['result']['best_epoch']} {'COLLAPSED' if collapsed else 'ok'}\")\n"
            "for p in glob.glob('logs/stab_*.log'):\n"
            "    t = open(p).read()\n"
            "    if 'Traceback' in t: ok = False; print(p, t[-1500:])\n"
            "print('STABILITY', 'PASSED' if ok else 'FAILED')\n"
            "snapshot('stability')\n"
        ))

    if "decoder_smoke" in phases:
        cells.append(md(
            "## Decoder smoke test\n"
            "Profiles the decoder SLM and runs every compared method for one short "
            "epoch, to exercise the decoder code path (padding, module discovery, "
            "classification head) before committing GPU hours."
        ))
        cells.append(code(
            "import json, glob\n"
            "def sh(*a):\n"
            "    r = subprocess.run([sys.executable, *a], capture_output=True, text=True)\n"
            "    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]\n"
            "    print('$', ' '.join(a[:6]), '->', 'OK' if r.returncode == 0 else f'EXIT {r.returncode}')\n"
            "    print('   ' + '\\n   '.join(tail))\n"
            "    return r.returncode\n"
            "fails = 0\n"
            "t0 = time.time()\n"
            "fails += sh('src/profile_drift.py', '--model', DECODER, '--task', 'chemprot') != 0\n"
            "print(f'profile {time.time()-t0:.0f}s', flush=True)\n"
            "for method in ('lora', 'eva', 'eva_white', 'drift'):\n"
            "    t0 = time.time()\n"
            "    fails += sh('src/run.py', '--model', DECODER, '--task', 'chemprot', '--method', method,\n"
            "                '--epochs', '1', '--max_train', '600', '--lr', '3e-4', '--amp',\n"
            "                '--tag', 'smoke') != 0\n"
            "    print(f'{method} {time.time()-t0:.0f}s', flush=True)\n"
            "for p in sorted(glob.glob('runs/results/*SmolLM2*smoke*.json')):\n"
            "    r = json.load(open(p)); a = r['args']; res = r['result']\n"
            "    print(a['method'], 'dev', round(res['dev_best']['micro_f1'], 4),\n"
            "          'test', round(res['test']['micro_f1'], 4), 'adapter params',\n"
            "          res.get('params_adapter'), 'modules', r['budget'].get('n_modules'),\n"
            "          'train_s', round(res.get('train_time_s', 0)), 'ranks', r['rank_hist'])\n"
            "print('DECODER SMOKE', 'PASSED' if fails == 0 else f'FAILED ({fails} failures)')\n"
            "snapshot('decoder_smoke')\n"
        ))

    if "cost" in phases:
        cells.append(md(
            "## Profiling cost at a single tau\n"
            "The sweep profiles five values of tau, sharing one reference "
            "eigendecomposition. This times what a practitioner actually runs -- "
            "one tau -- into separately named files, leaving the sweep untouched."
        ))
        cells.append(code(
            "for task in ['chemprot', 'rct20k', 'hoc']:\n"
            "    subprocess.run([sys.executable, 'src/profile_drift.py', '--model', MODEL,\n"
            "                    '--task', task, '--taus', '0.95', '--force',\n"
            "                    '--out_suffix', '__tau1'], check=False)\n"
            "snapshot('cost')\n"
        ))

    if "revision_smoke" in phases:
        cells.append(md(
            "## Revision smoke test\n"
            "Every new code path on short runs: profiles with the three reference "
            "controls and the GEV pencil, one run through each new configuration, "
            "a same-seed pair in deterministic fp32, and the plans of the revision "
            "notebook as a dry run."
        ))
        cells.append(code(
            "import json, glob\n"
            "def sh(*a, timeout=3600):\n"
            "    r = subprocess.run([sys.executable, *a], capture_output=True, text=True,\n"
            "                       timeout=timeout)\n"
            "    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]\n"
            "    print('$', ' '.join(a[:12]), '->', 'OK' if r.returncode == 0 else\n"
            "          f'EXIT {r.returncode}', flush=True)\n"
            "    print('   ' + '\\n   '.join(tail), flush=True)\n"
            "    return r.returncode\n"
            "fails = 0\n"
            "t0 = time.time()\n"
            "for extra in (['--ref', 'random'], ['--ref', 'news'], ['--ref', 'shuffled']):\n"
            "    fails += sh('src/profile_drift.py', '--model', MODEL, '--task', 'chemprot',\n"
            "                '--n_ref', '64', '--n_dom', '64', '--taus', '0.0,0.95', *extra) != 0\n"
            "fails += sh('src/profile_drift.py', '--model', MODEL, '--task', 'chemprot',\n"
            "            '--n_ref', '64', '--n_dom', '64', '--taus', '0.0', '--gev') != 0\n"
            "print(f'profiles {time.time()-t0:.0f}s', flush=True)\n"
            "base = ['src/run.py', '--model', MODEL, '--task', 'chemprot', '--epochs', '1',\n"
            "        '--max_train', '600', '--seed', '1', '--amp', '--tag', 'rsmoke']\n"
            "small = ['--n_ref', '64', '--n_dom', '64']\n"
            "for extra in (['--method', 'gev', *small],\n"
            "              ['--method', 'drift', '--ref', 'random', *small],\n"
            "              ['--method', 'drift', '--ref', 'news', *small],\n"
            "              ['--method', 'drift', '--ref', 'shuffled', *small],\n"
            "              ['--method', 'eva', '--alloc_mode', 'units'],\n"
            "              ['--method', 'lora', '--target', 'ffn', '--budget_rank', '14'],\n"
            "              ['--method', 'lora', '--budget_rank', '4']):\n"
            "    t1 = time.time()\n"
            "    fails += sh(*base, *extra) != 0\n"
            "    print(f'   {time.time()-t1:.0f}s', flush=True)\n"
            "# deterministic fp32 twice with one seed, and the same run in fp16 for speed\n"
            "det = ['src/run.py', '--model', MODEL, '--task', 'hoc', '--method', 'drift',\n"
            "       '--epochs', '1', '--max_train', '200', '--batch_size', '16',\n"
            "       '--max_len', '512', '--seed', '1']\n"
            "for extra in (['--deterministic', '--tag', 'rsmokeA'],\n"
            "              ['--deterministic', '--tag', 'rsmokeB'], ['--amp', '--tag', 'rsmokeC']):\n"
            "    t1 = time.time()\n"
            "    fails += sh(*det, *extra) != 0\n"
            "    print(f'   {time.time()-t1:.0f}s', flush=True)\n"
            "rs = {}\n"
            "for p in sorted(glob.glob('runs/results/*rsmoke*.json')):\n"
            "    r = json.load(open(p)); a = r['args']; res = r['result']; m = r['metric']\n"
            "    rs[a['tag']] = res\n"
            "    print(a['method'], a['task'], 'ref', a.get('ref'), 'alloc', a['alloc_mode'],\n"
            "          'init', a['init_mode'], 'tgt', a['target'], 'r', a['budget_rank'],\n"
            "          'amp', a['amp'], 'det', a.get('deterministic'), '| dev', round(res['dev_best'][m], 4),\n"
            "          'test', round(res['test'][m], 4), '| adapter', res['params_adapter'],\n"
            "          'head', res['params_head'], '| train_s', round(res['train_time_s']),\n"
            "          '| ranks', r['rank_hist'])\n"
            "    print('    history', res.get('history'))\n"
            "same = [rs[t] for t in ('rsmokeA', 'rsmokeB') if t in rs]\n"
            "if len(same) == 2:\n"
            "    ok = same[0]['dev_best'] == same[1]['dev_best'] and same[0]['test'] == same[1]['test']\n"
            "    print('DETERMINISTIC fp32 PAIR IDENTICAL:', ok)\n"
            "    fails += not ok\n"
            "for p in sorted(glob.glob('runs/profiles/*ref64__dom64*.json')):\n"
            "    s = json.load(open(p))\n"
            "    print(os.path.basename(p), 'n_ref_actual', s.get('n_ref_actual'),\n"
            "          'drift_ratio', {k: round(v, 4) for k, v in s['drift_ratio'].items()})\n"
            "    for t, lead in s.get('lead', {}).items():\n"
            "        hid = [v for n, v in lead.items() if 'intermediate.dense' in n or\n"
            "               'attention.self' in n]\n"
            "        hits = sum(v['argmax'] in (77, 588) for v in hid)\n"
            "        er = sorted(v['energy_rel'] for v in lead.values())\n"
            "        print(f'   lead[{t}]: {hits}/{len(hid)} hidden-state inputs peak on 77/588;'\n"
            "              f' median energy {er[len(er)//2]:.1f}x')\n"
            "print('ALL WIKITEXT PASSAGES:', subprocess.run([sys.executable, '-c',\n"
            "      'import sys; sys.path.insert(0,\"src\"); import data; '\n"
            "      'print(len(data.load_reference_corpus(n_docs=10**6)))'],\n"
            "      capture_output=True, text=True).stdout.strip())\n"
            "for plan, seeds in [('tune_rev', '1'), ('rev_core', '1,2,3'), ('placement_budget', '1,2,3'),\n"
            "                    ('refctl', '1,2,3'), ('eva_units', '1,2,3'), ('gev', '1,2,3'),\n"
            "                    ('ladder_tuned', '1,2,3'), ('tiny_prof', '1,2,3'),\n"
            "                    ('budget_tuned', '1,2,3'), ('fp32_hoc', '1,2,3')]:\n"
            "    r = subprocess.run([sys.executable, 'src/grid.py', '--plan', plan, '--model', MODEL,\n"
            "                        '--seeds', seeds, '--count'], capture_output=True, text=True)\n"
            "    print(f'{plan:18s}', (r.stdout.strip() or r.stderr.strip()[-600:]))\n"
            "print(subprocess.run([sys.executable, 'src/grid.py', '--plan', 'tune_rev', '--seeds', '1',\n"
            "                      '--dry'], capture_output=True, text=True).stdout[:3000])\n"
            "print('torch', torch.__version__, '| transformers', transformers.__version__)\n"
            "print('REVISION SMOKE', 'PASSED' if fails == 0 else f'FAILED ({fails} failures)')\n"
            "snapshot('revision_smoke')\n"
        ))

    if "revision_smoke2" in phases:
        cells.append(md(
            "## Smoke test of the post-audit code paths\n"
            "Linear head, rsLoRA scale, stored predictions (single- and multi-label), "
            "the decoder on 512-token HoC documents at batch 8 (memory and speed), and "
            "the divergence script on a small sample."
        ))
        cells.append(code(
            "import json, glob\n"
            "def sh(*a, timeout=3600):\n"
            "    r = subprocess.run([sys.executable, *a], capture_output=True, text=True,\n"
            "                       timeout=timeout)\n"
            "    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]\n"
            "    print('$', ' '.join(a[:14]), '->', 'OK' if r.returncode == 0 else\n"
            "          f'EXIT {r.returncode}', flush=True)\n"
            "    print('   ' + '\\n   '.join(tail), flush=True)\n"
            "    return r.returncode\n"
            "fails = 0\n"
            "enc = ['src/run.py', '--model', MODEL, '--epochs', '1', '--seed', '1', '--amp',\n"
            "       '--tag', 'rsmoke2']\n"
            "for extra in (['--task', 'chemprot', '--method', 'lora', '--head', 'linear',\n"
            "               '--max_train', '600'],\n"
            "              ['--task', 'chemprot', '--method', 'bitfit', '--head', 'linear',\n"
            "               '--max_train', '600', '--lr', '1e-3'],\n"
            "              ['--task', 'chemprot', '--method', 'lora', '--scaling', 'rslora',\n"
            "               '--budget_rank', '2', '--max_train', '600'],\n"
            "              ['--task', 'hoc', '--method', 'lora', '--max_train', '200',\n"
            "               '--batch_size', '16', '--max_len', '512']):\n"
            "    t1 = time.time(); fails += sh(*enc, *extra) != 0\n"
            "    print(f'   {time.time()-t1:.0f}s', flush=True)\n"
            "t1 = time.time()\n"
            "fails += sh('src/profile_drift.py', '--model', DECODER, '--task', 'hoc', '--n_ref', '64',\n"
            "            '--n_dom', '64', '--taus', '0.0,0.95') != 0\n"
            "print(f'   decoder HoC profile {time.time()-t1:.0f}s', flush=True)\n"
            "for m in ('lora', 'eva'):\n"
            "    t1 = time.time()\n"
            "    fails += sh('src/run.py', '--model', DECODER, '--task', 'hoc', '--method', m,\n"
            "                '--epochs', '1', '--max_train', '160', '--batch_size', '8',\n"
            "                '--max_len', '512', '--seed', '1', '--amp', '--lr', '3e-4',\n"
            "                '--n_ref', '64', '--n_dom', '64', '--tag', 'rsmoke2') != 0\n"
            "    print(f'   decoder HoC {m}: {time.time()-t1:.0f}s', flush=True)\n"
            "t1 = time.time()\n"
            "fails += sh('src/divergence.py', '--task', 'chemprot', '--n', '64', '--extra_refs') != 0\n"
            "print(f'   divergence (n=64) {time.time()-t1:.0f}s', flush=True)\n"
            "for p in sorted(glob.glob('runs/results/*rsmoke2*.json')):\n"
            "    r = json.load(open(p)); a = r['args']; res = r['result']; m = r['metric']\n"
            "    tp, tg = res.get('test_preds'), res.get('test_gold')\n"
            "    print(a['model'].split('/')[-1], a['task'], a['method'], 'head', a.get('head'),\n"
            "          'scaling', a.get('scaling'), '| test', round(res['test'][m], 4),\n"
            "          '| adapter', res['params_adapter'], 'head params', res['params_head'],\n"
            "          '| preds', None if tp is None else len(tp), 'gold', None if tg is None else len(tg),\n"
            "          '| peak GB', round(res['peak_mem_bytes'] / 1e9, 2), '| train_s', round(res['train_time_s']),\n"
            "          '| steps', res['steps'])\n"
            "    fails += tp is None or tg is None or len(tp) != len(tg)\n"
            "for p in glob.glob('runs/divergence/*.json'):\n"
            "    d = json.load(open(p))\n"
            "    first = next(iter(d['modules'].values()))\n"
            "    print(os.path.basename(p), 'pairs', list(first), 'first module', first)\n"
            "print('SMOKE2', 'PASSED' if fails == 0 else f'FAILED ({fails})')\n"
            "snapshot('revision_smoke2')\n"
        ))

    if "revision" in phases:
        cells.extend(revision_cells(REVISION_BLOCKS))

    tune_cells = []
    tune_cells.append(md(
        "## Phase 1 - Learning-rate selection (dev set, 1 seed)\n"
        "Every main-table method is tuned on its own; ablation variants inherit "
        "their parent method's rate."
    ))
    tune_cells.append(code("run_plan('tune', seeds='1')\n"))
    tune_cells.append(code(
        "# Later phases pick these up automatically (grid.resolve_lr); this cell is\n"
        "# just so the selected values are visible in the notebook output.\n"
        "import collections\n"
        "best = collections.defaultdict(lambda: (-1, None))\n"
        "for p in glob.glob('runs/results/*tune*.json'):\n"
        "    r = json.load(open(p)); a = r['args']\n"
        "    m = r['metric']\n"
        "    dev = r['result']['dev_best'].get(m, -1)\n"
        "    k = (a['task'], a['method'])\n"
        "    if dev > best[k][0]:\n"
        "        best[k] = (dev, a['lr'])\n"
        "for k in sorted(best):\n"
        "    print(k, 'dev=%.4f' % best[k][0], 'lr=', best[k][1])\n"
    ))
    if "tune" in phases:
        cells.extend(tune_cells)

    later = {
        "main": "## Phase 2 - Main table (3 tasks x 9 methods x 3 seeds)",
        "ablation": "## Phase 3 - Ablations (isolates where the gain comes from)",
        "budget": "## Phase 4 - Budget sweep",
        "ladder": ("## Phase 5 - Model-scale ladder\n"
                   "BERT-Tiny (4.4M) through BERT-Base (110M): all share one "
                   "vocabulary and one pretraining recipe, so scale is the only "
                   "variable."),
        "ladder_lr": ("## Ladder control - EVA at LoRA's learning rate\n"
                      "On the ladder EVA used the lower rate it was tuned to on "
                      "RoBERTa-base; this reruns it at the rate the other low-rank "
                      "methods use, so its deficit can be attributed."),
        "placement_x": ("## Placement ablation on RCT-20k and HoC\n"
                        "Attention-only and feed-forward-only adapters for LoRA and "
                        "DRIFT, as in the ChemProt ablation."),
        "seeds45": ("## Seeds 4 and 5\n"
                    "Two further seeds for LoRA, EVA, whitened EVA and DRIFT on every "
                    "task, at the rates tuned for the main table."),
        "decoder_tune": ("## Decoder SLM - learning-rate selection\n"
                         "Dev set, seed 1, the grids used for RoBERTa-base."),
        "decoder_tune_ext": ("## Decoder SLM - grid extension\n"
                             "Every method chose the top of its first grid, so each "
                             "grid is extended past it."),
        "decoder_main": ("## Decoder SLM - three seeds at the tuned rates"),
    }
    calls = {"seeds45": "run_plan('seeds45', seeds='4,5')\n",
             "decoder_tune": "run_plan('decoder_tune', seeds='1', model=DECODER)\n",
             "decoder_tune_ext": "run_plan('decoder_tune_ext', seeds='1', model=DECODER)\n",
             "decoder_main": "run_plan('decoder_main', model=DECODER)\n"}
    for phase, heading in later.items():
        if phase in phases:
            cells.append(md(heading))
            cells.append(code(calls.get(phase, f"run_plan('{phase}')\n")))

    cells.append(md(
        "## Collect results\n"
        "Download **drift_results.zip** (Kaggle: Output panel; Colab: "
        "`MyDrive/drift_results/`)."))
    cells.append(code(
        "snapshot('final')\n"
        "subprocess.run([sys.executable, 'src/analyze.py', '--model', MODEL, '--dump'],\n"
        "               check=False)\n"
    ))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}

    out = os.path.join(OUT_DIR, out_name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({os.path.getsize(out)/1e3:.0f} KB, "
          f"{len(MODULES)} modules embedded, phases {list(phases)})")


def build_revision():
    """The review round. Every notebook carries every finished result (so
    finished runs are skipped) and restores the profiles of any attached kernel
    (so reruns see identical initialisations); the parts are split so that two
    Kaggle accounts can share the work:
      drift_revision.ipynb    everything, in priority order
      drift_revision_a2.ipynb what needs no result of session 1 (second account)
      drift_revision_s2.ipynb what does, plus leftovers (first account, short)
      drift_revision_q1.ipynb only the short runs (first account's last hour)
      drift_revision_f2.ipynb the decoder's HoC in fp32, EVA (attach the kernel
                              that ran the decoder, for its profile)
      drift_revision_f1.ipynb the decoder's HoC in fp32, LoRA (any account)"""
    global REVISION_BLOCKS
    build("drift_revision_smoke.ipynb", phases=("revision_smoke",), profile_phase=False,
          restore_profiles=True, embed_results=True)
    build("drift_revision_smoke2.ipynb", phases=("revision_smoke2",), profile_phase=False)
    for name, blocks in [
            ("drift_revision.ipynb", ("divergence", "tune_rev", "quick", "stage2",
                                      "clinical", "independent", "dependent", "decoder",
                                      "decoder_fp32")),
            ("drift_revision_a2.ipynb", ("divergence", "clinical", "independent", "decoder")),
            ("drift_revision_s2.ipynb", ("tune_rev", "stage2", "dependent")),
            ("drift_revision_q1.ipynb", ("quick",)),
            ("drift_revision_f2.ipynb", ("decoder_fp32_eva",)),
            ("drift_revision_f1.ipynb", ("decoder_fp32_lora",))]:
        REVISION_BLOCKS = blocks
        build(name, phases=("revision",), profile_phase=False,
              restore_profiles=True, embed_results=True)


def main():
    import sys
    global DEADLINE_HOURS
    if sys.argv[1:2] == ["revision"]:
        # python src/make_kaggle.py revision [deadline hours]
        if len(sys.argv) > 2:
            DEADLINE_HOURS = float(sys.argv[2])
        build_revision()
        return
    # One notebook runs everything; attached outputs of earlier runs are restored,
    # and any run whose result already exists is skipped.
    build("drift_experiments.ipynb", phases=("cost",) + ALL_PHASES)
    # a few minutes on Kaggle that exercise every changed code path before
    # committing hours of GPU time to the full notebook
    build("drift_smoke.ipynb", phases=("smoke",), profile_phase=False)
    build("drift_stability.ipynb", phases=("stability",), profile_phase=False)
    # completes an interrupted run: only the backbone ladder. The tuned rates are
    # pinned from the local results so the notebook does not depend on restoring
    # them from an earlier session.
    import sys
    sys.path.insert(0, SRC)
    import grid
    pinned = {m: grid.resolve_lr("roberta-base", "chemprot", m)
              for m in grid.SWEEP_METHODS}
    print("ladder rates pinned:", pinned)
    build("drift_ladder.ipynb", phases=("ladder",), profile_phase=False,
          env={"DRIFT_LADDER_LR": json.dumps(pinned)})
    # EVA on the ladder at LoRA's rate, reusing the ladder kernel's profiles
    build("drift_ladder_lr.ipynb", phases=("ladder_lr",), profile_phase=False,
          env={"DRIFT_LADDER_LR": json.dumps(pinned)}, restore_profiles=True)
    # placement on the other two tasks and two more seeds, at the main-table rates
    # (pinned: the corrected tuning runs are not restorable in a new session)
    tuned = {t: {m: grid.resolve_lr("roberta-base", t, m)
                 for m in ("lora", "eva", "eva_white", "drift")} for t in grid.TASKS}
    print("main-table rates pinned:", tuned)
    build("drift_more.ipynb", phases=("placement_x", "seeds45"), profile_phase=False,
          env={"DRIFT_PINNED_LR": json.dumps(tuned)})
    # the decoder SLM: a short smoke test, then tuning and three seeds; the full
    # notebook reuses the smoke kernel's profile when that kernel is attached
    build("drift_decoder_smoke.ipynb", phases=("decoder_smoke",), profile_phase=False)
    # tuning and the three-seed runs are separate kernels, so the tuning results
    # can be checked before the main runs spend their GPU hours; each attaches its
    # predecessor, which restores the tuning results and the profile
    build("drift_decoder_tune.ipynb", phases=("decoder_tune",), profile_phase=False,
          restore_profiles=True)
    build("drift_decoder_main.ipynb", phases=("decoder_tune_ext", "decoder_main"),
          profile_phase=False, restore_profiles=True)


if __name__ == "__main__":
    main()
