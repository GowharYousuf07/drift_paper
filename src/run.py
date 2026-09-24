"""Single-experiment runner.

    python src/run.py --model roberta-base --task chemprot --method drift --seed 1
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common               # noqa: E402
import data as data_mod     # noqa: E402
import engine               # noqa: E402
import profile_drift        # noqa: E402
import runspec              # noqa: E402
from runspec import NEEDS_PROFILE, run_id   # noqa: E402,F401  (re-exported)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "runs", "results")


def main():
    a = runspec.parse()
    if a.deterministic:
        # must be set before the first cuBLAS call
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(RESULT_DIR, exist_ok=True)
    rid = run_id(a)
    out_path = os.path.join(RESULT_DIR, rid + ".json")
    if os.path.exists(out_path) and not a.force:
        print("SKIP (exists):", rid)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.set_seed(a.seed)

    task = data_mod.load_task(a.task, max_train=a.max_train, seed=0)
    max_len = a.max_len or data_mod.TASK_MAXLEN.get(a.task, 128)

    profile = None
    if a.method in NEEDS_PROFILE:
        key = profile_drift.profile_key(a.model, a.task, a.n_ref, a.n_dom,
                                        center=(a.cov == "centered"), ref=a.ref,
                                        gev=(a.method == "gev"))
        p = os.path.join(profile_drift.PROFILE_DIR, key + ".pt")
        if not os.path.exists(p):
            raise SystemExit("missing profile: " + p + "\nrun src/profile_drift.py first")
        profile = torch.load(p, weights_only=False)

    model, tok = engine.build_model(a.model, task["num_labels"], task["multilabel"],
                                    head=a.head)
    info = engine.apply_method(model, a.method, budget_rank=a.budget_rank,
                               alpha=a.alpha, dropout=a.dropout, profile=profile,
                               tau=a.tau, score_mode=a.score_mode, rho=a.rho,
                               r_min=a.r_min, init_mode=a.init_mode,
                               alloc_mode=a.alloc_mode,
                               target=a.target, seed=a.seed, scale=a.scale,
                               scaling=a.scaling)
    if getattr(a, "grad_ckpt", False):
        # recompute activations in the backward pass instead of storing them: the
        # same updates with far less memory. Non-reentrant checkpointing passes
        # gradients to the adapters inside each block although the frozen
        # embeddings feeding it do not require grad; only active in train mode.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    res = engine.train_eval(model, tok, task, device, info,
                            epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                            eval_batch_size=a.eval_batch_size, max_len=max_len,
                            seed=a.seed, weight_decay=a.weight_decay,
                            log_every=1 if a.verbose else 0, amp=a.amp,
                            save_preds=not a.no_save_preds)
    wall = time.perf_counter() - t0

    ranks = info.get("ranks", {})
    record = {
        "id": rid, "args": vars(a), "wall_s": wall,
        "n_train": len(task["splits"]["train"][0]),
        "n_dev": len(task["splits"]["dev"][0]),
        "n_test": len(task["splits"]["test"][0]),
        "num_labels": task["num_labels"], "metric": task["metric"],
        "max_len": max_len,
        "budget": {k: info[k] for k in ("budget_rank", "budget_params", "n_modules")
                   if k in info},
        "spent_params": info.get("spent_params"),
        "rank_hist": {str(k): sum(1 for v in ranks.values() if v == k)
                      for k in sorted(set(ranks.values()))} if ranks else {},
        "ranks": {k: int(v) for k, v in ranks.items()},
        "versions": {"torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
        "result": res,
    }
    common.save_json(record, out_path)
    m = task["metric"]
    print(f"DONE {rid}\n  test {m}={res['test'][m]:.4f} "
          f"dev={res['dev_best'].get(m, float('nan')):.4f} "
          f"adapter_params={res['params_adapter']} "
          f"time={res['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
