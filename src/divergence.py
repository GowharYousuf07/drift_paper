"""Tau-free divergences between the target and reference activation covariances.

For each adaptable module of a backbone, compares the covariance of the target
task's inputs (Sigma_D) with that of the general-domain reference (Sigma_G), and
calibrates the numbers against the divergence between two disjoint halves of the
reference corpus (Sigma_G' vs Sigma_G: the sampling-noise floor). On ChemProt
it also scores two further corpora against the reference (news text and
uniformly random tokens) to place the biomedical tasks on a scale.

Three divergences, each on trace-normalised covariances C = Sigma / tr(Sigma),
so that only the geometry, not the overall energy, is compared:
  CORAL    ||C_A - C_B||_F / ||C_B||_F           (Sun et al., 2016)
  Bures    d_BW^2 = 2 - 2 tr((C_B^1/2 C_A C_B^1/2)^1/2)
  Jeffreys (tr(A'^-1 B') + tr(B'^-1 A'))/(2d) - 1, a log-det (Gaussian KL)
           divergence per dimension, with both covariances shrunk by 0.1
           towards a scaled identity so that they are invertible.

    python src/divergence.py --model roberta-base --task chemprot
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as data_mod     # noqa: E402
import drift as drift_mod   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "runs", "divergence")


def _norm(s):
    s = 0.5 * (s + s.T)
    return s / torch.diagonal(s).sum().clamp_min(1e-30)


def _shrink(c, g=0.1):
    d = c.shape[0]
    return (1 - g) * c + g * (torch.diagonal(c).sum() / d) * torch.eye(d, dtype=c.dtype)


def divergences(sa, sb):
    """CORAL, Bures and Jeffreys divergences of covariance sa from reference sb."""
    a, b = _norm(sa.double()), _norm(sb.double())
    coral = float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-30))
    ev, U = torch.linalg.eigh(b)
    root = (U * ev.clamp_min(0).sqrt()) @ U.T
    m = root @ a @ root
    fid = torch.linalg.eigvalsh(0.5 * (m + m.T)).clamp_min(0).sqrt().sum()
    bures = float((2.0 - 2.0 * fid).clamp_min(0))
    a2, b2 = _shrink(a), _shrink(b)
    la, lb = torch.linalg.cholesky(a2), torch.linalg.cholesky(b2)
    t1 = torch.cholesky_solve(b2, la).diagonal().sum()      # tr(A^-1 B)
    t2 = torch.cholesky_solve(a2, lb).diagonal().sum()      # tr(B^-1 A)
    d = a.shape[0]
    jeff = float((t1 + t2) / (2 * d) - 1.0)
    return {"coral": coral, "bures": bures, "jeffreys": jeff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--task", default="chemprot")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--extra_refs", action="store_true",
                    help="also score news text and random tokens against the reference")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # runs next to training jobs: keep the eigendecompositions to a few cores
    torch.set_num_threads(2)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{a.model.replace('/', '__')}__{a.task}.json")
    if os.path.exists(out_path):
        print("exists:", out_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = data_mod.load_task(a.task)
    max_len = data_mod.TASK_MAXLEN.get(a.task, 128)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=task["num_labels"]).float().to(device)
    drift_mod.ensure_padding(tok, model)
    model.eval()
    modules = drift_mod.find_target_modules(model)

    # the reference half is exactly the profiles' reference (first n passages
    # after the fixed shuffle); the floor uses the next n, disjoint from it
    wiki = data_mod.load_reference_corpus(n_docs=2 * a.n)
    corpora = {"target": task["splits"]["train"][0][:a.n],
               "ref": wiki[:a.n], "ref2": wiki[a.n:2 * a.n]}
    if a.extra_refs:
        corpora["news"] = data_mod.load_reference_corpus(n_docs=a.n, kind="news")
        corpora["random"] = data_mod.load_reference_corpus(
            n_docs=a.n, kind="random", tokenizer=tok, n_tokens=max_len - 2)
    t0 = time.perf_counter()
    cov = {}
    for name, texts in corpora.items():
        cov[name] = drift_mod.collect_covariances(model, tok, texts, modules, device,
                                                  max_len=max_len,
                                                  batch_size=a.batch_size)
        print(f"{name}: {len(texts)} texts, {time.perf_counter()-t0:.0f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pairs = [p for p in ("target", "ref2", "news", "random") if p in cov]
    res = {"model": a.model, "task": a.task, "n": a.n, "max_len": max_len,
           "n_texts": {k: len(v) for k, v in corpora.items()},
           "dims": {n: m.in_features for n, m in modules.items()}, "modules": {}}
    done = {}
    for name in modules:
        # W_Q, W_K and W_V read one input and share one covariance: score it once
        site = name.replace("attention.self.key", "attention.self.query") \
                   .replace("attention.self.value", "attention.self.query")
        if site not in done:
            sb = cov["ref"][name][0]
            done[site] = {p: divergences(cov[p][name][0], sb) for p in pairs}
        res["modules"][name] = done[site]
    res["t_s"] = time.perf_counter() - t0
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out_path, f"({res['t_s']:.0f}s)")


if __name__ == "__main__":
    main()
