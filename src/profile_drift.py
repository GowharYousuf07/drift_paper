"""Compute and cache DRIFT profiles for a (model, task) pair.

One forward-only pass over a general-domain reference corpus and one over the
unlabelled task text yields, per adaptable linear module, the second-moment
matrices Sigma_G and Sigma_D.  For each requested tau we then store

    - the full drift spectrum (eigenvalues of Sigma~ in descending order),
    - the leading r_max drift eigenvectors (the adapter initialisation basis),
    - trace(Sigma_D) and the reference subspace dimension k.

tau = 0 reduces to plain in-domain activation PCA, i.e. the EVA baseline.

Usage:
    python src/profile_drift.py --model roberta-base --task chemprot
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as data_mod          # noqa: E402
import drift as drift_mod        # noqa: E402
from runspec import profile_key  # noqa: E402,F401  (torch-free, shared with grid.py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, "runs", "profiles")


def lead_summary(basis, evals, trace_d, d_in):
    """Leading direction of one module: its largest coordinate, the weight it
    puts there, and its energy relative to an average direction."""
    u = torch.as_tensor(basis)[:, 0].abs()
    lam_bar = trace_d / d_in
    return {"argmax": int(u.argmax()), "peak": float(u.max()),
            "energy_rel": float(evals[0]) / max(lam_bar, 1e-30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--task", default="chemprot")
    ap.add_argument("--n_ref", type=int, default=1024)
    ap.add_argument("--n_dom", type=int, default=1024)
    ap.add_argument("--taus", default="0.0,0.5,0.9,0.95,0.99")
    ap.add_argument("--r_max", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_suffix", default="",
                    help="write to a separately named profile, e.g. to time a "
                         "single-tau run without touching the cached sweep")
    ap.add_argument("--raw", action="store_true",
                    help="raw second moments instead of covariances")
    ap.add_argument("--ref", default="wikitext", choices=list(data_mod.REFERENCE_KINDS),
                    help="reference corpus: WikiText-103 (default), CNN/DailyMail "
                         "news, word-shuffled WikiText, or uniformly random tokens")
    ap.add_argument("--gev", action="store_true",
                    help="also store the generalised eigenvectors of "
                         "(Sigma_D, Sigma_G) for the GEV initialisation")
    ap.add_argument("--gev_shrink", type=float, default=0.1)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    os.makedirs(PROFILE_DIR, exist_ok=True)
    center = not args.raw
    key = profile_key(args.model, args.task, args.n_ref, args.n_dom,
                      center=center, ref=args.ref, gev=args.gev) + args.out_suffix
    out_path = os.path.join(PROFILE_DIR, key + ".pt")
    if os.path.exists(out_path) and not args.force:
        print("profile exists:", out_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = data_mod.load_task(args.task)
    max_len = args.max_len or data_mod.TASK_MAXLEN.get(args.task, 128)

    tok = AutoTokenizer.from_pretrained(args.model)
    # profile in fp32 whatever the checkpoint's stored dtype (bf16 for SmolLM2)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=task["num_labels"]).float().to(device)
    drift_mod.ensure_padding(tok, model)
    model.eval()

    modules = drift_mod.find_target_modules(model)
    print(f"{len(modules)} adaptable modules")

    dom_texts = task["splits"]["train"][0][:args.n_dom]
    ref_texts = data_mod.load_reference_corpus(n_docs=args.n_ref, kind=args.ref,
                                               tokenizer=tok, n_tokens=max_len - 2)
    print(f"reference {len(ref_texts)} docs ({args.ref}) | domain {len(dom_texts)} docs "
          f"| max_len {max_len}")

    t0 = time.perf_counter()
    cov_g = drift_mod.collect_covariances(model, tok, ref_texts, modules, device,
                                          max_len=max_len, batch_size=args.batch_size,
                                          center=center)
    t_ref = time.perf_counter() - t0
    print(f"reference pass {t_ref:.1f}s")

    t1 = time.perf_counter()
    cov_d = drift_mod.collect_covariances(model, tok, dom_texts, modules, device,
                                          max_len=max_len, batch_size=args.batch_size,
                                          center=center)
    t_dom = time.perf_counter() - t1
    print(f"domain pass {t_dom:.1f}s")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    taus = [float(x) for x in args.taus.split(",")]
    store = {"meta": {"model": args.model, "task": args.task, "n_ref": args.n_ref,
                      "n_dom": args.n_dom, "max_len": max_len, "r_max": args.r_max,
                      "centered": center, "ref": args.ref,
                      "n_ref_actual": len(ref_texts), "n_dom_actual": len(dom_texts),
                      "t_ref_s": t_ref, "t_dom_s": t_dom,
                      "costs": {n: drift_mod.module_cost(m) for n, m in modules.items()},
                      "dims": {n: [m.in_features, m.out_features] for n, m in modules.items()}},
             "taus": {}}
    if args.gev:
        store["gev"] = {}
        store["meta"]["gev_shrink"] = args.gev_shrink

    t2 = time.perf_counter()
    for tau in taus:
        store["taus"][str(tau)] = {}
    # Module-major: the reference eigendecomposition is the expensive step and does
    # not depend on tau, so it is computed once and reused for every tau.
    for name in modules:
        # promote one module at a time; the cached matrices stay float32
        sg = cov_g[name][0].double()
        sd = cov_d[name][0].double()
        if args.gev:
            mu, v, energy = drift_mod.gev_basis(sd, sg, shrink=args.gev_shrink,
                                                r_keep=args.r_max)
            store["gev"][name] = {"evals": mu.float().numpy(),
                                  "basis": v.float().numpy(),
                                  "energy": energy.float().numpy(),
                                  "trace_d": float(torch.diagonal(sd).sum())}
        evals_g, evecs_g = drift_mod.reference_eigh(sg)
        del sg
        for tau in taus:
            if tau <= 0:
                v_comp, k = None, 0          # no deflation: plain target PCA
            else:
                _, k = drift_mod.subspace_from_eigh(evals_g, evecs_g, tau=tau)
                v_comp = evecs_g[:, k:]      # trailing eigenvectors span (I-P)
            evals, evecs, tr_d, tr_drift = drift_mod.drift_spectrum(
                sd, v_comp, r_keep=args.r_max, device=device)
            store["taus"][str(tau)][name] = {
                "evals": evals.float().numpy(),
                "basis": evecs.float().numpy(),
                "trace_d": tr_d,
                "trace_drift": tr_drift,
                "k": k,
            }
        del evals_g, evecs_g, sd
        cov_g[name] = None
        cov_d[name] = None
    for tau in taus:
        per_mod = store["taus"][str(tau)]
        mean_ratio = sum(per_mod[n]["trace_drift"] / max(per_mod[n]["trace_d"], 1e-12)
                         for n in modules) / len(modules)
        print(f"tau={tau}: mean drift ratio {mean_ratio:.4f} | "
              f"mean k {sum(per_mod[n]['k'] for n in modules)/len(modules):.1f}")
    t_eig = time.perf_counter() - t2
    store["meta"]["t_eig_s"] = t_eig
    print(f"spectral analysis {t_eig:.1f}s")

    torch.save(store, out_path)
    print("saved", out_path, f"({os.path.getsize(out_path)/1e6:.1f} MB)")

    dims = store["meta"]["dims"]
    summ = {"key": key, "t_ref_s": t_ref, "t_dom_s": t_dom, "t_eig_s": t_eig,
            "n_modules": len(modules), "taus": taus, "centered": center,
            "ref": args.ref, "n_ref_actual": len(ref_texts),
            "n_dom_actual": len(dom_texts),
            # mean drift ratio per tau, for the paper text
            "drift_ratio": {str(t): sum(store["taus"][str(t)][n]["trace_drift"]
                                        / max(store["taus"][str(t)][n]["trace_d"], 1e-12)
                                        for n in modules) / len(modules)
                            for t in taus},
            # reference-subspace dimension k per module, and where each module's
            # leading initial direction points, so the reference controls can be
            # read without downloading the profile tensors
            "k": {str(t): {n: store["taus"][str(t)][n]["k"] for n in modules}
                  for t in taus},
            "lead": {str(t): {n: lead_summary(store["taus"][str(t)][n]["basis"],
                                              store["taus"][str(t)][n]["evals"],
                                              store["taus"][str(t)][n]["trace_d"],
                                              dims[n][0]) for n in modules}
                     for t in taus}}
    if args.gev:
        summ["lead"]["gev"] = {n: lead_summary(store["gev"][n]["basis"],
                                               store["gev"][n]["energy"],
                                               store["gev"][n]["trace_d"], dims[n][0])
                               for n in modules}
    with open(os.path.join(PROFILE_DIR, key + ".json"), "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
