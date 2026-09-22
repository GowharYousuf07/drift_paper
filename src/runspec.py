"""Command-line spec of one run and its result id.

Kept free of torch/transformers imports so the grid can decide which runs are
already finished without paying for a model-library import per command.
"""
import argparse

NEEDS_PROFILE = {"eva", "eva_white", "drift", "drift_abs", "drift_nodeflate", "gev"}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--task", default="chemprot")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget_rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--score_mode", default="relative")
    ap.add_argument("--init_mode", default="drift")
    ap.add_argument("--alloc_mode", default="drift")
    ap.add_argument("--r_min", type=int, default=0)
    ap.add_argument("--target", default="all")
    ap.add_argument("--cov", default="centered", choices=["centered", "raw"],
                    help="profile type for EVA/DRIFT: covariance (as in EVA's "
                         "reference implementation) or raw second moment")
    ap.add_argument("--scale", default="adjusted", choices=["adjusted", "rank"],
                    help="adapter scaling for rank-redistributing methods: "
                         "'adjusted' keeps alpha/r of the uniform budget rank for "
                         "every module (EVA's default), 'rank' uses alpha/r_m")
    ap.add_argument("--n_ref", type=int, default=1024)
    ap.add_argument("--n_dom", type=int, default=1024)
    ap.add_argument("--ref", default="wikitext",
                    help="reference corpus of the profile (see data.REFERENCE_KINDS)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast; use on Turing+ GPUs, NOT on Pascal "
                         "(GP10x runs fp16 at 1/64 rate)")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic kernels (for fp32 reruns: two runs with the "
                         "same seed then give the same result)")
    ap.add_argument("--head", default="default", choices=["default", "linear"],
                    help="classification head: the backbone's own (RoBERTa: dense, "
                         "tanh, linear) or a single linear layer")
    ap.add_argument("--scaling", default="alpha_r", choices=["alpha_r", "rslora"],
                    help="LoRA scale for uniform-rank methods: alpha/r, or rsLoRA's "
                         "alpha/sqrt(r)")
    ap.add_argument("--no_save_preds", action="store_true",
                    help="do not store per-example test predictions")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap


def finalize(a):
    """Settings implied by others. The GEV initialisation is compared at uniform
    rank, so its id records that explicitly."""
    if a.method == "gev":
        a.init_mode, a.alloc_mode = "gev", "uniform"
    return a


def parse(argv=None):
    return finalize(build_parser().parse_args(argv))


def profile_key(model_name, task, n_ref, n_dom, center=True, ref="wikitext",
                gev=False):
    safe = model_name.replace("/", "__")
    # centred (covariance) profiles are the default; the suffix keeps them from
    # ever being confused with raw second-moment profiles cached earlier. The
    # default WikiText reference carries no suffix, so existing keys are unchanged.
    return (f"{safe}__{task}__ref{n_ref}__dom{n_dom}"
            + ("" if ref == "wikitext" else f"__{ref}")
            + ("__cen" if center else "") + ("__gev" if gev else ""))


def run_id(a):
    bits = [a.model.replace("/", "__"), a.task, a.method, f"r{a.budget_rank}",
            f"lr{a.lr:g}", f"s{a.seed}"]
    if a.method in NEEDS_PROFILE:
        bits.append(f"tau{a.tau:g}")
        bits.append(a.score_mode)
        bits.append(f"init-{a.init_mode}")
        bits.append(f"alloc-{a.alloc_mode}")
        bits.append(f"rho{a.rho:g}")
        bits.append(f"cov-{a.cov}")
        bits.append(f"sc-{a.scale}")
        # only non-default profiles add to the id, so existing ids are unchanged
        if a.ref != "wikitext":
            bits.append(f"ref-{a.ref}")
        if (a.n_ref, a.n_dom) != (1024, 1024):
            bits.append(f"prof{a.n_ref}-{a.n_dom}")
    if a.deterministic:
        bits.append("det")
    # like the profile bits above, only non-default values add to the id
    if getattr(a, "head", "default") != "default":
        bits.append(f"head-{a.head}")
    if getattr(a, "scaling", "alpha_r") != "alpha_r":
        bits.append(f"scal-{a.scaling}")
    if a.target != "all":
        bits.append("tgt-" + a.target)
    if a.max_train:
        bits.append(f"n{a.max_train}")
    if a.tag:
        bits.append(a.tag)
    return "__".join(bits)
