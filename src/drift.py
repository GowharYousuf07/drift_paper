"""DRIFT: Domain-Residual Informed Fine-Tuning.

Training-free, gradient-free, label-free profiling that decides (a) how much LoRA
rank each linear module receives and (b) which subspace its adapter is initialised in.

Core idea
---------
Existing activation-geometry methods (EVA, CorDA, AIRA, TLoRA, RSLoRA) characterise
the *target* activation distribution in isolation.  For domain adaptation the useful
question is different: which directions of the target-domain representation are ones
the pretrained model has never had to model?  We answer it by contrasting the target
covariance against that of a general-domain reference corpus:

    Sigma_G = Cov_G[x]            (reference / pretraining proxy)
    Sigma_D = Cov_D[x]            (target domain)
    P_G     = U_k U_k^T           (top-k eigenspace of Sigma_G capturing energy tau)
    Sigma~  = (I - P_G) Sigma_D (I - P_G)      <-- the *drift* (residual) covariance

Rank is allocated by greedy marginal coverage of the drift spectrum under a global
parameter budget (provably optimal, see allocate_ranks), and adapters are initialised
with the leading drift eigenvectors.

tau interpolates the method family: tau -> 0 gives k = 0, P_G = 0 and Sigma~ = Sigma_D,
i.e. plain in-domain activation PCA (EVA).  tau > 0 deflates the directions the base
model already covers.
"""
import gc
import heapq
import numpy as np
import torch


# --------------------------------------------------------------------------
# module discovery
# --------------------------------------------------------------------------
def find_target_modules(model, include_ffn=True, include_attn=True):
    """Return {name: nn.Linear} for the adaptable linear modules of an encoder."""
    import torch.nn as nn
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        low = name.lower()
        if ("embeddings" in low or low.startswith("head") or "classifier" in low
                or "pooler" in low):
            continue
        is_attn = any(k in low for k in ("query", "key", "value",
                                         "attention.output.dense",
                                         "q_proj", "k_proj", "v_proj", "out_proj",
                                         "o_proj"))          # Llama-style decoders
        is_ffn = (("intermediate.dense" in low)
                  or (low.endswith("output.dense") and "attention" not in low)
                  or "fc1" in low or "fc2" in low
                  or "gate_proj" in low or "up_proj" in low or "down_proj" in low)
        if (is_attn and include_attn) or (is_ffn and include_ffn):
            out[name] = mod
    return out


def ensure_padding(tokenizer, model=None):
    """Decoder tokenizers often ship without a padding token, and a decoder's
    sequence-classification head needs one to find each sequence's last token.
    Reuse EOS and pad on the right, as the training collate does."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


def module_cost(mod):
    """Parameters consumed per unit of rank: A is (r x d_in), B is (d_out x r)."""
    return mod.in_features + mod.out_features


# --------------------------------------------------------------------------
# streaming second-moment accumulation
# --------------------------------------------------------------------------
class _CovHook:
    """Accumulates first and second moments over non-padding token positions of a
    module input.

    Inputs are shifted by the first batch's mean before accumulation. Transformer
    activations have a large mean (and a few massive outlier dimensions), so
    forming E[xx^T] - mu mu^T directly in float32 would cancel catastrophically;
    with the shift, the final mean correction is small.
    """

    def __init__(self, d, device, dtype=torch.float32):
        self.acc = torch.zeros(d, d, device=device, dtype=dtype)
        self.sum = torch.zeros(d, device=device, dtype=dtype)
        self.shift = None
        self.n = 0
        self.mask = None

    def __call__(self, module, inputs, output):
        x = inputs[0]
        if x.dim() == 3:                       # (B, T, d)
            if self.mask is not None:
                m = self.mask.reshape(-1).bool()
                x = x.reshape(-1, x.shape[-1])[m]
            else:
                x = x.reshape(-1, x.shape[-1])
        x = x.to(self.acc.dtype)
        if self.shift is None:
            self.shift = x.mean(0)
        x = x - self.shift
        self.acc += x.T @ x
        self.sum += x.sum(0)
        self.n += x.shape[0]

    def moments(self, center=True):
        """Covariance (center=True) or raw second moment, in float64 on CPU."""
        n = max(self.n, 1)
        acc = self.acc.double().cpu() / n
        ms = self.sum.double().cpu() / n            # mean of the shifted inputs
        cov = acc - torch.outer(ms, ms)
        if center:
            return cov
        mu = ms + self.shift.double().cpu()
        return cov + torch.outer(mu, mu)


def _batched(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def _encode(tokenizer, batch, max_len, device):
    """Tokenise a batch of strings, or wrap a batch of token-id lists (the
    random-token reference) in the model's special tokens, and pad on the right."""
    if isinstance(batch[0], str):
        return tokenizer(list(batch), truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt").to(device)
    # <s> ... </s> for RoBERTa, [CLS] ... [SEP] for BERT (added by hand: recent
    # tokenizers no longer expose build_inputs_with_special_tokens)
    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    seqs = [([bos] if bos is not None else []) + list(ids)[:max_len - 2]
            + ([eos] if eos is not None else []) for ids in batch]
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tokenizer.pad_token_id, dtype=torch.long)
    am = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        am[i, :len(s)] = 1
    return {"input_ids": ids.to(device), "attention_mask": am.to(device)}


@torch.no_grad()
def collect_covariances(model, tokenizer, texts, modules, device, max_len=128,
                        batch_size=16, dtype=torch.float32, center=True):
    """Forward-only pass; returns {name: (Sigma, n_tokens)} with Sigma on CPU float32.

    Sigma is the covariance of the module input (center=True, the default, matching
    EVA's reference implementation, whose incremental PCA always centres) or the
    raw second moment (center=False).

    Kept in float32 deliberately: a full set of second moments for a 125M encoder
    is ~0.6 GB in float32 and ~1.2 GB in float64, and holding two of those
    (reference and target) in float64 is enough to push an 8 GB machine into
    swapping, which costs far more than the precision is worth. The spectral
    stage promotes one module at a time to float64.
    """
    hooks, handles = {}, []
    for name, mod in modules.items():
        h = _CovHook(mod.in_features, device, dtype)
        hooks[name] = h
        handles.append(mod.register_forward_hook(h))

    model.eval()
    keep = ("input_ids", "attention_mask", "token_type_ids")
    for batch in _batched(texts, batch_size):
        enc = _encode(tokenizer, batch, max_len, device)
        am = enc["attention_mask"]
        for h in hooks.values():
            h.mask = am
        model(**{k: v for k, v in enc.items() if k in keep})

    for h in handles:
        h.remove()
    out = {}
    for name, h in hooks.items():
        out[name] = (h.moments(center).float(), h.n)
        h.acc = h.sum = None
    del hooks
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def chunk_modules(modules, budget_bytes=400_000_000, n_corpora=2, bytes_per=8):
    """Split modules into groups whose covariance matrices fit in budget_bytes."""
    groups, cur, cur_b = [], {}, 0
    for name, mod in modules.items():
        b = mod.in_features ** 2 * bytes_per * n_corpora
        if cur and cur_b + b > budget_bytes:
            groups.append(cur)
            cur, cur_b = {}, 0
        cur[name] = mod
        cur_b += b
    if cur:
        groups.append(cur)
    return groups


# --------------------------------------------------------------------------
# subspace contrast
# --------------------------------------------------------------------------
def reference_eigh(sigma_g):
    """Descending eigendecomposition of the reference second moment.

    Computed once per module and reused across every tau -- the decomposition does
    not depend on tau, only the truncation point does.
    """
    evals, evecs = torch.linalg.eigh(sigma_g)          # ascending
    return torch.flip(evals, [0]).clamp_min(0), torch.flip(evecs, [1])


def subspace_from_eigh(evals_g, evecs_g, tau=0.95, k_max=None):
    """Truncate a precomputed reference eigenbasis at energy fraction tau."""
    d = evecs_g.shape[0]
    if tau <= 0:
        return evecs_g[:, :0], 0
    tot = evals_g.sum()
    if tot <= 0:
        return evecs_g[:, :0], 0
    csum = torch.cumsum(evals_g, 0) / tot
    k = int(torch.searchsorted(csum, torch.tensor(tau, dtype=csum.dtype)).item()) + 1
    k = min(k, d - 1)
    if k_max is not None:
        k = min(k, k_max)
    return evecs_g[:, :k].contiguous(), k


def reference_subspace(sigma_g, tau=0.95, k_max=None):
    """Convenience wrapper: eigendecompose and truncate in one call."""
    if tau <= 0:
        return torch.zeros(sigma_g.shape[0], 0, dtype=sigma_g.dtype), 0
    ev, evec = reference_eigh(sigma_g)
    return subspace_from_eigh(ev, evec, tau, k_max)


def drift_spectrum(sigma_d, v_comp, r_keep=64, device=None):
    """Spectrum of Sigma~ = (I-P) Sigma_D (I-P), computed in the complement basis.

    `v_comp` is a (d, d-k) orthonormal basis of the orthogonal complement of the
    reference subspace, i.e. the *trailing* reference eigenvectors, so that
    I - P = V V^T.  Then Sigma~ = V M V^T with M = V^T Sigma_D V, and the two
    share every nonzero eigenvalue while the eigenvectors are related by V.

    Working with M instead of Sigma~ is both cheaper and better conditioned: the
    eigendecomposition shrinks from d^3 to (d-k)^3 -- at tau = 0.95 the reference
    subspace typically absorbs most of the space, so this is a large saving --
    and forming M avoids the catastrophic cancellation of subtracting two nearly
    equal d x d matrices.

    Pass `v_comp` with zero columns to mean "no deflation" (tau = 0), in which
    case the plain spectrum of Sigma_D is returned.

    Returns (eigvals_desc, top-r_keep eigvecs in the original space,
             trace(Sigma_D), trace(Sigma~)).
    """
    trace_d = float(torch.diagonal(sigma_d).sum())
    d = sigma_d.shape[0]

    if v_comp is None:                       # explicit "no deflation"
        m, back = sigma_d, None
    elif v_comp.shape[1] == 0:               # reference subspace fills the space
        z = torch.zeros(0, dtype=sigma_d.dtype)
        return z, torch.zeros(d, 0, dtype=sigma_d.dtype), trace_d, 0.0
    else:
        # BLAS-3 work goes to the GPU in float32; the covariance was accumulated
        # in float32 anyway, so this costs no real precision.
        if device is not None and device.type == "cuda":
            sd = sigma_d.to(device=device, dtype=torch.float32)
            v = v_comp.to(device=device, dtype=torch.float32)
            m = (v.T @ (sd @ v)).double().cpu()
            del sd, v
            torch.cuda.empty_cache()
        else:
            m = v_comp.T @ (sigma_d @ v_comp)
        back = v_comp

    m = 0.5 * (m + m.T)
    evals, evecs = torch.linalg.eigh(m)
    evals = torch.flip(evals, [0]).clamp_min(0)
    evecs = torch.flip(evecs, [1])
    r = min(r_keep, evecs.shape[1])
    top = evecs[:, :r]
    if back is not None:
        top = back.to(top.dtype) @ top       # map back to the original space
    return evals, top.contiguous(), trace_d, float(evals.sum())


def gev_basis(sigma_d, sigma_g, shrink=0.1, r_keep=64):
    """Generalised eigenvectors of the pencil (Sigma_D, Sigma_G).

    Solves Sigma_D v = mu Sigma_G' v with Sigma_G' = (1 - shrink) Sigma_G
    + shrink * (tr Sigma_G / d) I, a shrinkage estimate that keeps the pencil
    well conditioned. The leading v maximise the ratio of target to reference
    energy v^T Sigma_D v / v^T Sigma_G' v -- the contrast that whitening by the
    reference covariance followed by PCA optimises, of which hard deflation (DRIFT)
    and per-direction rescaling (whitened EVA) are two approximations. In signal
    processing this is the common-spatial-patterns criterion.

    Returns (mu descending, top-r_keep directions as unit-norm columns,
    target energy of each returned direction v^T Sigma_D v).
    """
    d = sigma_g.shape[0]
    sd = sigma_d.double()
    sg = sigma_g.double()
    sg = (1.0 - shrink) * sg + shrink * (torch.diagonal(sg).sum() / d) \
        * torch.eye(d, dtype=sg.dtype)
    L = torch.linalg.cholesky(0.5 * (sg + sg.T))
    # C = L^-1 Sigma_D L^-T, symmetric; its eigenvectors w give v = L^-T w
    x = torch.linalg.solve_triangular(L, sd, upper=False)
    c = torch.linalg.solve_triangular(L, x.T, upper=False)
    c = 0.5 * (c + c.T)
    mu, w = torch.linalg.eigh(c)
    mu = torch.flip(mu, [0]).clamp_min(0)
    w = torch.flip(w, [1])[:, :r_keep]
    v = torch.linalg.solve_triangular(L.T, w, upper=True)
    v = v / torch.linalg.norm(v, dim=0, keepdim=True).clamp_min(1e-12)
    energy = ((sd @ v) * v).sum(0)
    return mu, v.contiguous(), energy


# --------------------------------------------------------------------------
# budgeted rank allocation
# --------------------------------------------------------------------------
def allocate_ranks(spectra, costs, budget_params, r_min=0, r_max=64,
                   score_mode="relative", norms=None, sensitivity=None):
    """Greedy marginal-gain allocation of a global parameter budget.

    spectra   : {name: 1-D descending array of drift eigenvalues}
    costs     : {name: parameters consumed per unit rank}
    norms     : {name: trace(Sigma_D)} used when score_mode == "relative"
    budget    : total adapter parameters available

    Objective:  max  sum_m w_m * sum_{i<=r_m} lambda_hat_{m,i}
                s.t. sum_m r_m * c_m <= B.

    Each per-module value function is concave in r_m because the eigenvalues are
    sorted descending, so this separable concave knapsack is solved exactly by
    greedy marginal-value-per-parameter selection -- no search, no training.
    """
    vals = {}
    for name, ev in spectra.items():
        ev = np.asarray(ev, dtype=np.float64)
        if score_mode == "relative":
            denom = float(norms[name]) if norms and norms.get(name, 0) > 0 else max(ev.sum(), 1e-12)
            v = ev / denom
        else:
            v = ev
        if sensitivity is not None:
            v = v * float(sensitivity.get(name, 1.0))
        vals[name] = v

    ranks = {n: 0 for n in spectra}
    spent = 0
    if r_min > 0:
        for n in spectra:
            k = min(r_min, len(vals[n]), r_max)
            ranks[n] = k
            spent += k * costs[n]

    heap = []
    for n in spectra:
        r = ranks[n]
        if r < min(r_max, len(vals[n])):
            heapq.heappush(heap, (-vals[n][r] / costs[n], n, r))
    while heap and spent < budget_params:
        _, n, r = heapq.heappop(heap)
        if ranks[n] != r:
            continue
        if spent + costs[n] > budget_params:
            break
        ranks[n] = r + 1
        spent += costs[n]
        if r + 1 < min(r_max, len(vals[n])):
            heapq.heappush(heap, (-vals[n][r + 1] / costs[n], n, r + 1))
    return ranks, spent


def uniform_ranks(modules, budget_params, r_cap=None):
    """Largest uniform rank fitting the budget (the matched-budget LoRA baseline)."""
    total_cost = sum(module_cost(m) for m in modules.values())
    r = int(budget_params // total_cost)
    if r_cap is not None:
        r = min(r, r_cap)
    r = max(r, 1)
    return {n: r for n in modules}, r * total_cost
