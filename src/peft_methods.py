"""Unified implementations of the PEFT methods compared in the paper.

Everything is implemented inside one framework so that trainable-parameter budgets,
optimiser settings and training code are *identical* across methods; only the
adapter parameterisation and its rank allocation / initialisation differ.

Methods
-------
full     : full fine-tuning (reference upper bound on trainable parameters)
linear   : linear probe -- classification head only
bitfit   : all bias terms + head                      (Ben Zaken et al., 2022)
lora     : uniform rank across modules                (Hu et al., 2022)
dora     : weight-decomposed LoRA                     (Liu et al., 2024)
pissa    : LoRA initialised from the top-r SVD of W   (Meng et al., 2024)
adalora  : SVD parameterisation + training-time importance pruning (Zhang et al., 2023)
eva      : in-domain activation PCA init + explained-variance rank redistribution
           (Paischer et al., 2025); exactly the tau = 0 case of drift
drift    : ours -- reference-contrastive drift subspace (see drift.py)
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# adapter layers
# ==========================================================================
class LoRALinear(nn.Module):
    """Frozen base linear + trainable low-rank update (optionally DoRA-style)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0,
                 use_dora: bool = False, scaling_mode: str = "alpha_over_r"):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = int(r)
        self.use_dora = use_dora
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if scaling_mode == "one":
                self.scaling = 1.0
            elif scaling_mode == "rslora":
                self.scaling = alpha / math.sqrt(self.r)
            else:
                self.scaling = alpha / self.r
            if use_dora:
                with torch.no_grad():
                    m = torch.linalg.norm(base.weight, dim=1)
                self.dora_m = nn.Parameter(m)
        else:
            self.scaling = 0.0

    def delta_w(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        if self.r == 0:
            return self.base(x)
        if not self.use_dora:
            out = self.base(x)
            h = self.drop(x) @ self.lora_A.T
            return out + (h @ self.lora_B.T) * self.scaling
        w = self.base.weight + self.delta_w()
        norm = torch.linalg.norm(w, dim=1).clamp_min(1e-8).detach()
        w = w * (self.dora_m / norm).unsqueeze(1)
        return F.linear(self.drop(x), w, self.base.bias)


class AdaLoRALinear(nn.Module):
    """SVD-style parameterisation dW = P diag(E) Q used by AdaLoRA.

    Triplets are masked (E_i <- 0) by the global budget controller; masked triplets
    stop contributing but stay allocated until the schedule ends, exactly as in the
    original formulation.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = int(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_P = nn.Parameter(torch.zeros(base.out_features, self.r))
        self.lora_E = nn.Parameter(torch.zeros(self.r))
        self.lora_Q = nn.Parameter(torch.zeros(self.r, base.in_features))
        nn.init.normal_(self.lora_P, std=0.02)
        nn.init.normal_(self.lora_Q, std=0.02)
        nn.init.zeros_(self.lora_E)
        self.scaling = alpha / self.r
        self.register_buffer("mask", torch.ones(self.r))

    def forward(self, x):
        out = self.base(x)
        e = self.lora_E * self.mask
        h = self.drop(x) @ self.lora_Q.T
        h = h * e
        return out + (h @ self.lora_P.T) * self.scaling

    def ortho_penalty(self):
        p, q = self.lora_P, self.lora_Q
        ip = p.T @ p
        iq = q @ q.T
        eye = torch.eye(self.r, device=p.device, dtype=p.dtype)
        return ((ip - eye) ** 2).sum() + ((iq - eye) ** 2).sum()


# ==========================================================================
# injection helpers
# ==========================================================================
def _get_parent(model, name):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_adapters(model, ranks, alpha=16.0, dropout=0.0, kind="lora",
                    scaling_mode="alpha_over_r"):
    """Replace the named nn.Linear modules with adapter-wrapped versions."""
    injected = {}
    for name, r in ranks.items():
        if r <= 0:
            continue
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        if kind == "adalora":
            new = AdaLoRALinear(base, r, alpha, dropout)
        else:
            new = LoRALinear(base, r, alpha, dropout,
                             use_dora=(kind == "dora"), scaling_mode=scaling_mode)
        setattr(parent, attr, new)
        injected[name] = new
    return injected


def freeze_backbone(model, head_prefixes=("classifier", "score", "head")):
    for name, p in model.named_parameters():
        p.requires_grad_(any(name.startswith(h) or ("." + h + ".") in name
                             for h in head_prefixes))


def unfreeze_head(model, head_prefixes=("classifier", "score", "head")):
    for name, p in model.named_parameters():
        if any(h in name.split(".") for h in head_prefixes):
            p.requires_grad_(True)


def set_trainable_adapters(model):
    for name, p in model.named_parameters():
        if any(k in name for k in ("lora_A", "lora_B", "lora_P", "lora_E",
                                   "lora_Q", "dora_m")):
            p.requires_grad_(True)


# ==========================================================================
# initialisation schemes
# ==========================================================================
@torch.no_grad()
def init_pissa(layer: LoRALinear):
    """A = sqrt(S_r) V_r^T, B = U_r sqrt(S_r); residual weight W - BA stays frozen."""
    w = layer.base.weight.data.double()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = layer.r
    ur, sr, vr = u[:, :r], s[:r], vh[:r, :]
    sq = torch.sqrt(sr)
    a = (torch.diag(sq) @ vr)
    b = (ur @ torch.diag(sq))
    layer.lora_A.data.copy_(a.to(layer.lora_A.dtype))
    layer.lora_B.data.copy_(b.to(layer.lora_B.dtype))
    layer.scaling = 1.0
    layer.base.weight.data.copy_((w - b @ a).to(layer.base.weight.dtype))


@torch.no_grad()
def init_subspace(layer: LoRALinear, basis: torch.Tensor):
    """A <- leading subspace directions (rows), B <- 0 so dW = 0 at initialisation.

    basis: (d_in, k) column-orthonormal, k >= layer.r in normal operation.

    If the cached basis has fewer columns than the allocated rank, the surplus
    rows are filled with random directions orthogonalised against the basis --
    never left as zeros, which would make those ranks permanently dead (a zero
    row of A gives a zero gradient to the corresponding column of B).
    """
    r = min(layer.r, basis.shape[1])
    layer.lora_A.data.zero_()
    layer.lora_A.data[:r].copy_(basis[:, :r].T.to(layer.lora_A.dtype))
    if layer.r > r:
        extra = torch.randn(layer.base.in_features, layer.r - r,
                            dtype=basis.dtype)
        extra -= basis[:, :r] @ (basis[:, :r].T @ extra)
        q, _ = torch.linalg.qr(extra)
        layer.lora_A.data[r:].copy_(q.T.to(layer.lora_A.dtype))
    layer.lora_B.data.zero_()


# ==========================================================================
# AdaLoRA budget controller
# ==========================================================================
class AdaLoRAController:
    """Global importance-based budget scheduler (Zhang et al., ICLR 2023).

    Importance of triplet i combines the sensitivity of E_i and of the corresponding
    row/column of P and Q, each smoothed by an exponential moving average, plus an
    uncertainty term.  The total budget follows a cubic schedule from b_init to
    b_target; the lowest-importance triplets are masked.
    """

    def __init__(self, layers, target_rank_total, init_rank_total,
                 total_steps, warmup_frac=0.1, final_frac=0.75,
                 beta1=0.85, beta2=0.85):
        self.layers = layers
        self.b_target = target_rank_total
        self.b_init = init_rank_total
        self.ti = int(warmup_frac * total_steps)
        self.tf = int(final_frac * total_steps)
        self.total_steps = total_steps
        self.beta1, self.beta2 = beta1, beta2
        self.ipt, self.exp_ipt, self.exp_unc = {}, {}, {}

    def _score(self, layer, name):
        with torch.no_grad():
            parts = []
            for p in (layer.lora_E, layer.lora_P, layer.lora_Q):
                if p.grad is None:
                    return None
                s = (p * p.grad).abs().detach()
                parts.append(s)
            e_s = parts[0]                                   # (r,)
            p_s = parts[1].mean(dim=0)                       # (r,)
            q_s = parts[2].mean(dim=1)                       # (r,)
            raw = e_s + p_s + q_s
            if name not in self.exp_ipt:
                self.exp_ipt[name] = torch.zeros_like(raw)
                self.exp_unc[name] = torch.zeros_like(raw)
            self.exp_ipt[name] = self.beta1 * self.exp_ipt[name] + (1 - self.beta1) * raw
            self.exp_unc[name] = (self.beta2 * self.exp_unc[name]
                                  + (1 - self.beta2) * (raw - self.exp_ipt[name]).abs())
            return self.exp_ipt[name] * self.exp_unc[name]

    def budget(self, step):
        if step <= self.ti:
            return self.b_init
        if step >= self.tf:
            return self.b_target
        frac = 1.0 - (step - self.ti) / max(self.tf - self.ti, 1)
        return int(self.b_target + (self.b_init - self.b_target) * (frac ** 3))

    def step(self, global_step):
        scores = {}
        for name, layer in self.layers.items():
            s = self._score(layer, name)
            if s is None:
                return
            scores[name] = s
        b = self.budget(global_step)
        allv = torch.cat([v for v in scores.values()])
        k = max(int(b), 1)
        if k >= allv.numel():
            for layer in self.layers.values():
                layer.mask.fill_(1.0)
            return
        thresh = torch.topk(allv, k, largest=True).values.min()
        for name, layer in self.layers.items():
            layer.mask.copy_((scores[name] >= thresh).to(layer.mask.dtype))

    def active_rank_total(self):
        return int(sum(l.mask.sum().item() for l in self.layers.values()))
