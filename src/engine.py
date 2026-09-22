"""Training / evaluation engine shared by every method in the comparison."""
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common               # noqa: E402
import drift as drift_mod   # noqa: E402
import peft_methods as pm   # noqa: E402

HEAD_KEYS = ("classifier", "score", "pooler")


# --------------------------------------------------------------------------
# data plumbing
# --------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = labels
        self.lengths = [len(x) for x in self.enc["input_ids"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["label"] = self.labels[i]
        return item


def make_collate(pad_id, multilabel):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, am, tt = [], [], []
        has_tt = "token_type_ids" in batch[0]
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            ids.append(b["input_ids"] + [pad_id] * pad)
            am.append([1] * n + [0] * pad)
            if has_tt:
                tt.append(b["token_type_ids"] + [0] * pad)
        out = {"input_ids": torch.tensor(ids, dtype=torch.long),
               "attention_mask": torch.tensor(am, dtype=torch.long)}
        if has_tt:
            out["token_type_ids"] = torch.tensor(tt, dtype=torch.long)
        if multilabel:
            out["labels"] = torch.tensor(np.stack([b["label"] for b in batch]),
                                         dtype=torch.float)
        else:
            out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out
    return collate


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, then sort within mega-batches so padding waste stays low."""

    def __init__(self, lengths, batch_size, seed, mega=50):
        self.lengths = lengths
        self.bs = batch_size
        self.seed = seed
        self.mega = mega
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = random.Random(self.seed * 1000 + self.epoch)
        idx = list(range(len(self.lengths)))
        g.shuffle(idx)
        chunk = self.bs * self.mega
        out = []
        for i in range(0, len(idx), chunk):
            block = idx[i:i + chunk]
            block.sort(key=lambda j: self.lengths[j])
            batches = [block[j:j + self.bs] for j in range(0, len(block), self.bs)]
            g.shuffle(batches)
            for b in batches:
                out.extend(b)
        return iter(out)


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------
class LinearHead(nn.Module):
    """A single linear layer on the first token's final hidden state: the
    minimal classification head, replacing RoBERTa's dense-tanh-linear head
    (0.60M parameters) so that the trainable component shared by all methods
    shrinks to hidden x labels."""

    def __init__(self, hidden, num_labels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, num_labels)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features, **kwargs):
        return self.out_proj(self.dropout(features[:, 0, :]))


def build_model(model_name, num_labels, multilabel, head="default"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {"num_labels": num_labels}
    if multilabel:
        kw["problem_type"] = "multi_label_classification"
    # Recent transformers load a checkpoint in its stored dtype; bf16 checkpoints
    # (e.g. SmolLM2) would give bf16 adapters, which the fp16 GradScaler cannot
    # unscale. Train from fp32 master weights for every backbone, as the fp32
    # encoder checkpoints already are.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kw).float()
    drift_mod.ensure_padding(tok, model)
    if head == "linear":
        if not hasattr(model, "classifier") or not hasattr(model.classifier, "dense"):
            raise ValueError("the linear-head control is defined for RoBERTa-style heads")
        model.classifier = LinearHead(model.config.hidden_size, num_labels,
                                      model.config.hidden_dropout_prob)
    return model, tok


def _is_head(name):
    return any(k in name.split(".") for k in HEAD_KEYS)


def apply_method(model, method, budget_rank=8, alpha=16.0, dropout=0.0,
                 profile=None, tau=0.95, score_mode="relative", rho=2.0,
                 r_min=0, init_mode="drift", alloc_mode="drift",
                 target="all", seed=0, scale="adjusted", scaling="alpha_r"):
    """Configure `model` for `method`; returns an info dict describing the budget."""
    modules = drift_mod.find_target_modules(
        model,
        include_attn=target in ("all", "attn"),
        include_ffn=target in ("all", "ffn"))
    costs = {n: drift_mod.module_cost(m) for n, m in modules.items()}
    budget_params = budget_rank * sum(costs.values())
    info = {"n_modules": len(modules), "budget_rank": budget_rank,
            "budget_params": budget_params, "method": method}

    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if _is_head(n):
            p.requires_grad_(True)

    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        info["ranks"] = {}
        return info

    if method == "linear":
        info["ranks"] = {}
        return info

    if method == "bitfit":
        for n, p in model.named_parameters():
            if n.endswith(".bias") and "embeddings" not in n:
                p.requires_grad_(True)
        info["ranks"] = {}
        return info

    if method in ("lora", "dora", "pissa"):
        ranks, spent = drift_mod.uniform_ranks(modules, budget_params)
        # rsLoRA's alpha / sqrt(r) (Kalajdzievski 2023) instead of alpha / r
        mode = "rslora" if scaling == "rslora" else "alpha_over_r"
        layers = pm.inject_adapters(model, ranks, alpha=alpha, dropout=dropout,
                                    kind=("dora" if method == "dora" else "lora"),
                                    scaling_mode=mode)
        if method == "pissa":
            for l in layers.values():
                pm.init_pissa(l)
        info.update(ranks=ranks, spent_params=spent)

    elif method == "adalora":
        r_init = int(math.ceil(1.5 * budget_rank))
        ranks = {n: r_init for n in modules}
        layers = pm.inject_adapters(model, ranks, alpha=alpha, dropout=dropout,
                                    kind="adalora")
        info.update(ranks=ranks, spent_params=sum(r_init * costs[n] for n in modules),
                    target_rank_total=budget_rank * len(modules),
                    init_rank_total=r_init * len(modules))
        info["adalora_layers"] = layers

    elif method in ("eva", "eva_white", "drift", "drift_abs", "drift_nodeflate", "gev"):
        if profile is None:
            raise ValueError(method + " requires a cached profile")
        use_tau = 0.0 if method in ("eva", "eva_white", "drift_nodeflate", "gev") else tau
        key = str(use_tau)
        if key not in profile["taus"]:
            key = min(profile["taus"], key=lambda k: abs(float(k) - use_tau))
        per_mod = profile["taus"][key]
        spectra = {n: per_mod[n]["evals"] for n in modules if n in per_mod}
        norms = {n: per_mod[n]["trace_d"] for n in spectra}
        sm = "absolute" if method == "drift_abs" else score_mode
        r_cap = int(rho * budget_rank) if rho else 64
        if method == "gev":
            # the generalised-eigenvector initialisation is compared at uniform
            # rank, so only the choice of subspace differs from LoRA
            alloc_mode, init_mode = "uniform", "gev"
        if alloc_mode == "uniform":
            ranks, spent = drift_mod.uniform_ranks(
                {n: modules[n] for n in spectra}, budget_params)
        elif alloc_mode == "units":
            # EVA's own rule: the budget is a count of rank units (rank r per
            # module on average), so an FFN rank unit costs the same as an
            # attention one and the parameters spent float with the allocation
            ranks, _ = drift_mod.allocate_ranks(
                spectra, {n: 1 for n in spectra}, budget_rank * len(spectra),
                r_min=r_min, r_max=min(r_cap, 64), score_mode=sm, norms=norms)
            spent = sum(r * costs[n] for n, r in ranks.items())
        else:
            ranks, spent = drift_mod.allocate_ranks(
                spectra, {n: costs[n] for n in spectra}, budget_params,
                r_min=r_min, r_max=min(r_cap, 64), score_mode=sm, norms=norms)
        layers = pm.inject_adapters(model, ranks, alpha=alpha, dropout=dropout,
                                    kind="lora")
        if scale == "adjusted":
            # EVA's reference implementation rescales alpha with the allocated
            # rank (alpha * r_m / r), so every module keeps the scaling alpha / r
            # of the uniform budget rank and redistribution does not change any
            # module's effective learning rate.
            for l in layers.values():
                l.scaling = alpha / budget_rank
        if init_mode == "rand_ortho":
            # control: same rank allocation and same initialisation *scale* as the
            # drift basis, but a randomly chosen subspace.
            g = torch.Generator().manual_seed(seed)
            for n, l in layers.items():
                d_in = l.base.in_features
                q, _ = torch.linalg.qr(torch.randn(d_in, l.r, generator=g))
                pm.init_subspace(l, q)
        elif init_mode == "gev":
            if "gev" not in profile:
                raise ValueError("the GEV initialisation needs a --gev profile")
            for n, l in layers.items():
                pm.init_subspace(l, torch.from_numpy(profile["gev"][n]["basis"]))
        elif init_mode != "random":
            for n, l in layers.items():
                basis = torch.from_numpy(per_mod[n]["basis"])
                pm.init_subspace(l, basis)
                if method == "eva_white":
                    # Whitening: rescale each principal row so its response
                    # variance a^T Sigma a equals the mean eigenvalue tr(Sigma)/d,
                    # i.e. the response of a random unit direction. Top
                    # eigenvalues exceed the mean, so rows only ever shrink.
                    ev = torch.as_tensor(per_mod[n]["evals"], dtype=torch.float64)
                    k = min(l.r, basis.shape[1], ev.numel())
                    lam_bar = per_mod[n]["trace_d"] / l.base.in_features
                    f = torch.sqrt(lam_bar / ev[:k].clamp_min(lam_bar)).to(l.lora_A.dtype)
                    l.lora_A.data[:k] *= f[:, None]
        info.update(ranks=ranks, spent_params=spent, tau_used=float(key),
                    score_mode=sm, r_cap=r_cap, init_mode=init_mode,
                    alloc_mode=alloc_mode)
    else:
        raise ValueError("unknown method " + method)

    pm.set_trainable_adapters(model)
    for n, p in model.named_parameters():
        if _is_head(n):
            p.requires_grad_(True)
    return info


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def pack_preds(p, multilabel):
    """Per-example predictions in a compact JSON-able form: the class index for
    single-label tasks, the bitmask of predicted labels (threshold 0.5) for
    multi-label ones. Enough to recompute any instance-level statistic."""
    if multilabel:
        bits = (p >= 0.5).astype(np.int64)
        return [int(sum(int(b) << j for j, b in enumerate(row))) for row in bits]
    return [int(x) for x in p]


@torch.no_grad()
def evaluate(model, loader, device, multilabel, num_labels, amp=False,
             return_preds=False):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(amp and device.type == "cuda")):
            logits = model(**batch).logits
        logits = logits.float().cpu()
        if multilabel:
            preds.append(torch.sigmoid(logits).numpy())
            gold.append(labels.numpy())
        else:
            preds.append(logits.argmax(-1).numpy())
            gold.append(labels.numpy())
    p = np.concatenate(preds)
    g = np.concatenate(gold)
    m = common.multilabel_metrics(g, p) if multilabel else common.clf_metrics(g, p, num_labels)
    if return_preds:
        return m, pack_preds(p, multilabel), pack_preds(g, multilabel)
    return m


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_eval(model, tok, task, device, method_info, epochs=10, lr=3e-4,
               batch_size=32, eval_batch_size=64, max_len=128, seed=0,
               weight_decay=0.01, warmup_frac=0.06, max_grad_norm=1.0,
               ortho_lambda=0.1, log_every=0, amp=False, save_preds=False):
    common.set_seed(seed)
    multilabel = task["multilabel"]
    tr_t, tr_y = task["splits"]["train"]
    dv_t, dv_y = task["splits"]["dev"]
    te_t, te_y = task["splits"]["test"]

    ds_tr = TextDataset(tr_t, tr_y, tok, max_len)
    ds_dv = TextDataset(dv_t, dv_y, tok, max_len)
    ds_te = TextDataset(te_t, te_y, tok, max_len)
    collate = make_collate(tok.pad_token_id, multilabel)

    sampler = LengthGroupedSampler(ds_tr.lengths, batch_size, seed)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler,
                       collate_fn=collate, num_workers=0, drop_last=False)
    dl_dv = DataLoader(ds_dv, batch_size=eval_batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n
                      or "lora_E" in n or "dora_m" in n) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    total_steps = max(1, epochs * math.ceil(len(ds_tr) / batch_size))
    warmup = int(warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ada_layers = method_info.get("adalora_layers")
    controller = None
    if ada_layers:
        controller = pm.AdaLoRAController(
            ada_layers, method_info["target_rank_total"],
            method_info["init_rank_total"], total_steps)

    metric_key = task["metric"]
    best = {"dev": -1.0, "epoch": -1}
    best_state = None
    step = 0
    t_start = time.perf_counter()
    peak_mem = 0
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # per-epoch telemetry: training loss, gradient norm before clipping, and (under
    # fp16) the steps the loss scaler skipped for inf/nan gradients, so a
    # divergence can be told apart from a slow start after the fact
    history = []

    for ep in range(epochs):
        sampler.epoch = ep
        model.train()
        ep_loss, ep_norm, ep_max, ep_skip, ep_n, ep_nonfinite = 0.0, 0.0, 0.0, 0, 0, 0
        for batch in dl_tr:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**batch)
                loss = out.loss
            if ada_layers and ortho_lambda > 0:
                pen = sum(l.ortho_penalty() for l in ada_layers.values())
                loss = loss + ortho_lambda * pen.float() / max(len(ada_layers), 1)
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(opt)      # so clipping and AdaLoRA see true grads
            gnorm = None
            if max_grad_norm:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm)
            if controller is not None:
                controller.step(step)
            scale_before = scaler.get_scale() if use_amp else None
            scaler.step(opt)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                ep_skip += 1
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            lv = float(loss.detach())
            if math.isfinite(lv):
                ep_loss += lv
            else:
                ep_nonfinite += 1
            if gnorm is not None:
                g = float(gnorm)
                if math.isfinite(g):
                    ep_norm += g
                    ep_max = max(ep_max, g)
            ep_n += 1
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        dv = evaluate(model, dl_dv, device, multilabel, task["num_labels"], amp=amp)
        history.append({"epoch": ep, "dev": dv[task["metric"]],
                        "train_loss": ep_loss / max(ep_n - ep_nonfinite, 1),
                        "grad_norm_mean": ep_norm / max(ep_n, 1),
                        "grad_norm_max": ep_max, "amp_skipped_steps": ep_skip,
                        "nonfinite_loss_steps": ep_nonfinite,
                        "loss_scale": scaler.get_scale() if use_amp else None})
        if dv[metric_key] > best["dev"]:
            best = {"dev": dv[metric_key], "epoch": ep, "dev_all": dv}
            best_state = {n: p.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad}
            if ada_layers:
                best_state["__masks__"] = {n: l.mask.detach().cpu().clone()
                                           for n, l in ada_layers.items()}
        if log_every:
            print(f"  ep{ep} dev {dv[metric_key]:.4f}", flush=True)

    train_time = time.perf_counter() - t_start

    if best_state is not None:
        masks = best_state.pop("__masks__", None)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n].to(device))
            if masks and ada_layers:
                for n, l in ada_layers.items():
                    l.mask.copy_(masks[n].to(device))

    test_preds = test_gold = None
    if save_preds:
        te, test_preds, test_gold = evaluate(model, dl_te, device, multilabel,
                                             task["num_labels"], amp=amp,
                                             return_preds=True)
    else:
        te = evaluate(model, dl_te, device, multilabel, task["num_labels"], amp=amp)
    total, trainable = common.count_params(model)
    head = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and _is_head(n))

    res = {"test": te, "dev_best": best.get("dev_all", {}), "best_epoch": best["epoch"],
           "train_time_s": train_time, "peak_mem_bytes": int(peak_mem),
           "params_total": total, "params_trainable": trainable,
           "params_head": head, "params_adapter": trainable - head,
           "steps": step, "history": history}
    if save_preds:
        # test predictions at the selected epoch, in test-set order; the gold
        # labels travel with them so the record is self-contained
        res["test_preds"] = test_preds
        res["test_gold"] = test_gold
    if controller is not None:
        res["adalora_final_active_rank"] = controller.active_rank_total()
        # AdaLoRA holds 1.5x the target budget during training and prunes down to
        # it. Reporting the raw parameter count would overstate what it actually
        # keeps, so we also record the post-pruning (effective) budget and compare
        # methods on that.
        res["params_adapter_effective"] = int(sum(
            int(l.mask.sum().item()) * (l.base.in_features + l.base.out_features)
            for l in ada_layers.values()))
    return res
