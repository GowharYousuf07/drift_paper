"""Shared utilities: seeding, metrics, parameter accounting, timing."""
import json, os, random, time, hashlib
import numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- metrics
def clf_metrics(y_true, y_pred, n_classes):
    """Micro-F1 (== accuracy for single-label) and macro-F1."""
    from sklearn.metrics import f1_score, accuracy_score
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def multilabel_metrics(y_true, y_prob, thresh=0.5):
    """Example-based F1 (BLURB convention for HoC) + micro/macro F1."""
    from sklearn.metrics import f1_score
    y_pred = (y_prob >= thresh).astype(int)
    inter = (y_pred * y_true).sum(1)
    denom = y_pred.sum(1) + y_true.sum(1)
    ex_f1 = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-9), 1.0)
    return {
        "example_f1": float(ex_f1.mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

# ---------------------------------------------------------- param counting
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_adapter_params(model):
    """Trainable params excluding the task head (head is required by every method,
    so budget comparisons are made on the *adapter* parameters only)."""
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad and not name.startswith("head."):
            n += p.numel()
    return n

# ---------------------------------------------------------------- io
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

class Timer:
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *a): self.dt = time.perf_counter() - self.t0
