"""Dataset loading for biomedical classification tasks + general-domain reference corpus.

Tasks
-----
chemprot : 13-way chemical-protein relation classification, sentence level, PubMed
           abstracts with entity mentions marked by << >> and [[ ]].  Canonical
           DAPT/TAPT and BLURB task.  4169 / 2427 / 3469.
rct20k   : 5-way rhetorical-role classification of sentences in RCT abstracts
           (BACKGROUND / OBJECTIVE / METHODS / RESULTS / CONCLUSIONS).
hoc      : Hallmarks of Cancer -- 10-label multi-label classification of PubMed
           abstracts.  Reconstructed at document level (the BLURB formulation) by
           grouping the sentence-level release on PMID and taking the union of
           hallmark labels; the "no hallmark" class is dropped, so abstracts with
           no hallmark carry an all-zero target.

Reference corpus
----------------
wikitext-103 (raw) -- a general-domain proxy for the pretraining distribution, used
by DRIFT to estimate the subspace the base model has already been optimised for.
Three controls replace it (load_reference_corpus(kind=...)): CNN/DailyMail news
articles (a second general-domain corpus), WikiText with the word order shuffled
inside each passage, and uniformly random vocabulary tokens.
"""
import json
import os
import random
import re

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _subsample(texts, labels, n, seed=0):
    if n is None or n >= len(texts):
        return texts, labels
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [texts[i] for i in idx], [labels[i] for i in idx]


def _jsonl_task(folder, max_train=None, seed=0, eval_cap=None, name=""):
    raw, label_set = {}, None
    for split, fn in [("train", "train.jsonl"), ("dev", "dev.jsonl"), ("test", "test.jsonl")]:
        rows = _read_jsonl(os.path.join(DATA, folder, fn))
        raw[split] = ([r["text"] for r in rows], [r["label"] for r in rows])
        if label_set is None:
            label_set = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(label_set)}
    out = {}
    for split, (t, l) in raw.items():
        y = [l2i[x] for x in l]
        if split == "train":
            t, y = _subsample(t, y, max_train, seed)
        elif eval_cap is not None:
            # Capped with a FIXED seed so every method/seed sees the identical
            # evaluation subset; comparisons therefore stay paired.
            t, y = _subsample(t, y, eval_cap, 12345)
        out[split] = (t, y)
    return {"splits": out, "num_labels": len(label_set), "labels": label_set,
            "multilabel": False, "metric": "micro_f1", "name": name}


def load_chemprot(max_train=None, seed=0):
    return _jsonl_task("chemprot", max_train, seed, eval_cap=None, name="chemprot")


def load_rct20k(max_train=5000, seed=0):
    return _jsonl_task("rct20k", max_train, seed, eval_cap=6000, name="rct20k")


def load_hoc(max_train=None, seed=0):
    import pandas as pd
    import numpy as np
    NONE_CLASS = 7
    frames = {}
    for split, fn in [("train", "train.parquet"), ("dev", "validation.parquet"),
                      ("test", "test.parquet")]:
        df = pd.read_parquet(os.path.join(DATA, "hoc", fn))
        df["pmid"] = df["document_id"].str.split("_").str[0]
        df["sidx"] = df["document_id"].str.split("_").str[1].astype(int)
        g = df.sort_values(["pmid", "sidx"]).groupby("pmid")
        texts = g["text"].apply(lambda s: " ".join(s))
        labs = g["label"].apply(
            lambda s: sorted({int(x) for l in s for x in l if int(x) != NONE_CLASS}))
        frames[split] = (texts.tolist(), labs.tolist())

    present = sorted({x for _, labs in frames.values() for l in labs for x in l})
    l2i = {c: i for i, c in enumerate(present)}
    out = {}
    for split, (t, labs) in frames.items():
        y = np.zeros((len(labs), len(present)), dtype="float32")
        for i, l in enumerate(labs):
            for c in l:
                y[i, l2i[c]] = 1.0
        y = [row for row in y]
        if split == "train":
            t, y = _subsample(t, y, max_train, seed)
        out[split] = (t, y)
    return {"splits": out, "num_labels": len(present), "labels": present,
            "multilabel": True, "metric": "example_f1", "name": "hoc"}


# MTSamples labels that name a document type or a catch-all rather than a
# medical specialty; their notes are duplicated under specific specialties.
MTS_DROP = {"Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes",
            "Discharge Summary", "Emergency Room Reports", "Office Notes", "Letters",
            "IME-QME-Work Comp etc.", "General Medicine"}
MTS_MIN = 70          # keep specialties with at least this many unambiguous notes
# the release's ClassLabel order (its names carry a leading space, stripped here)
MTS_NAMES = [
    "Pain Management", "Chiropractic", "Podiatry", "Pediatrics - Neonatal",
    "Discharge Summary", "Cosmetic / Plastic Surgery", "Neurology", "Endocrinology",
    "Rheumatology", "Orthopedic", "Dentistry", "Allergy / Immunology",
    "Psychiatry / Psychology", "Consult - History and Phy.", "Dermatology",
    "Radiology", "Speech - Language", "Physical Medicine - Rehab", "Sleep Medicine",
    "Hospice - Palliative Care", "Diets and Nutritions", "Urology",
    "ENT - Otolaryngology", "Gastroenterology", "Letters", "Surgery", "Bariatrics",
    "Ophthalmology", "Neurosurgery", "Emergency Room Reports", "Nephrology",
    "Lab Medicine - Pathology", "Office Notes", "Cardiovascular / Pulmonary",
    "SOAP / Chart / Progress Notes", "Autopsy", "General Medicine",
    "IME-QME-Work Comp etc.", "Obstetrics / Gynecology", "Hematology - Oncology"]


def load_mtsamples(max_train=None, seed=0):
    """Clinical-style specialty classification from MTSamples transcriptions.

    The public release (galileo-ai/medical_transcription_40, 4,500 + 500 notes,
    40 labels) mixes specialties with document types and lists many notes under
    several labels. We pool its two splits, drop the document-type and catch-all
    labels (MTS_DROP), drop every note that appears under more than one remaining
    label, keep the specialties with at least MTS_MIN notes, and split each
    specialty 70/15/15 into train/dev/test with a fixed seed, so every method sees
    identical data. Scored with micro-F1."""
    import pyarrow.parquet as pq
    rows = []
    names = list(MTS_NAMES)
    for fn in ("train.parquet", "test.parquet"):
        p = os.path.join(DATA, "mtsamples", fn)
        meta = pq.read_schema(p).metadata or {}
        if b"huggingface" in meta:
            # the file's own label order, if it carries one, must agree
            feats = json.loads(meta[b"huggingface"]).get("info", {}).get("features", {})
            own = [n.strip() for n in feats.get("label", {}).get("names", [])]
            if own and own != names:
                raise ValueError("MTSamples label order differs from MTS_NAMES")
        t = pq.read_table(p).to_pydict()
        rows += [(x.strip(), int(y)) for x, y in zip(t["text"], t["label"])
                 if isinstance(x, str) and x.strip()]
    labels_of = {}
    for x, y in rows:
        labels_of.setdefault(x, set()).add(names[y])
    # keep a note when exactly one specialty remains once the dropped labels are
    # removed: a note cross-listed under "Surgery" and "Orthopedic" is an
    # orthopedic note; one listed under two specialties is ambiguous and dropped
    keep = {x: next(iter(ls - MTS_DROP)) for x, ls in labels_of.items()
            if len(ls - MTS_DROP) == 1}
    by = {}
    for x, lab in sorted(keep.items()):
        by.setdefault(lab, []).append(x)
    classes = sorted(lab for lab, xs in by.items() if len(xs) >= MTS_MIN)
    rng = random.Random(seed + 2024)
    split = {"train": ([], []), "dev": ([], []), "test": ([], [])}
    for ci, lab in enumerate(classes):
        xs = list(by[lab])
        rng.shuffle(xs)
        n_dev = n_test = max(1, round(0.15 * len(xs)))
        parts = {"test": xs[:n_test], "dev": xs[n_test:n_test + n_dev],
                 "train": xs[n_test + n_dev:]}
        for s, part in parts.items():
            split[s][0].extend(part)
            split[s][1].extend([ci] * len(part))
    out = {}
    for s, (t, y) in split.items():
        order = list(range(len(t)))
        random.Random(seed + 7).shuffle(order)
        t, y = [t[i] for i in order], [y[i] for i in order]
        if s == "train":
            t, y = _subsample(t, y, max_train, seed)
        out[s] = (t, y)
    return {"splits": out, "num_labels": len(classes), "labels": classes,
            "multilabel": False, "metric": "micro_f1", "name": "mtsamples"}


def load_task(name, max_train=None, seed=0):
    if name == "chemprot":
        return load_chemprot(max_train, seed)
    if name == "rct20k":
        return load_rct20k(max_train if max_train is not None else 5000, seed)
    if name == "hoc":
        return load_hoc(max_train, seed)
    if name == "mtsamples":
        return load_mtsamples(max_train, seed)
    raise ValueError("unknown task " + name)


TASK_MAXLEN = {"chemprot": 128, "rct20k": 96, "hoc": 512, "mtsamples": 512}


REFERENCE_KINDS = ("wikitext", "news", "shuffled", "random")


def _clean_news(t):
    """Strip the CNN/DailyMail bylines and time stamps that open many articles."""
    head = t[:400]
    cut = 0
    for pat in (r"UPDATED:\s*\.\s*[^.]*\.\s*", r"PUBLISHED:\s*\.\s*[^.]*\.\s*",
                r"Last updated at[^.]*\.\s*"):
        for m in re.finditer(pat, head):
            cut = max(cut, m.end())
    t = t[cut:]
    m = re.match(r"^.{0,80}?\(CNN\)\s*--\s*", t)
    if m:
        t = t[m.end():]
    return re.sub(r"\s+\.\s+", ". ", t).strip()


def load_reference_corpus(n_docs=2000, min_chars=200, seed=0, kind="wikitext",
                          tokenizer=None, n_tokens=126):
    """General-domain reference text.

    kind="wikitext" : WikiText-103 paragraphs (validation + test), the default.
    kind="news"     : CNN/DailyMail news articles (test split), a second
                      general-domain corpus.
    kind="shuffled" : the WikiText passages with their word order shuffled inside
                      each passage (same words, no syntax).
    kind="random"   : lists of n_tokens token ids drawn uniformly from the
                      tokenizer's vocabulary (special tokens excluded); these are
                      fed to the model as ids, never re-tokenised.
    """
    import pandas as pd
    if kind == "random":
        import numpy as np
        rng = np.random.default_rng(seed)
        special = set(tokenizer.all_special_ids)
        vocab = np.array([i for i in range(tokenizer.vocab_size) if i not in special])
        return [vocab[rng.integers(0, len(vocab), size=n_tokens)].tolist()
                for _ in range(n_docs)]
    if kind == "news":
        df = pd.read_parquet(os.path.join(DATA, "reference", "cnn_dailymail_test.parquet"))
        texts = [_clean_news(t) for t in df["article"].tolist() if isinstance(t, str)]
        texts = [t for t in texts if len(t) >= min_chars]
    else:
        dfs = []
        for fn in ("wikitext_val.parquet", "wikitext_test.parquet"):
            p = os.path.join(DATA, "reference", fn)
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))
        df = pd.concat(dfs, ignore_index=True)
        texts = [t.strip() for t in df["text"].tolist() if isinstance(t, str)]
        texts = [t for t in texts if len(t) >= min_chars and not t.startswith("=")]
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:n_docs]
    if kind == "shuffled":
        srng = random.Random(seed + 1)
        out = []
        for t in texts:
            words = t.split()
            srng.shuffle(words)
            out.append(" ".join(words))
        texts = out
    return texts
