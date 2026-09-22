"""Fetch every dataset used in the paper. Idempotent; safe to re-run."""
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

S3 = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data"
HF = "https://huggingface.co/api/datasets"

FILES = [
    # ChemProt: 13-way chemical-protein relation classification (BioCreative VI),
    # in the split released with Gururangan et al. (2020).
    (f"{S3}/chemprot/train.jsonl", "chemprot/train.jsonl"),
    (f"{S3}/chemprot/dev.jsonl", "chemprot/dev.jsonl"),
    (f"{S3}/chemprot/test.jsonl", "chemprot/test.jsonl"),
    # RCT-20k: 5-way sentence-role classification in RCT abstracts.
    (f"{S3}/rct-20k/train.jsonl", "rct20k/train.jsonl"),
    (f"{S3}/rct-20k/dev.jsonl", "rct20k/dev.jsonl"),
    (f"{S3}/rct-20k/test.jsonl", "rct20k/test.jsonl"),
    # Hallmarks of Cancer, sentence-level release (aggregated to documents in data.py).
    (f"{HF}/qanastek/HoC/parquet/HoC/train/0.parquet", "hoc/train.parquet"),
    (f"{HF}/qanastek/HoC/parquet/HoC/validation/0.parquet", "hoc/validation.parquet"),
    (f"{HF}/qanastek/HoC/parquet/HoC/test/0.parquet", "hoc/test.parquet"),
    # General-domain reference corpus for the DRIFT contrast.
    (f"{HF}/Salesforce/wikitext/parquet/wikitext-103-raw-v1/validation/0.parquet",
     "reference/wikitext_val.parquet"),
    (f"{HF}/Salesforce/wikitext/parquet/wikitext-103-raw-v1/test/0.parquet",
     "reference/wikitext_test.parquet"),
    # A second general-domain reference (news), for the reference-corpus control.
    (f"{HF}/abisee/cnn_dailymail/parquet/3.0.0/test/0.parquet",
     "reference/cnn_dailymail_test.parquet"),
    # Clinical-style text: MTSamples medical transcriptions (specialty labels),
    # regrouped into a specialty-classification task in data.py.
    (f"{HF}/galileo-ai/medical_transcription_40/parquet/default/train/0.parquet",
     "mtsamples/train.parquet"),
    (f"{HF}/galileo-ai/medical_transcription_40/parquet/default/test/0.parquet",
     "mtsamples/test.parquet"),
]


def main():
    ok = True
    for url, rel in FILES:
        dst = os.path.join(DATA, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"have  {rel}")
            continue
        print(f"get   {rel} ... ", end="", flush=True)
        # the Hub intermittently answers 503; retry with backoff before giving up
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read()
                with open(dst, "wb") as f:
                    f.write(body)
                print(f"{os.path.getsize(dst)/1e6:.2f} MB")
                break
            except Exception as e:                               # noqa: BLE001
                if attempt == 5:
                    ok = False
                    print("FAILED:", e)
                else:
                    wait = 15 * 2 ** attempt
                    print(f"retry in {wait}s ({e}) ... ", end="", flush=True)
                    time.sleep(wait)
    if not ok:
        sys.exit(1)
    print("all data present")


if __name__ == "__main__":
    main()
