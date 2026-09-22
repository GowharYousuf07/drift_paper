"""Minimal Kaggle REST client for running the experiments remotely.

    python kaggle/kaggle_api.py status SLUG
    python kaggle/kaggle_api.py watch  SLUG            # poll until done, then download
    python kaggle/kaggle_api.py push   NOTEBOOK SLUG "Title" [SOURCE_REF ...]

The API token is read from "kaggle API.txt" in the project root and is never
printed. Outputs are saved under kaggle/output/<SLUG>/.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://www.kaggle.com/api/v1"
POLL_S = 600

# Accounts: {id: (username, token file in the project root)}. Account 1 is the
# default; further accounts come from kaggle_accounts.json ({"2": {"user": ...,
# "token_file": ...}}), and KAGGLE_ACCOUNT=<id> selects one. Tokens are read
# from their files and never printed.
ACCOUNTS = {"1": ("gowharyousuf", "kaggle API.txt")}
_cfg = os.path.join(PROJ, "kaggle_accounts.json")
if os.path.exists(_cfg):
    with open(_cfg, encoding="utf-8") as _f:
        for _k, _v in json.load(_f).items():
            ACCOUNTS[str(_k)] = (_v["user"], _v["token_file"])
ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "1")
USER, TOKEN_FILE = ACCOUNTS[ACCOUNT]


def _token():
    with open(os.path.join(PROJ, TOKEN_FILE), encoding="utf-8-sig") as f:
        return f.read().strip()


def _request(path, body=None):
    req = urllib.request.Request(
        API + path, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + _token(),
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read() or b"{}")


def status(slug):
    q = urllib.parse.urlencode({"userName": USER, "kernelSlug": slug})
    return _request("/kernels/status?" + q)


def download(slug, profiles=True):
    """Save the run's zip, per-GPU logs and (optionally) the RoBERTa profiles.
    Files are streamed to disk in chunks so large .pt files never sit in RAM."""
    import shutil
    out = os.path.join(PROJ, "kaggle", "output", slug)
    os.makedirs(out, exist_ok=True)
    q = urllib.parse.urlencode({"userName": USER, "kernelSlug": slug})
    d = _request("/kernels/output?" + q)
    with open(os.path.join(out, "kaggle_log.json"), "w", encoding="utf-8") as f:
        f.write(d.get("log") or "")
    for fobj in d.get("files", []):
        name, url = fobj.get("fileName", ""), fobj.get("url", "")
        # the zip, the per-GPU logs, and the RoBERTa profiles the figures read
        # (the zip deliberately omits the large .pt tensors), including the GEV
        # and reference-control profiles of the revision
        wanted = (name.endswith(".zip") or name.startswith("logs/")
                  or (profiles and name.startswith("runs/profiles/roberta-base__")
                      and (name.endswith("__cen.pt") or name.endswith("__gev.pt"))))
        if not url or not wanted:
            continue
        dest = os.path.join(out, name.replace("/", os.sep))
        local = os.path.join(PROJ, "runs", "profiles", os.path.basename(name))
        if name.endswith(".pt") and (os.path.exists(dest) or os.path.exists(local)):
            print("have", name, flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=4 << 20)
        print("downloaded", name, os.path.getsize(dest), "bytes", flush=True)
    return out


def watch(slug):
    while True:
        try:
            st = status(slug).get("status", "")
        except Exception as e:                                   # noqa: BLE001
            print(time.strftime("%H:%M"), "status check failed:", e, flush=True)
            time.sleep(POLL_S)
            continue
        print(time.strftime("%H:%M"), "status:", st, flush=True)
        if st not in ("running", "queued"):
            break
        time.sleep(POLL_S)
    print("saved to", download(slug), flush=True)
    print("FINAL STATUS:", st, flush=True)


def push(nb_path, slug, title, sources, machine_shape="NvidiaTeslaT4"):
    with open(nb_path, encoding="utf-8") as f:
        text = f.read()
    body = {
        "slug": f"{USER}/{slug}", "newTitle": title, "text": text,
        "language": "python", "kernelType": "notebook", "isPrivate": True,
        "enableGpu": True, "enableTpu": False, "enableInternet": True,
        "datasetDataSources": [], "competitionDataSources": [],
        "kernelDataSources": sources, "modelDataSources": [], "categoryIds": [],
    }
    if machine_shape:
        body["machineShape"] = machine_shape
    try:
        return _request("/kernels/push", body)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:800]}"}


def main():
    cmd, *args = sys.argv[1:]
    if cmd == "status":
        print(status(args[0]))
    elif cmd == "watch":
        watch(args[0])
    elif cmd == "push":
        nb, slug, title, *srcs = args
        resp = push(nb, slug, title, srcs)
        if resp.get("error"):
            print("first attempt failed:", resp["error"], "-- retrying without machineShape")
            resp = push(nb, slug, title, srcs, machine_shape=None)
        print(resp)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
