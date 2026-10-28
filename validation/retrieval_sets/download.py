"""download -- order 0046's held-out sets and the Wikidata5M aliases, fetched into ``$CUBBY_DATA_ROOT/retrieval/raw``
through the server's fetcher (``cubby_serve.sources.http.download``). Every file is well under the 1 GB line that
needs Nick's yes; the sizes are checked before each fetch and the run stops above it. An archive is unpacked beside
itself (no file is executed; a member that would land outside its folder is refused).

    python validation/retrieval_sets/download.py [--only mintaka qald9plus ...] [--list]

Wired: STANDALONE (a data fetcher).
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import raw_dir  # noqa: E402

__wiring__ = "STANDALONE"

UA = "CubbyLLM-retrieval-bench/0.1 (order 0046; research measurement)"
LIMIT = 1 << 30
GH = "https://raw.githubusercontent.com"
SETS = {
    "mintaka": {"licence": "CC BY 4.0 (the repository's README)",
                "files": [(f"{GH}/amazon-science/mintaka/main/data/mintaka_{s}.json", f"mintaka_{s}.json") for s in ("dev", "test")]},
    "qald9plus": {"licence": "CC BY 4.0 (LICENSE)",
                  "files": [(f"{GH}/KGQA/QALD_9_plus/main/data/qald_9_plus_{s}_wikidata.json", f"qald_9_plus_{s}_wikidata.json")
                            for s in ("train", "test")]},
    "el4qa": {"licence": "released with BLINK/ELQ (MIT); the questions are WebQSP's and GraphQuestions'",
              "files": [("http://dl.fbaipublicfiles.com/elq/EL4QA_data.tar.gz", "EL4QA_data.tar.gz")]},
    "simplequestions_wd": {"licence": "not stated in the repository (SimpleQuestions itself: CC BY 3.0)",
                           "files": [(f"{GH}/askplatypus/wikidata-simplequestions/master/annotated_wd_data_{s}_answerable.txt",
                                      f"annotated_wd_data_{s}_answerable.txt") for s in ("valid", "test")]},
    "shadowlink": {"licence": "see the dataset card (vera-pro/ShadowLink)",
                   "files": [(f"https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/{f}", f)
                             for f in ("Top.json", "Shadow.json", "Neutral.json", "Tail.json", "README.md")]},
    "amber": {"licence": "CC0 1.0 (the data directory)",
              "files": [(f"{GH}/anthonywchen/AmbER-Sets/main/data/{k}/{t}/amber_sets.jsonl", f"{k}_{t}_amber_sets.jsonl")
                        for k in ("human", "nonhuman") for t in ("qa",)]
                       + [(f"{GH}/anthonywchen/AmbER-Sets/main/data/{k}/qa_templates.json", f"{k}_qa_templates.json") for k in ("human", "nonhuman")]},
    "cronquestions": {"licence": "MIT (the repository)",
                      "files": [("https://drive.usercontent.google.com/download?id=1fe7-x7ChszqzczKncoZcpwmWc1PBq1_0&export=download&confirm=t",
                                 "data_v2.zip")]},
    "wikidata5m": {"licence": "CC0 (Wikidata; Wikidata5M, Wang et al. 2021)",
                   "files": [("https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1", "wikidata5m_alias.tar.gz"),
                             ("https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1", "wikidata5m_transductive.tar.gz")]},
}


def size_of(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:                                    # noqa: BLE001 -- a HEAD some hosts refuse: the GET decides
        return None


def unpack(path: str, folder: str) -> None:
    """A tar or zip unpacked into `folder`; a member whose path leaves it is refused."""
    root = os.path.realpath(folder)

    def safe(name: str) -> str:
        dst = os.path.realpath(os.path.join(folder, name))
        if not (dst == root or dst.startswith(root + os.sep)):
            raise SystemExit(f"{path}: member {name!r} would land outside {folder}")
        return dst
    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                safe(m.name)
                if m.issym() or m.islnk():
                    raise SystemExit(f"{path}: link member {m.name!r} refused")
            t.extractall(folder)
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                safe(n)
            z.extractall(folder)


def fetch(name: str) -> dict:
    from cubby_serve.sources.http import download
    spec, out = SETS[name], raw_dir(name)
    got = {}
    for url, rel in spec["files"]:
        dest = os.path.join(out, rel)
        if not os.path.exists(dest):
            n = size_of(url)
            if n is not None and n > LIMIT:
                raise SystemExit(f"{name}/{rel} is {n / 1e9:.2f} GB: over the line that needs Nick's yes")
        got[rel] = download(url, dest, UA)
        mark = dest + ".unpacked"
        if rel.endswith((".tar.gz", ".zip")) and not os.path.exists(mark):
            unpack(dest, out)
            open(mark, "w").close()
        print(f"  {name}/{rel}: {got[rel] / 1e6:.1f} MB", flush=True)
    with open(os.path.join(out, "LICENCE.txt"), "w", encoding="utf-8") as f:
        f.write(spec["licence"] + "\n")
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SETS))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in SETS.items():
            print(f"{k:20s} {v['licence']}")
        return
    total = 0
    for name in a.only:
        print(f"{name}:", flush=True)
        total += sum(fetch(name).values())
    print(f"{total / 1e6:.1f} MB in {os.path.dirname(raw_dir('x'))}")


if __name__ == "__main__":
    main()
