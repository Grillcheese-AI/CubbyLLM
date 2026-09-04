"""Rows from a Hub dataset through the datasets-server rows API, cached per page as JSON.

Wired: STANDALONE (a data helper for the stand-in builders; nothing in cubbyllm/ imports it).

No pandas/pyarrow needed, no full parquet download: the API hands back `length` rows per call (max 100),
already decoded. Good for the small human-written sources of the SFT gap map (disfl_qa, CANARD, SocialIQA,
DailyDialog, EmpatheticDialogues, the safety sets); the big ones stay local parquet (see build_chat_sft.HF_*).
Script-based datasets (no parquet conversion) answer 501 — use a parquet mirror (recorded per source).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "out", "hf_rows")          # standin/data/out is gitignored
API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length={n}"
PAGE = 100


def fetch_rows(ds: str, cfg: str, split: str, n: int, cache: str = CACHE_DIR, page: int = PAGE,
               retries: int = 6, verbose: bool = True, pause: float = 0.4) -> list[dict]:
    """The first `n` rows of ds/cfg/split (fewer if the split is shorter). Pages are cached as
    `<cache>/<owner>__<name>__<cfg>__<split>__<offset>.json`; a cached page is never re-fetched.
    The API rate-limits a burst (429 after ~15 quick pages): a short pause between pages, and a
    429 waits 15 s x attempt before retrying."""
    os.makedirs(cache, exist_ok=True)
    out: list[dict] = []
    for off in range(0, n, page):
        want = min(page, n - off)
        f = os.path.join(cache, f"{ds.replace('/', '__')}__{cfg}__{split}__{off}.json")
        if not os.path.exists(f):
            url = API.format(ds=urllib.parse.quote(ds, safe=""), cfg=cfg, split=split, off=off, n=want)
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(url, timeout=300) as r:
                        data = r.read()
                    json.loads(data)                     # a page is cached only if it parses
                    open(f, "wb").write(data)
                    time.sleep(pause)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
                    if attempt == retries - 1:
                        if verbose:
                            print(f"  rows API gave up on {ds} {split} offset {off}: {e}")
                        return out
                    code = getattr(e, "code", None)
                    time.sleep(15.0 * (attempt + 1) if code == 429 else 2.0 * (attempt + 1))
        j = json.load(open(f, encoding="utf-8"))
        rows = [r["row"] for r in j.get("rows", [])]
        out += rows
        if len(rows) < want:                             # the split ended
            break
    return out[:n]
