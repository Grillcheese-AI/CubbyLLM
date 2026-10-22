"""fetch_hdc_labels -- the English and French labels of the hdc set's entities that `hdc_names.tsv.gz` lacks, from
the Wikidata API (order 0014; Nick chose the API over a dump, 2026-10-22).

The set's triplet columns name 3,489,229 entities by Q-id; the local table labels ~881k of them. The rest are asked
of `wbgetentities` 50 at a time, in series (Wikimedia's etiquette: one request at a time, `maxlag`, a User-Agent),
labels only (en, fr). Every answer is appended to `standin/data/out/hdc_labels_api.tsv` (`qid  en  fr  how`, how =
label / missing / redirect:<qid>) as it arrives, so a stopped run resumes where it stopped and nothing is asked twice.

    python standin/data/fetch_hdc_labels.py --src <hdc folder> [--sleep 0.2] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

__wiring__ = "STANDALONE"

API = "https://www.wikidata.org/w/api.php"
OUT = ROOT / "standin" / "data" / "out" / "hdc_labels_api.tsv"
NAMES = ROOT / "standin" / "data" / "out" / "hdc_names.tsv.gz"
_QID = re.compile(r"^Q\d+$")


def entities(src: pathlib.Path) -> set[str]:
    import pyarrow.parquet as pq
    out: set[str] = set()
    for f in sorted(src.rglob("*.pq")):
        cols = set(pq.read_schema(f).names)
        use = [c for c in ("triplet_subject", "triplet_object") if c in cols]
        if not use:
            continue
        t = pq.read_table(f, columns=use)
        for c in use:
            out.update(x for x in t.column(c).to_pylist() if x and _QID.match(str(x)))
    return out


def labelled() -> set[str]:
    have: set[str] = set()
    with gzip.open(NAMES, "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] and parts[2]:
                have.add(parts[1])
    return have


def fetched() -> set[str]:
    if not OUT.exists():
        return set()
    return {line.split("\t", 1)[0] for line in OUT.read_text(encoding="utf-8").splitlines() if line}


def ask(ids: list[str], ua: str, tries: int = 6) -> dict:
    q = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(ids), "props": "labels", "languages": "en|fr",
                                "format": "json", "maxlag": "5"})
    wait = 5.0
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers={"User-Agent": ua}), timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            if d.get("error", {}).get("code") == "maxlag":
                time.sleep(wait); wait = min(wait * 2, 120); continue
            return d
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(float(e.headers.get("Retry-After") or wait)); wait = min(wait * 2, 300); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(wait); wait = min(wait * 2, 300)
    raise RuntimeError(f"no answer for {ids[0]}..{ids[-1]} after {tries} tries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the hdc folder (the parquet files)")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many requests (0: all)")
    a = ap.parse_args()
    from cubbyllm.sources.wikidata import UA
    t0 = time.time()
    want = sorted(entities(pathlib.Path(a.src)) - labelled() - fetched(), key=lambda q: int(q[1:]))
    print(f"{len(want):,} entities to ask about ({(len(want) + 49) // 50:,} requests) [{time.time() - t0:.0f}s]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("a", encoding="utf-8") as f:
        for i in range(0, len(want), 50):
            ids = want[i:i + 50]
            d = ask(ids, UA)
            for q in ids:
                e = (d.get("entities") or {}).get(q) or {}
                if "missing" in e:
                    f.write(f"{q}\t\t\tmissing\n"); continue
                lab = e.get("labels") or {}
                how = f"redirect:{e['id']}" if e.get("id") and e["id"] != q else "label"
                en, fr = (lab.get("en") or {}).get("value", ""), (lab.get("fr") or {}).get("value", "")
                f.write(f"{q}\t{en.replace(chr(9), ' ')}\t{fr.replace(chr(9), ' ')}\t{how}\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n:,} requests, {i + len(ids):,} entities [{time.time() - t0:.0f}s]", flush=True)
            if a.limit and n >= a.limit:
                break
            time.sleep(a.sleep)
    print(f"done: {n:,} requests [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
