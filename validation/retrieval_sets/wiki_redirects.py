"""wiki_redirects -- order 0046, R1b: the EN and FR Wikipedia redirect tables (Nick's yes, 2026-10-31; decision 5).

A redirect is another name for an article ("JFK" -> "John F. Kennedy"). MediaWiki's dumps keep it in two tables:
``redirect`` (the redirect page's id -> the target's title) and ``page`` (that id -> the redirect page's own title).
Both are streamed from dumps.wikimedia.org as gzipped SQL, parsed tuple by tuple, and only article-space pairs are
kept (namespace 0 on both sides, no interwiki target): ``from``, ``to``, ``fragment``, titles with spaces. No raw
file is stored. The pairs name items through the enwiki / frwiki titles of ``names_extract`` (the join is the
segment build's).

    python validation/retrieval_sets/wiki_redirects.py plan --out <folder>           # the dumps' sizes and dates
    python validation/retrieval_sets/wiki_redirects.py pairs --out <folder> [--wiki enwiki frwiki]

Wired: STANDALONE (a data builder; ``notebooks/r1b_names.ipynb`` runs it on Colab).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import urllib.request

__wiring__ = "STANDALONE"

WIKIS = ("enwiki", "frwiki")
BASE = "https://dumps.wikimedia.org/{w}/latest/{w}-latest-{t}.sql.gz"
UA = "CubbyLLM-retrieval-bench/0.1 (order 0046 R1b; research measurement)"
_S = rb"'((?:[^'\\]|\\.)*)'"
# page: (page_id, page_namespace, page_title, page_is_redirect, ...)
PAGE = re.compile(rb"\((\d+),(-?\d+)," + _S + rb",([01]),")
# redirect: (rd_from, rd_namespace, rd_title, rd_interwiki, rd_fragment)
REDIRECT = re.compile(rb"\((\d+),(-?\d+)," + _S + rb",(NULL|" + _S + rb"),(NULL|" + _S + rb")\)")
_ESC = re.compile(rb"\\(.)")
_MAP = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"0": b"\0", b"Z": b"\x1a"}


def unescape(b: bytes) -> str:
    """A MySQL string literal's body as text, a title's underscores as spaces."""
    return _ESC.sub(lambda m: _MAP.get(m.group(1), m.group(1)), b).decode("utf-8", "replace").replace("_", " ")


def redirect_pages(lines) -> dict[int, str]:
    """{page id: title} of the article-space pages that are redirects."""
    out = {}
    for ln in lines:
        if not ln.startswith(b"INSERT INTO"):
            continue
        for m in PAGE.finditer(ln):
            if m.group(2) == b"0" and m.group(4) == b"1":
                out[int(m.group(1))] = unescape(m.group(3))
    return out


def pairs(lines, pages: dict[int, str]):
    """(from, to, fragment) for each article-space redirect whose page is known and whose target is local."""
    for ln in lines:
        if not ln.startswith(b"INSERT INTO"):
            continue
        for m in REDIRECT.finditer(ln):
            src = pages.get(int(m.group(1)))
            iw = m.group(5)
            if src is None or m.group(2) != b"0" or (iw is not None and iw != b""):
                continue
            frag = m.group(7)
            yield src, unescape(m.group(3)), unescape(frag) if frag else ""


def _lines(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, gzip.GzipFile(fileobj=r) as g:
        yield from g


def head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return {"url": url, "bytes": int(r.headers.get("Content-Length") or 0), "modified": r.headers.get("Last-Modified")}


def build(wiki: str, out: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst = os.path.join(out, f"redirects_{wiki}.parquet")
    meta = {t: head(BASE.format(w=wiki, t=t)) for t in ("page", "redirect")}
    if os.path.exists(dst):
        return {"wiki": wiki, "out": dst, "skipped": True, **meta}
    t0 = time.time()
    pages = redirect_pages(_lines(meta["page"]["url"]))
    print(f"{wiki}: {len(pages):,} redirect pages in {time.time() - t0:.0f} s", flush=True)
    rows = list(pairs(_lines(meta["redirect"]["url"]), pages))
    t = pa.table({"from": [r[0] for r in rows], "to": [r[1] for r in rows], "fragment": [r[2] for r in rows]})
    pq.write_table(t, dst + ".part", compression="zstd")
    os.replace(dst + ".part", dst)
    print(f"{wiki}: {len(rows):,} redirects -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB, {time.time() - t0:.0f} s)", flush=True)
    return {"wiki": wiki, "out": dst, "pairs": len(rows), "redirect_pages": len(pages), "bytes": os.path.getsize(dst), **meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "pairs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", nargs="*", default=list(WIKIS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "plan":
        p = {w: {t: head(BASE.format(w=w, t=t)) for t in ("page", "redirect")} for w in a.wiki}
        for w, d in p.items():
            for t, h in d.items():
                print(f"{w} {t}: {h['bytes'] / 1e9:.2f} GB streamed (not kept), dump of {h['modified']}")
        json.dump(p, open(os.path.join(a.out, "redirects_plan.json"), "w", encoding="utf-8"), indent=1)
    else:
        man = [build(w, a.out) for w in a.wiki]
        json.dump(man, open(os.path.join(a.out, "redirects_manifest.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
