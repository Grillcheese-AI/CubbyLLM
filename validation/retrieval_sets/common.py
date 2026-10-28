"""common -- what every set builder and every experiment of order 0046 shares: where the data lives (off-repo), the one
row schema, the fixed splits and samples (seed 0), name folding, and the Wikidata lookups by id (labels, aliases,
sitelinks; titles to ids), all through the source's own cache.

A row:
    {"id": "<set>:<native id>", "set": "<set>", "lang": "en" | "fr", "question": str,
     "mentions": [{"qid": "Q..", "text": str | None, "span": [s, e] | None}],
     "prop": "P.." | None, "answers": [str], "answer_qids": ["Q.."], "split": "dev" | "test", "extra": {...}}

Splits: a set's own dev (or valid) and test where it has them; a set with a train and a test only uses its train as
dev (nothing is trained here); a set with neither is cut in half by seed 0. Tune on dev only; read test once, at the end.

Wired: STANDALONE (validation helpers).
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata

__wiring__ = "STANDALONE"

SEED = 0
LANGS = ("en", "fr")


def data_root() -> str:
    """``$CUBBY_DATA_ROOT/retrieval`` (the environment, then ``paths.local.json``), else ``$CUBBY_DATASETS/retrieval``."""
    from cubbyllm.local_paths import value
    base = value("CUBBY_DATA_ROOT") or value("CUBBY_DATASETS")
    if not base:
        raise SystemExit("set CUBBY_DATA_ROOT (or CUBBY_DATASETS): the sets stay off-repo")
    out = os.path.join(base, "retrieval")
    os.makedirs(out, exist_ok=True)
    return out


def raw_dir(name: str) -> str:
    return os.path.join(data_root(), "raw", name)


def sets_dir() -> str:
    d = os.path.join(data_root(), "sets")
    os.makedirs(d, exist_ok=True)
    return d


def write_rows(name: str, rows: list[dict]) -> str:
    path = os.path.join(sets_dir(), f"{name}.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    return path


def read_rows(name: str, split: str | None = None, lang: str | None = None) -> list[dict]:
    path = os.path.join(sets_dir(), f"{name}.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (split is None or r["split"] == split) and (lang is None or r["lang"] == lang):
                out.append(r)
    return out


def halves(ids: list[str], seed: int = SEED) -> dict[str, str]:
    """{id: "dev" | "test"}: a fixed half each, for a set that has no split of its own."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    cut = len(order) // 2
    return {i: ("dev" if k < cut else "test") for k, i in enumerate(order)}


def sample(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """At most `n` rows, a fixed draw (sorted by id first, so a rebuilt set draws the same rows)."""
    rows = sorted(rows, key=lambda r: r["id"])
    if n <= 0 or len(rows) <= n:
        return rows
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r["id"])


_PUNCT = re.compile(r"[^\w\s]", re.U)


def fold(s: str) -> str:
    """A name as compared: lowercase, accents off, punctuation off, spaces collapsed."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(_PUNCT.sub(" ", s).split())


def source():
    """The Wikidata source, live, its calls cached (the cache is the source's own); no askers' ledger."""
    from cubbyllm.sources.wikidata import WikidataSource
    from cubby_serve.sources.http import wikidata as fetch
    return WikidataSource(offline=False, use_choices=False, fetch=fetch(), sleep_s=0.2)


def get(src, params: dict, tries: int = 5):
    """The source's cached GET, retried with backoff: Wikimedia answers 503 now and then under load, and the source
    caches only a success (a None is never stored)."""
    import time
    for k in range(tries):
        d = src._get(params)
        if d is not None:
            return d
        time.sleep(2 * 2 ** k)
    return None


def entities(src, qids, props: str = "labels|aliases|sitelinks", langs: str = "en|fr") -> dict:
    """{qid: entity json} for `qids`, 50 to a call, through the source's cache."""
    out, ids = {}, [q for q in dict.fromkeys(qids) if q and re.fullmatch(r"[QP]\d+", q)]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        d = get(src, {"action": "wbgetentities", "ids": "|".join(chunk), "props": props, "languages": langs,
                      "sitefilter": "enwiki|frwiki"}) or {}
        for q, e in (d.get("entities") or {}).items():
            if "missing" not in e:
                out[q] = e
    return out


def names_of(entity: dict, lang: str) -> list[str]:
    """An entity's label and aliases in `lang` (and its `mul` label, Wikidata's name shared by every language)."""
    out = []
    for lg in (lang, "mul"):
        lab = ((entity.get("labels") or {}).get(lg) or {}).get("value")
        if lab:
            out.append(lab)
        out += [a["value"] for a in (entity.get("aliases") or {}).get(lg, [])]
    return list(dict.fromkeys(out))


def titles_to_qids(src, titles, site: str = "enwiki") -> dict:
    """{title: qid} for Wikipedia titles, 50 to a call, redirects followed, through the source's cache."""
    out, ts = {}, [t for t in dict.fromkeys(titles) if t]
    for i in range(0, len(ts), 50):
        chunk = ts[i:i + 50]
        d = get(src, {"action": "wbgetentities", "sites": site, "titles": "|".join(chunk), "props": "sitelinks",
                      "sitefilter": site, "redirects": "yes"}) or {}
        for q, e in (d.get("entities") or {}).items():
            t = ((e.get("sitelinks") or {}).get(site) or {}).get("title")
            if t and q.startswith("Q"):
                out[t] = q
        norm = {n["to"]: n["from"] for n in ((d.get("normalized") or []) + (d.get("redirects") or []))}
        for to, frm in norm.items():
            if to in out:
                out.setdefault(frm, out[to])
    return out


def wikipedia_qids(titles, lang: str = "en") -> dict:
    """{title: qid} through Wikipedia's own API (redirects followed, ``pageprops.wikibase_item``), 50 titles a call,
    each answer cached off-repo (``wiki_cache/``). Wikidata's title lookup answers 503 for a title that is a redirect;
    this one follows it."""
    import hashlib
    import time
    import urllib.parse
    from cubby_serve.sources.http import wikidata as fetch
    get_url, cache = fetch(), os.path.join(data_root(), "wiki_cache")
    os.makedirs(cache, exist_ok=True)
    out, ts = {}, [t for t in dict.fromkeys(titles) if t]
    for i in range(0, len(ts), 50):
        chunk = ts[i:i + 50]
        params = {"action": "query", "titles": "|".join(chunk), "redirects": "1", "prop": "pageprops",
                  "ppprop": "wikibase_item", "format": "json"}
        key = hashlib.sha256(urllib.parse.urlencode(sorted(params.items())).encode()).hexdigest()[:24]
        path = os.path.join(cache, f"{lang}_{key}.json")
        d = None
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
        else:
            for k in range(5):
                try:
                    d = json.loads(get_url(f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))
                    json.dump(d, open(path, "w", encoding="utf-8"))
                    break
                except Exception:                        # noqa: BLE001 -- a 503 under load: back off and retry
                    time.sleep(2 * 2 ** k)
            time.sleep(0.2)
        q = d or {}
        pages = ((q.get("query") or {}).get("pages") or {}).values()
        by_title = {p.get("title"): (p.get("pageprops") or {}).get("wikibase_item") for p in pages}
        hops = {}
        for n in ((q.get("query") or {}).get("normalized") or []) + ((q.get("query") or {}).get("redirects") or []):
            hops[n["from"]] = n["to"]
        for t in chunk:
            cur, seen = t, set()
            while cur in hops and cur not in seen:
                seen.add(cur)
                cur = hops[cur]
            if by_title.get(cur):
                out[t] = by_title[cur]
    return out


__all__ = ["LANGS", "SEED", "data_root", "entities", "fold", "get", "halves", "names_of", "raw_dir", "read_rows", "sample",
           "sets_dir", "source", "titles_to_qids", "wikipedia_qids", "write_rows"]
