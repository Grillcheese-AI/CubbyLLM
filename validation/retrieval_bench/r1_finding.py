"""r1_finding -- order 0046, R1: does a local name index find the item a mention means, against Wikidata's search API?

Arms (``docs/RETRIEVAL.md``, "Find"):
  api        wbsearchentities in the row's language, its own order, up to 100 hits (today's candidate source);
  exact      the local index, lowercase equal;                         a set, unordered
  normalized + folded equal (accents and punctuation off);             a set, unordered
  trigram    + names sharing letter trigrams, ordered by the capsules' code distance
  bm25       + BM25 over the names' words, only filling what the others left short (never displacing)
recall@K for K in {5, 20, 100}. **Only a score separates candidates:** a set (or a run of equal scores) enters the
shortlist whole or not at all, so a name shared by 300 items fits no shortlist of 20 -- no tie is broken by fame or
order. Mentions are the sets' own where they mark them, else the gold item's own names located in the question
(``mentions``); an item the index has no name for is scored apart, and a mention no name of its item spells is counted
apart. The typo set: 20% of the mentions (seed 0) with one or two letters changed.

Gate (pre-registered in the order): the best local arm's recall@20 >= 0.90 on Mintaka EN and >= 0.85 on Mintaka FR,
and above the API's on every set; the trigram arm kept only if it adds >= 5 points on the typo set. Kill: no local arm
beats the API at @20 on Mintaka.

    python validation/retrieval_bench/r1_finding.py --split dev [--n 500] [--n-mintaka 1000] [--workers 6] [--api-threads 3]
-> validation/logs/retrieval_R1_<split>_<date>.json (ids and counts only)

Wired: STANDALONE (an experiment).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_sets.common import SEED, get, read_rows, sample, source  # noqa: E402

__wiring__ = "STANDALONE"

SETS = ("mintaka", "qald9plus", "webqsp_el", "graphq_el", "simplequestions_wd", "shadowlink", "amber", "cronquestions")
KS = (5, 20, 100)
LOCAL = ("exact", "normalized", "trigram", "bm25")
TYPO_SHARE = 0.2
_IX = None


def typo(s: str, rng: random.Random) -> str:
    """One or two letters changed: a substitution, a deletion or a swap of neighbours, on letters only."""
    chars = list(s)
    for _ in range(rng.choice((1, 2))):
        pos = [i for i, c in enumerate(chars) if c.isalpha()]
        if len(pos) < 3:
            break
        i = rng.choice(pos)
        op = rng.choice(("sub", "del", "swap"))
        if op == "sub":
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz".replace(chars[i].lower(), ""))
        elif op == "del":
            del chars[i]
        elif i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def position(groups: list, gold: str) -> int | None:
    """The shortlist's size once the group holding `gold` is in (every group whole), or None."""
    seen, cum = set(), 0
    for g in groups:
        new = [q for q in g if q not in seen]
        seen.update(new)
        cum += len(new)
        if gold in new:
            return cum
    return None


def by_score(pairs) -> list[list[str]]:
    """[(score, qid)] best first -> groups of equal score, in order."""
    out, last = [], object()
    for s, q in pairs:
        if s != last:
            out.append([])
            last = s
        out[-1].append(q)
    return out


def _init() -> None:
    global _IX
    from retrieval_bench.name_index import NameIndex
    _IX = NameIndex()


def local_one(job: tuple) -> dict:
    """The local arms for one mention: the gold's position in each arm's shortlist."""
    key, text, gold = job
    ix = _IX
    ex = sorted(ix.exact(text))
    no = sorted(ix.normalized(text) - set(ex))
    near = by_score(ix.near(text, 100))
    bm = by_score(ix.bm25(text, 100))
    arms = {"exact": [ex], "normalized": [ex, no], "trigram": [ex, no] + near, "bm25": [ex, no] + near + bm}
    return {"key": key, "pos": {a: position(g, gold) for a, g in arms.items()}, "n_exact": len(ex), "n_norm": len(ex) + len(no),
            "in_index": ix.has(gold)}


def api_one(src, text: str, lang: str, gold: str) -> dict:
    """Wikidata's search in the row's language: the gold's rank (1-based) among up to 100 hits, or None."""
    hits, offset = [], 0
    while len(hits) < 100:
        d = get(src, {"action": "wbsearchentities", "search": text, "language": lang, "uselang": lang, "type": "item",
                      "limit": 50, "continue": offset}) or {}
        page = [h.get("id") for h in d.get("search") or []]
        hits += page
        if gold in hits or "search-continue" not in d or not page:
            break
        offset = d["search-continue"]
    return {"rank": hits.index(gold) + 1 if gold in hits else None, "n_hits": len(hits)}


def mentions_of(split: str, n: int, n_mintaka: int, sets) -> tuple[list[dict], dict]:
    from retrieval_bench.mentions import Gold, fill
    src, out, located = source(), [], {}
    gold = Gold(src)
    for name in sets:
        for lang in ("en", "fr"):
            rows = read_rows(name, split=split, lang=lang)
            if not rows:
                continue
            rows = sample(rows, n_mintaka if name == "mintaka" else n)
            located[f"{name}/{lang}"] = fill(rows, gold)
            for r in rows:
                for k, m in enumerate(r["mentions"]):
                    if m.get("qid") and m.get("text"):
                        out.append({"key": f"{r['id']}#{k}", "set": name, "lang": lang, "text": m["text"], "gold": m["qid"],
                                    "extra": {kk: r["extra"].get(kk) for kk in ("subset", "head") if kk in r["extra"]}})
    rng = random.Random(SEED)
    for m in out:
        if rng.random() < TYPO_SHARE:
            t = typo(m["text"], rng)
            if t != m["text"]:
                m["typo"] = t
    return out, located


def summarize(ms: list[dict]) -> dict:
    def rate(sub, arm, k, typo_=False):
        if not sub:
            return None
        f = (lambda m: m.get("typo_api_rank")) if (arm == "api" and typo_) else (lambda m: m["api"]["rank"]) if arm == "api" \
            else (lambda m: (m.get("typo_pos") or {}).get(arm)) if typo_ else (lambda m: m["pos"][arm])
        return round(sum(1 for m in sub if (f(m) or 10 ** 9) <= k) / len(sub), 4)
    out = {}
    groups = sorted({(m["set"], m["lang"]) for m in ms})
    for s, lang in groups:
        sub = [m for m in ms if m["set"] == s and m["lang"] == lang]
        ins = [m for m in sub if m["in_index"]]
        ty = [m for m in sub if "typo" in m]
        row = {"mentions": len(sub), "in_index": len(ins), "typo": len(ty)}
        for arm in ("api",) + LOCAL:
            row[arm] = {f"@{k}": rate(sub, arm, k) for k in KS}
            row[arm + "_in_index"] = {f"@{k}": rate(ins, arm, k) for k in KS}
            row[arm + "_typo"] = {f"@{k}": rate(ty, arm, k, True) for k in KS}
        row["exact_set_size"] = {"p50": sorted(m["n_exact"] for m in sub)[len(sub) // 2],
                                 "over_20": sum(1 for m in sub if m["n_norm"] > 20)}
        out[f"{s}/{lang}"] = row
    return out


def gates(summary: dict) -> dict:
    best = lambda row: max(row[a]["@20"] or 0 for a in LOCAL)  # noqa: E731
    g = {}
    for key, line in (("mintaka/en", 0.90), ("mintaka/fr", 0.85)):
        if key in summary:
            g[f"best local recall@20 >= {line} on {key}"] = best(summary[key]) >= line
    g["the best local arm above the API at @20 on every set"] = all(best(r) > (r["api"]["@20"] or 0) for r in summary.values())
    adds = [((r["trigram_typo"]["@20"] or 0) - (r["normalized_typo"]["@20"] or 0)) for r in summary.values() if r["typo"]]
    g["trigram adds >= 5 points on the typo set (mean over sets)"] = bool(adds) and sum(adds) / len(adds) >= 0.05
    g["KILL: no local arm beats the API at @20 on Mintaka"] = all(best(summary[k]) <= (summary[k]["api"]["@20"] or 0)
                                                               for k in ("mintaka/en", "mintaka/fr") if k in summary)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-mintaka", type=int, default=1000)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--api-threads", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    ms, located = mentions_of(a.split, a.n, a.n_mintaka, a.sets)
    print(f"{len(ms)} mentions ({sum(1 for m in ms if 'typo' in m)} with a typo variant); located: {located}", flush=True)
    jobs = [(m["key"], m["text"], m["gold"]) for m in ms] + [(m["key"] + "~typo", m["typo"], m["gold"]) for m in ms if "typo" in m]
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        loc = {r["key"]: r for r in pool.map(local_one, jobs, chunksize=16)}
    print(f"local arms done ({time.time() - t0:.0f}s)", flush=True)
    srcs = [source() for _ in range(a.api_threads)]
    api_jobs = [(m["key"], m["text"], m["lang"], m["gold"]) for m in ms] + [(m["key"] + "~typo", m["typo"], m["lang"], m["gold"]) for m in ms if "typo" in m]
    with ThreadPoolExecutor(a.api_threads) as tp:
        api = dict(zip([j[0] for j in api_jobs], tp.map(lambda ij: api_one(srcs[ij[0] % len(srcs)], *ij[1][1:]), enumerate(api_jobs))))
    print(f"api arm done ({time.time() - t0:.0f}s)", flush=True)
    for m in ms:
        r = loc[m["key"]]
        m.update(pos=r["pos"], n_exact=r["n_exact"], n_norm=r["n_norm"], in_index=r["in_index"], api=api[m["key"]])
        if "typo" in m:
            m["typo_pos"] = loc[m["key"] + "~typo"]["pos"]
            m["typo_api_rank"] = api[m["key"] + "~typo"]["rank"]
    summary = summarize(ms)
    rep = {"what": "order 0046 R1: finding", "split": a.split, "date": _dt.date.today().isoformat(), "n": a.n, "n_mintaka": a.n_mintaka,
           "located": located, "summary": summary, "gates": gates(summary), "wall_s": round(time.time() - t0),
           "mentions": [{"key": m["key"], "gold": m["gold"], "pos": m["pos"], "api_rank": m["api"]["rank"], "in_index": m["in_index"],
                         "n_exact": m["n_exact"], "n_norm": m["n_norm"], "typo_pos": m.get("typo_pos"), "typo_api_rank": m.get("typo_api_rank"),
                         **m["extra"]} for m in ms]}
    out = os.path.join(os.path.dirname(HERE), "logs", f"retrieval_R1_{a.split}_{rep['date']}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": {k: {a_: v[a_]["@20"] for a_ in ("api",) + LOCAL} for k, v in summary.items()}, "gates": rep["gates"]}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
