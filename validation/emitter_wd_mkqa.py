"""emitter_wd_mkqa -- order 0048's French records from questions people wrote in French, whose item is not marked:
  mkqa   MKQA (CC BY-SA 3.0; its repository's README): human-translated Natural Questions, answers linked to Wikidata
  piaf   PIAF v1.2 (MIT), the rows of CATIE-AQ/frenchQA titled "piaf": questions on French Wikipedia, answers quoted
         from the passage; an answer is an item when its words are a key of the names segment (at most ``MAX_ITEMS``)
The item is found by rule, never by a model:

  spans     every run of 1 to 6 words of the French question that is a key of the VM's names segment (R1b: Wikidata's
            labels, aliases, titles and redirects as CubeLang knowledge keys; exact lookups, ``knowledge batch``); a
            run of function words alone (R2's ``STOP``), or a name of 2 letters or fewer, is not a span; a span naming
            more than ``MAX_ITEMS`` items is not one either (too common a name to tell anything)
  link      one SPARQL query a question: which of the spans' items states an answer item as a value, and by which
            property (item -> answer, direct properties); answered once, cached off-repo, resumable
  written   only when exactly one property links them and every linking item sits under one span; the span is the
            item's slot, the property's French label (the frozen lexicon) is HOP1. Otherwise counted, not written.

    python validation/emitter_wd_mkqa.py [--source mkqa|piaf]
-> $CUBBY_DATA_ROOT/retrieval/<source>_fr_records.jsonl (``emitter_wd_data`` adds them, audited with the rest)

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

MAX_WORDS, MAX_ITEMS = 6, 50
WORD = re.compile(r"\w+(?:['’]\w+)*", re.U)
DIRECT = "http://www.wikidata.org/prop/direct/"


def spans_of(question: str) -> list[tuple[int, int, str]]:
    """Every run of 1 to MAX_WORDS words: (start, end, text), function-word runs and short names left out."""
    from retrieval_bench.r2_arms import STOP
    ws = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(question)]
    out = []
    for i in range(len(ws)):
        for j in range(i, min(i + MAX_WORDS, len(ws))):
            words = [w[2].lower() for w in ws[i:j + 1]]
            text = question[ws[i][0]:ws[j][1]]
            if all(w in STOP for w in words) or len(text) <= 2:
                continue
            out.append((ws[i][0], ws[j][1], text))
    return out


def lookup(keys: list[str]) -> dict[str, list[str]]:
    """{key: [items]} through the names segment, exact (one ``knowledge batch`` over every key)."""
    from retrieval_sets.common import data_root
    from retrieval_sets.names_segment import cubelang_exe
    qp = os.path.join(tempfile.mkdtemp(), "keys.jsonl")
    with open(qp, "w", encoding="utf-8") as f:
        for i, k in enumerate(keys):
            f.write(json.dumps({"id": str(i), "key": k}, ensure_ascii=False) + "\n")
    r = subprocess.run([cubelang_exe(), "knowledge", "batch", os.path.join(data_root(), "r1b_segment"), qp],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(r.stderr.strip())
    out = {}
    for line in r.stdout.splitlines():
        d = json.loads(line)
        out[keys[int(d["id"])]] = sorted({f["text"] for f in d["exact"]})
    return out


def links(items: list[str], answers: list[str]) -> list[tuple[str, str]] | None:
    """[(item, property)] where the item states an answer as the property's value; None when the service cannot say."""
    import urllib.parse
    import urllib.request
    from cubbyllm.sources.wikidata import UA
    q = ("SELECT ?s ?p WHERE { VALUES ?s { " + " ".join(f"wd:{x}" for x in items) + " } VALUES ?o { "
         + " ".join(f"wd:{a}" for a in answers) + " } ?s ?p ?o . FILTER(STRSTARTS(STR(?p), \"" + DIRECT + "\")) }")
    for k in range(3):
        try:
            req = urllib.request.Request("https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": q}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.2)
            return sorted({(b["s"]["value"].rsplit("/", 1)[1], b["p"]["value"][len(DIRECT):]) for b in rows})
        except Exception:                                  # noqa: BLE001 -- retried, then recorded as unknown
            time.sleep(5 * (k + 1))
    return None


def decide(spans: list[tuple[int, int, str]], items_of: dict, found: list[tuple[str, str]]) -> tuple[str, dict | None]:
    """("written", {span, item, pid}) or (why, None), by the rule above."""
    props = {p for _s, p in found}
    if not props:
        return "no link", None
    if len(props) > 1:
        return "several properties", None
    linking = {s for s, _p in found}
    under = [sp for sp in spans if linking <= set(items_of.get(sp[2], ()))]
    if not under:
        return "linking items under several spans", None
    best = max(under, key=lambda sp: sp[1] - sp[0])         # the longest span holding every linking item
    return "written", {"span": [best[0], best[1]], "item": sorted(linking)[0], "pid": props.pop()}


def mkqa_rows() -> list[dict]:
    from retrieval_sets.common import raw_dir
    out = []
    for line in gzip.open(os.path.join(raw_dir("mkqa"), "mkqa.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(line)
        q = (r.get("queries") or {}).get("fr")
        ans = sorted({a["entity"] for a in (r.get("answers") or {}).get("fr") or [] if (a.get("entity") or "").startswith("Q")})
        if q and ans:
            out.append({"id": r["example_id"], "question": q, "answers": ans})
    return out


def piaf_rows() -> list[dict]:
    """PIAF's answered questions, one per distinct question; their answers as text, made items by ``main``."""
    import pyarrow.parquet as pq
    from retrieval_sets.common import raw_dir
    seen, out = set(), []
    for r in pq.read_table(os.path.join(raw_dir("frenchqa"), "frenchqa_train.parquet")).to_pylist():
        if r["title"] == "piaf" and r["answers"]["text"] and r["question"] not in seen:
            seen.add(r["question"])
            out.append({"id": f"piaf-{r['id']}", "question": r["question"], "answer_texts": sorted(set(r["answers"]["text"]))})
    return out


def main() -> None:
    import argparse
    from retrieval_sets.common import data_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("mkqa", "piaf"), default="mkqa")
    a = ap.parse_args()
    todo = mkqa_rows() if a.source == "mkqa" else piaf_rows()
    for t in todo:
        t["spans"] = spans_of(t["question"])
    keys = sorted({sp[2] for t in todo for sp in t["spans"]} | {x for t in todo for x in t.get("answer_texts", ())})
    print(f"{len(todo)} French {a.source} questions; {len(keys)} distinct spans looked up", flush=True)
    found = lookup(keys)
    items_of = {k: v for k, v in found.items() if 0 < len(v) <= MAX_ITEMS}
    for t in todo:                                       # PIAF: an answer is the items its words name, if few enough
        if "answer_texts" in t:
            t["answers"] = sorted({x for a_ in t["answer_texts"] for x in items_of.get(a_, ())})
    todo = [t for t in todo if t["answers"]]
    cache_p = os.path.join(data_root(), f"{a.source}_fr_links.jsonl")
    cache = {}
    if os.path.exists(cache_p):
        for line in open(cache_p, encoding="utf-8"):
            d = json.loads(line)
            cache[d["id"]] = d["links"]
    c, out = collections.Counter(), []
    with open(cache_p, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo):
            spans = [sp for sp in t["spans"] if sp[2] in items_of]
            items = sorted({x for sp in spans for x in items_of[sp[2]]})
            if not items:
                c["no span names an item"] += 1
                continue
            if t["id"] not in cache:
                got = links(items, t["answers"])
                cache[t["id"]] = None if got is None else [list(x) for x in got]
                f.write(json.dumps({"id": t["id"], "links": cache[t["id"]]}) + "\n")
                f.flush()
            if cache[t["id"]] is None:
                c["link query unanswered"] += 1
                continue
            why, rec = decide(spans, items_of, [tuple(x) for x in cache[t["id"]]])
            c[why] += 1
            if rec:
                out.append({"id": f"wd-{a.source}-{t['id']}-fr", "question": t["question"], **rec})
            if i % 250 == 0:
                print(f"{i}/{len(todo)} questions; {dict(c)}", flush=True)
    path = os.path.join(data_root(), f"{a.source}_fr_records.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump({"counts": dict(c), "written": len(out)}, open(path + ".counts.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(c), indent=1), f"\n{len(out)} French candidates -> {path}")


if __name__ == "__main__":
    main()
