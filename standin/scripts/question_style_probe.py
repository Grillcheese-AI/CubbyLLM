"""[stand-in] question-style probe: what real and disfluent questions do to the fact grammar and the router.

Wired: STANDALONE (a measurement script; nothing imports it).

Two public slices, fetched through the Hub's datasets-server rows API (no pandas/pyarrow needed; the NQ
parquet is 1.3 GB of Wikipedia HTML, the rows API hands back the question and the short answers):

  google-research-datasets/disfl_qa          (CC-BY-4.0)  SQuAD-v2 dev questions, each with a contextual
                                              disfluency (a restart with a distractor from the passage):
                                              "What is the second level of ... no make that the basic unit ..."
  google-research-datasets/natural_questions (CC-BY-SA-3.0) real user search queries, lowercase, no '?':
                                              "who is the secretary of state for northern ireland"

Measured per question: does the fact grammar parse it (`cubbyllm.reasoning.parse_question`), does it parse the
SAME as its clean twin (disfl_qa), what does the thalamus read it as (`CubbyBrain.needs_facts`), does the
question detector fire. Owner's ask (2026-09-03): "could we use disfl_qa, with a slice of natural_questions?"

  python standin/scripts/question_style_probe.py [--n-disfl 1000] [--n-nq 300] [--out standin/data/out]
  tee: validation/logs/standin_question_style_probe.log
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length={n}"


def fetch_rows(ds: str, cfg: str, split: str, n: int, page: int, cache: str) -> list[dict]:
    """Rows through the datasets-server API, cached as JSON per page."""
    out = []
    for off in range(0, n, page):
        f = os.path.join(cache, f"{ds.split('/')[-1]}_{cfg}_{split}_{off}.json")
        if not os.path.exists(f):
            url = API.format(ds=ds.replace("/", "%2F"), cfg=cfg, split=split, off=off, n=min(page, n - off))
            with urllib.request.urlopen(url, timeout=300) as r:
                open(f, "wb").write(r.read())
        j = json.load(open(f, encoding="utf-8"))
        out += [r["row"] for r in j.get("rows", [])]
    return out[:n]


def nq_question(row: dict) -> dict:
    q = row["question"]["text"] if isinstance(row.get("question"), dict) else row.get("question")
    ann = row.get("annotations") or {}
    shorts: list[str] = []
    for sa in ann.get("short_answers", []) or []:
        shorts += list(sa.get("text", []) or [])
    return {"q": q, "short": shorts, "title": (row.get("document") or {}).get("title")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-disfl", type=int, default=1000)
    ap.add_argument("--n-nq", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "standin", "data", "out"))
    args = ap.parse_args()
    cache = os.path.join(args.out, "question_style_cache")
    os.makedirs(cache, exist_ok=True)

    from cubbyllm.reasoning import parse_question
    import serve as sv

    class Dummy:                                         # the router and the grammar never touch the model
        name = "dummy"

        def emit(self, *a, **k):
            raise RuntimeError("no model in this probe")

    brain = sv.CubbyBrain(Dummy(), sv.FactStore([]), exe=None, route_tau=0.30)

    def key(p):
        return None if p is None else (tuple(p.relations), getattr(p, "subject", None))

    # ── disfl_qa: clean vs disfluent twins ────────────────────────────────────
    rows = fetch_rows("google-research-datasets/disfl_qa", "default", "validation", args.n_disfl, 100, cache)
    n = len(rows)
    po = pd_ = same = nf = qd = 0
    cues: collections.Counter = collections.Counter()
    for r in rows:
        o, d = r["original question"], r["disfluent question"]
        a, b = parse_question(o), parse_question(d)
        po += a is not None
        pd_ += b is not None
        same += a is not None and key(a) == key(b)
        nf += brain.needs_facts(o)[0] == brain.needs_facts(d)[0]
        qd += bool(brain._QUESTION.search(o)) == bool(brain._QUESTION.search(d))
        for c in re.findall(r"\b(no wait|no|wait|sorry|scratch that|i mean|actually|rather|um+|uh+|er+|make that|correction)\b", d.lower()):
            cues[c] += 1
    print(f"[stand-in] disfl_qa validation, n={n}")
    print(f"  fact grammar parses: clean {po}/{n} ({po/n:.1%}) | disfluent {pd_}/{n} ({pd_/n:.1%}) | "
          f"the disfluent twin parses the SAME as its clean question: {same}/{po} of the parsed clean ones ({same/max(1,po):.1%})")
    print(f"  router agrees clean vs disfluent: needs_facts {nf}/{n} ({nf/n:.1%}) | question detector {qd}/{n}")
    print(f"  words: clean {statistics.mean(len(r['original question'].split()) for r in rows):.1f} | "
          f"disfluent {statistics.mean(len(r['disfluent question'].split()) for r in rows):.1f} | repair cues {cues.most_common(8)}")

    # ── natural_questions: real search queries ────────────────────────────────
    nq = [nq_question(r) for r in fetch_rows("google-research-datasets/natural_questions", "dev", "validation", args.n_nq, 50, cache)]
    m = len(nq)
    parsed = sum(parse_question(r["q"]) is not None for r in nq)
    reasons = collections.Counter(brain.needs_facts(r["q"])[1] for r in nq)
    qdet = sum(bool(brain._QUESTION.search(r["q"])) for r in nq)
    print(f"[stand-in] natural_questions dev, n={m}")
    print(f"  fact grammar parses {parsed}/{m} ({parsed/m:.1%}) | question detector fires {qdet}/{m} | has a short answer "
          f"{sum(bool(r['short']) for r in nq)}/{m} | lowercase start {sum(r['q'][:1].islower() for r in nq)}/{m} | "
          f"ends with '?' {sum(r['q'].strip().endswith('?') for r in nq)}/{m} | words {statistics.mean(len(r['q'].split()) for r in nq):.1f}")
    print(f"  thalamus reads: {dict(reasons)}")
    for why in ("a feeling, opinion or creative ask", "about Cubby himself"):
        bad = [r["q"] for r in nq if brain.needs_facts(r["q"])[1] == why]
        if bad:
            print(f"  read as '{why}' ({len(bad)}): " + " | ".join(repr(q) for q in bad[:6]))
    json.dump({"disfl_n": n, "disfl_parse_clean": po, "disfl_parse_disfluent": pd_, "disfl_same_parse": same,
               "disfl_needs_facts_agree": nf, "nq_n": m, "nq_parse": parsed, "nq_reasons": dict(reasons),
               "nq_questions": nq}, open(os.path.join(args.out, "question_style_probe.json"), "w", encoding="utf-8"), indent=1)
    print(f"wrote {os.path.join(args.out, 'question_style_probe.json')}")


if __name__ == "__main__":
    main()
