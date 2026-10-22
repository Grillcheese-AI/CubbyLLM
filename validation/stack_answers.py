"""stack_answers -- the real stack's answers to 20 fixed questions and its board, recorded so a refactor can be checked
byte for byte (order 0031: "the real stack gives the same answers").

Against a running server (``serve_api.py ... --board --ask wikidata-offline --web off --history '' --learned ''``):
each question through ``POST /ask`` in the public audience, the record's journey, and every ``/board`` route in both
audiences (the audit token from ``CUBBY_AUDIT_TOKEN``, never written down). ``--compare`` checks two recordings equal
but for times and ids (the keys in ``VOLATILE``).

    python validation/stack_answers.py --base http://127.0.0.1:8792 --out validation/logs/0031_stack_before_<date>.json
    python validation/stack_answers.py --compare <before.json> <after.json>

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

__wiring__ = "STANDALONE"

QUESTIONS = [
    "What is the capital of France?", "Where was Ada Lovelace born?", "Who wrote Pride and Prejudice?",
    "When was the Eiffel Tower built?", "What is the capital of Canada?", "Who is the father of Queen Elizabeth II?",
    "What is the population of Toronto?", "Who painted the Mona Lisa?", "Where is the Globe Theatre?",
    "What language is spoken in Brazil?", "Who discovered penicillin?", "What is the currency of Japan?",
    "Quelle est la capitale de l'Italie ?", "Qui a écrit Les Misérables ?", "What river flows through Cairo?",
    "Who was the first president of the United States?", "How tall is Mount Everest?", "What is the speed of light?",
    "Who is the mayor of Montreal?", "What year did World War II end?",
]
BOARD = ["/board/home", "/board/system", "/board/mood", "/board/plugins", "/board/watch", "/board/worlds",
         "/board/forecast", "/board/agents", "/board/plugins/route?q=how+many+kilometres+is+5+miles",
         "/board/plugins/route?q=hi+Cubby%2C+how+are+you%3F", "/board/plugins/review?name=units", "/health", "/worlds"]
VOLATILE = {"id", "ts", "wall_s", "ms", "t", "updated", "age_min", "as_of", "last_poll_ago", "items_today", "up", "next",
            "at", "read_at", "used", "turn_id", "time", "elapsed", "started"}


def call(base, path, data=None, audit=False):
    h = {"Content-Type": "application/json"}
    if audit and os.environ.get("CUBBY_AUDIT_TOKEN"):
        h["X-Audit-Token"] = os.environ["CUBBY_AUDIT_TOKEN"]
    req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=h,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def record(base: str) -> dict:
    out = {"asks": [], "board": {}}
    for q in QUESTIONS:
        code, rec = call(base, "/ask", {"text": q})
        j = call(base, f"/board/journey/{rec.get('id')}")[1] if isinstance(rec, dict) and rec.get("id") else None
        out["asks"].append({"question": q, "code": code, "record": rec, "journey": j})
        print(f"  {q[:50]:50s} -> {(rec or {}).get('answer')!r} ({(rec or {}).get('reason')})", flush=True)
    for p in BOARD:
        out["board"][p] = {"public": call(base, p), "audit": call(base, p, audit=True)}
    return out


def compare(a: dict, b: dict) -> dict:
    diffs = []
    for x, y in zip(a["asks"], b["asks"]):
        if strip(x) != strip(y):
            diffs.append({"question": x["question"], "before": strip(x["record"]), "after": strip(y["record"])})
    for p in a["board"]:
        if strip(a["board"][p]) != strip(b["board"].get(p)):
            diffs.append({"board": p})
    same_text = all((x["record"] or {}).get(k) == (y["record"] or {}).get(k) for x, y in zip(a["asks"], b["asks"])
                    for k in ("answer", "reply", "verified", "reason"))
    return {"same": not diffs, "replies_identical": same_text, "n_asks": len(a["asks"]), "differences": diffs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8792")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        r = compare(*(json.load(open(p, encoding="utf-8")) for p in a.compare))
        print(json.dumps({k: v for k, v in r.items() if k != "differences"}, indent=1), len(r["differences"]), "differences")
        for d in r["differences"][:10]:
            print(" ", json.dumps(d, ensure_ascii=False)[:300])
        return 0 if r["same"] else 1
    rec = record(a.base)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
