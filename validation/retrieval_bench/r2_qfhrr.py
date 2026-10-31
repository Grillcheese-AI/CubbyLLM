"""r2_qfhrr -- order 0046, R2's qFHRR arm: the question's context against one prototype per item, on grilly2.

The arm starts as every R2 arm does (``r2_arms._start``: the items that bear the name, then, when the relation is
known, those that carry it). Among what is left:
  prototype   the item's own words, the same ones BM25 reads (``r2_arms.doc``: its description in the row's language
              and in English, its values), as one qFHRR code (``cubbyllm.worlds.fast_words``, the serving table)
  context     the question's words, the same ones BM25 reads (``r2_arms.cues``: the mention and stopwords out)
  score       the cosine of the two codes, computed on grilly2 (``r2_qfhrr_score``, under grilly2's own interpreter)
  pick        the best item, only when its rank-normalized margin over the next, (s1 - s2) / s1, is at least ``tau``
              (set on dev, as BM25's is); otherwise the namesakes stay "several"
Only the scoring differs from BM25: the words are the same, so the two arms compare the similarity, not the inputs.

Where grilly2 is: ``$GRILLY2_DIR`` (the environment, then ``paths.local.json``), else the checkout beside this repo.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

__wiring__ = "STANDALONE"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCORER = os.path.join(HERE, "r2_qfhrr_score.py")


def grilly2_dir() -> str:
    from cubbyllm.local_paths import value
    d = value("GRILLY2_DIR") or os.path.join(os.path.dirname(ROOT), "grilly2")
    if not os.path.isfile(os.path.join(d, "pyproject.toml")):
        raise SystemExit("no grilly2 checkout: set GRILLY2_DIR (the qFHRR arm runs on grilly2, never grilly v1)")
    return d


def job(rows: list[dict], profiles: dict, table: str) -> tuple[dict, dict]:
    """(the job for the scorer, {row id: an outcome decided before any score}) -- the rows ``_start`` settles (a refusal,
    or one item left) never reach the card."""
    from retrieval_bench import r2_arms as A
    docs, todo, done = {}, [], {}
    for r in rows:
        left, out = A._start(r["mention"], r["cands"], profiles, r["pids"])
        if out is not None:
            done[r["id"]] = out
            continue
        keys = [f"{q}|{r['lang']}" for q in left]
        for q, k in zip(left, keys):
            docs.setdefault(k, " ".join(A.doc(profiles[q], r["lang"])))
        todo.append({"id": r["id"], "cues": " ".join(A.cues(r["question"], r["mention"])), "items": keys})
    return {"table": table, "docs": docs, "rows": todo}, done


def score(jb: dict) -> dict:
    """Run the scorer under grilly2's interpreter; its output, read back."""
    tmp = tempfile.mkdtemp()
    jp, op = os.path.join(tmp, "job.json"), os.path.join(tmp, "out.json")
    json.dump(jb, open(jp, "w", encoding="utf-8"))
    subprocess.run(["uv", "run", "--project", grilly2_dir(), "python", SCORER, jp, op], check=True, cwd=ROOT)
    return json.load(open(op, encoding="utf-8"))


def decide(items: list[str], scores: list[float], tau: float) -> dict:
    ranked = sorted(zip(items, scores), key=lambda kv: (-kv[1], kv[0]))
    s1, s2 = ranked[0][1], ranked[1][1]
    margin = (s1 - s2) / s1 if s1 > 0 else 0.0
    if s1 > 0 and margin >= tau:
        return {"outcome": "pick", "item": ranked[0][0], "by": "the question's context", "margin": round(margin, 4)}
    return {"outcome": "several", "items": sorted(items), "margin": round(margin, 4)}


def arms(rows: list[dict], profiles: dict, taus: list[float], table: str) -> tuple[dict, dict]:
    """({row id: {"qfhrr@<tau>": outcome}}, the scorer's report without its scores)."""
    jb, done = job(rows, profiles, table)
    rep = score(jb) if jb["rows"] else {"scores": {}}
    out = {}
    for rid, o in done.items():
        out[rid] = {f"qfhrr@{t}": o for t in taus}
    for r in jb["rows"]:
        items = [k.split("|")[0] for k in r["items"]]
        out[r["id"]] = {f"qfhrr@{t}": decide(items, rep["scores"][r["id"]], t) for t in taus}
    return out, {k: v for k, v in rep.items() if k != "scores"}


__all__ = ["arms", "decide", "grilly2_dir", "job", "score"]
