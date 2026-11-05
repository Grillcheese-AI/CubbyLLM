"""validation/system_eval/grade.py -- order 0054's grading, with no judge model.

  a spoken answer that matches a gold or an alias after normalization is right (``exp_everyday_webq.matches``;
  CLUTRR: the relation word, normalized, equal); one that matches none is ``unmatched`` until a person reads it
  (WebQuestions' golds are Freebase's of 2013: a value that changes is read, never counted wrong by the normalizer);
  a refusal is ``refused``; M's text decline ("The facts don't say.", "I don't know ...") is ``declined``, counted
  with the refusals and reported apart
  the read   C's unmatched answers all; C-VM's and M's by a seeded sample of 200 per arm and benchmark
             (``to_read``). A read's verdicts (``reads.jsonl``: {arm, id, verdict: right | wrong, why}) turn
             unmatched into right or wrong; the unread rest is estimated from the sample's right share
  the rates  right, wrong, refused (counts and shares); the wrong rate among spoken answers with a Wilson 95%
             interval; for an estimated arm, the right and wrong counts with the sample's interval carried

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

READ_SAMPLE = 200
SEED = 54
READ_ALL = ("C",)                          # arms whose unmatched answers are all read


def matches(answer: str, golds: list[str], bench: str) -> bool:
    if bench == "clutrr":                   # the relation word: the answer, or a whole word of a sentence (M writes text)
        from cubbyllm.reasoning.planner import normalize
        a = normalize(answer or "")
        return bool(a) and any(a == normalize(g) or f" {normalize(g)} " in f" {a} " for g in golds)
    from exp_everyday_webq import matches as m
    return m(answer or "", list(golds))


DECLINES = ("the facts don't say", "the facts do not say", "i don't know", "i do not know")


def declined(answer: str | None) -> bool:
    """M's own decline in text: the talk seat's absent line (``ask_words.ABSENT_REPLY``) or an "I don't know" opening
    (the talk sets train both). Found in the CPU smoke of 2026-11-05: the order expected M never to refuse."""
    a = " ".join(str(answer or "").lower().replace("’", "'").split())
    return a.startswith(DECLINES)


def verdict(spoken: bool, answer: str | None, golds: list[str], bench: str, declines: bool = False) -> str:
    """right | unmatched | refused (| declined: M's text decline, when ``declines``; counted with the refusals)."""
    if not spoken or not (answer or "").strip():
        return "refused"
    if declines and declined(answer):
        return "declined"
    return "right" if matches(answer, golds, bench) else "unmatched"


def to_read(rows: list[dict], arm: str, n: int = READ_SAMPLE, seed: int = SEED) -> list[str]:
    """The ids of ``rows`` (one arm, one benchmark; each with ``verdict``) to read: every unmatched one for C, else a
    seeded sample of ``n`` (ids sorted first, so a rerun draws the same)."""
    ids = sorted(r["id"] for r in rows if r["verdict"] == "unmatched")
    if arm in READ_ALL or len(ids) <= n:
        return ids
    return sorted(random.Random(f"{seed}:{arm}").sample(ids, n))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of k successes in n (0, 0 for n = 0)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rates(rows: list[dict], reads: dict[str, str] | None = None) -> dict:
    """The three rates of one arm on one benchmark. ``reads``: {id: right | wrong} from the read. Unmatched answers
    left unread are split by the read's right share (an estimate, said so, with the interval of that share)."""
    reads = reads or {}
    n = len(rows)
    right = sum(r["verdict"] == "right" for r in rows)
    refused = sum(r["verdict"] in ("refused", "declined") for r in rows)
    declined_n = sum(r["verdict"] == "declined" for r in rows)
    un = [r["id"] for r in rows if r["verdict"] == "unmatched"]
    read = [i for i in un if i in reads]
    read_right = sum(reads[i] == "right" for i in read)
    read_wrong = len(read) - read_right
    rest = len(un) - len(read)
    share = read_right / len(read) if read else None
    lo, hi = wilson(read_right, len(read))
    est_right = right + read_right + (rest * share if share is not None else 0.0)
    est_wrong = read_wrong + (rest * (1 - share) if share is not None else float(rest))
    spoken = n - refused
    out = {"n": n, "matched": right, "unmatched": len(un), "read": len(read), "read_right": read_right,
           "read_wrong": read_wrong, "unread": rest, "refused": refused, "declined": declined_n, "spoken": spoken,
           "right": round(est_right, 1), "wrong": round(est_wrong, 1), "estimated": rest > 0,
           "right_pct": round(100 * est_right / max(1, n), 2), "wrong_pct": round(100 * est_wrong / max(1, n), 2),
           "refused_pct": round(100 * refused / max(1, n), 2)}
    if rest == 0:
        k = read_wrong
        wl, wh = wilson(k, spoken)
        out["wrong_of_spoken"] = {"rate": round(k / max(1, spoken), 4), "ci95": [round(wl, 4), round(wh, 4)]}
    else:                                   # the read's wrong share, carried to the unread: its interval, widened
        wl = (read_wrong + rest * (1 - hi)) / max(1, spoken) if share is not None else None
        wh = (read_wrong + rest * (1 - lo)) / max(1, spoken) if share is not None else None
        out["wrong_of_spoken"] = {"rate": round(est_wrong / max(1, spoken), 4),
                                  "ci95": [round(wl, 4), round(wh, 4)] if wl is not None else None,
                                  "from_sample": len(read)}
    return out


__all__ = ["DECLINES", "READ_ALL", "READ_SAMPLE", "SEED", "declined", "matches", "rates", "to_read", "verdict", "wilson"]
