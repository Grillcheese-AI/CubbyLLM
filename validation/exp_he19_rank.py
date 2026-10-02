"""H-E19, next-step ranking: is the right next step already in the model, behind a bad greedy pick?

The loop fails on WHICH step comes next (gate B: practice steps 75-90% right, held-out loop unchanged at 0.229,
5+ step questions 0/54). A step is one binary op on two values the host holds -- the question's numbers ($N)
and the earlier steps' VM values ($S) -- or the stop (`return $S<j>;`). That menu is small (~50-250) and the
host can list it, the way a vocabulary lists tokens. This read scores every candidate on the menu with the
adapter (teacher-forced log-prob of the whole emission, prompt shared) and asks where the right one ranks.

  top-1 high  -> generation is the problem, not knowledge: pick from the menu (argmax / beam) instead of writing
  top-5 high  -> beam search over the menu, VM-checked, should lift the loop with no training
  top-5 low   -> the knowledge is missing: train a next-step (pointer) head on the trunk

Candidates are grouped by what they compute (value scoring, like the judge: `a+b` and `b+a` are one choice);
a group scores its best spelling. Free: local, grilly2, no credits.

    python validation/exp_he19_rank.py --export <grilly export> --adapter <adapter dir> --tokenizer <bbpe128k>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from cubbyllm.reasoning.step_loop import render_step, render_stop  # noqa: E402

OPS = ("add", "sub", "mul", "div")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def _near(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def menu(row: dict) -> tuple[list[dict], dict]:
    """Every next step the host can offer for this row: one op on two held values, or a stop. Each candidate:
    {"text", "key" (what it computes), "correct"}. Also returns facts about the row (k, operands, is_stop)."""
    from emitter_data import table_of
    from cubbyllm.reasoning.slots import num_value
    table = table_of(row["spans"], row.get("question", ""))
    vals: dict[str, float] = {}
    for s in table.spans:
        if s.kind in ("N", "S") and s.id not in vals:
            v = num_value(s.filled)
            if v is not None:
                vals[s.id] = v
    sids = [i for i in vals if i.startswith("$S")]
    k = len(sids)
    gold_prog = row["program"].strip()
    is_stop = gold_prog.startswith("return")
    gold = num_value(str(row["gold"]))
    out = []
    ids = list(vals)
    for op in OPS:
        for a in ids:
            for b in ids:                                  # a == b too: `$N2 * $N2` is a gold step
                x, y = vals[a], vals[b]
                if op == "div" and y == 0:
                    continue
                v = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y if y else None}[op]
                out.append({"text": render_step(k, [("assign", a), (op, b)]), "key": ("v", round(v, 9)),
                            "correct": (not is_stop) and _near(v, gold)})
    for sid in sids:
        out.append({"text": render_stop(sid), "key": ("stop", sid),
                    "correct": is_stop and gold_prog.split()[1].rstrip(";") == sid})
    return out, {"k": k, "operands": len(ids), "is_stop": is_stop}


def score(model, tk, eos, prompt: str, texts: list[str], chunk: int) -> np.ndarray:
    """Sum log-prob of each candidate emission (its tokens and </s>) after the shared prompt, encoded the way
    training encoded them (prompt and target separately; target = text.rstrip() + newline)."""
    from exp_e5_base450m_probes import logprobs_of
    p = tk.encode(prompt).ids
    cs = [tk.encode(t.rstrip() + "\n").ids + [eos] for t in texts]
    out = np.zeros(len(cs))
    for s in range(0, len(cs), chunk):
        part = cs[s:s + chunk]
        keep = max(len(c) for c in part)
        T = len(p) + keep - 1
        x = np.zeros((len(part), T), dtype=np.int64)
        y = np.zeros((len(part), T), dtype=np.int64)
        for i, c in enumerate(part):
            seq = p + c
            x[i, :len(seq) - 1] = seq[:-1]
            y[i, :len(seq) - 1] = seq[1:]
        lp = logprobs_of(model, x, y, keep)                # (B, keep): positions len(p)-1 .. T-1
        for i, c in enumerate(part):
            out[s + i] = float(lp[i, :len(c)].sum())
    return out


def score_cached(model, tk, eos, prompt: str, texts: list[str], chunk: int) -> np.ndarray:
    """`score`, with the prompt read ONCE: the recurrent state (one vector per MinGRU layer, a 512-slot ring
    per attention layer) is the whole past, so it is prefilled on one row and copied to each chunk's rows."""
    import copy
    import grilly
    from grilly.nn import functional as GF
    p = tk.encode(prompt).ids
    cs = [tk.encode(t.rstrip() + "\n").ids + [eos] for t in texts]
    out = np.zeros(len(cs))
    with grilly.no_grad():
        base = model.make_cache(1, 0)
        model(grilly.tensor([p[:-1]]), past_key_values=base, logits_to_keep=1)
        for s in range(0, len(cs), chunk):
            part = cs[s:s + chunk]
            L = max(len(c) for c in part)
            x = np.zeros((len(part), L), dtype=np.int64)
            y = np.zeros((len(part), L), dtype=np.int64)
            for i, c in enumerate(part):
                x[i, :len(c)] = [p[-1]] + c[:-1]
                y[i, :len(c)] = c
            st = copy.copy(base)
            st.live = base.live.clone()
            st.select_rows(grilly.from_numpy(np.zeros(len(part), dtype=np.int64)))
            lp = np.zeros((len(part), L))
            for t in range(L):                             # decode path: one token per call against the state
                logits = model(grilly.from_numpy(np.ascontiguousarray(x[:, t:t + 1])), past_key_values=st).logits
                g = GF.log_softmax(logits[:, -1], -1).gather(-1, grilly.from_numpy(np.ascontiguousarray(y[:, t:t + 1])))
                lp[:, t] = g.numpy()[:, 0]
            for i, c in enumerate(part):
                out[s + i] = float(lp[i, :len(c)].astype(np.float64).sum())
            del st, logits, g                              # free the copied rings before the next chunk
        del base
    return out


def rank_row(cands: list[dict], scores: np.ndarray) -> dict:
    """Group by what a candidate computes; a group scores its best spelling. Where does the right group rank,
    and how much of the menu's probability does it hold?"""
    best: dict = {}
    for c, s in zip(cands, scores):
        g = best.setdefault(c["key"], {"score": -math.inf, "correct": False})
        g["score"] = max(g["score"], float(s))
        g["correct"] = g["correct"] or c["correct"]
    groups = sorted(best.values(), key=lambda g: -g["score"])
    pos = next((i for i, g in enumerate(groups) if g["correct"]), None)
    m = max(scores)
    z = np.exp(scores - m)
    mass = float(z[[c["correct"] for c in cands]].sum() / z.sum()) if any(c["correct"] for c in cands) else 0.0
    return {"rank": pos, "groups": len(groups), "mass": mass,
            "top_margin": (groups[0]["score"] - groups[1]["score"]) if len(groups) > 1 else None}


def summarize(rows: list[dict], label: str) -> str:
    n = len(rows)
    if not n:
        return f"  {label:<12} n=0"
    hit = lambda t: sum(r["rank"] is not None and r["rank"] < t for r in rows) / n
    return (f"  {label:<12} n={n:<4} top-1 {hit(1):.3f}  top-3 {hit(3):.3f}  top-5 {hit(5):.3f}  "
            f"right-group mass {sum(r['mass'] for r in rows) / n:.3f}  menu {sum(r['groups'] for r in rows) / n:.0f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "standin", "data", "out", "pf_heldout_eval_w_step.jsonl"))
    ap.add_argument("--n", type=int, default=300, help="step rows read (random, seeded); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64, help="candidates decoded together against the copied state (each copy carries a 512-slot ring per attention layer: 256 ran out of device memory)")
    ap.add_argument("--tag", default="held")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "arithmetic"]
    random.Random(a.seed).shuffle(rows)
    if a.n:
        rows = rows[:a.n]
    from emitter import Cubby450mEmitter
    em = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    em._load()
    model, tk, eos = em._model, em._tk, em._eos
    log(f"rank | {os.path.basename(os.path.normpath(a.adapter))} | {len(rows)} step rows from {os.path.basename(a.data)}")
    res, t0, uncovered = [], time.time(), 0
    for i, r in enumerate(rows, 1):
        cands, facts = menu(r)
        if not any(c["correct"] for c in cands):
            uncovered += 1
            res.append({"id": r["id"], **facts, "rank": None, "groups": 0, "mass": 0.0, "top_margin": None,
                        "covered": False})
            continue
        sc = score_cached(model, tk, eos, r["prompt"], [c["text"] for c in cands], a.chunk)   # parity with `score`: exact
        top = cands[int(np.argmax(sc))]["text"]
        res.append({"id": r["id"], **facts, **rank_row(cands, sc), "covered": True, "top": top.strip()})
        if i % 25 == 0:
            log(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s) " + summarize([x for x in res if x["covered"]], "so far").strip())
    cov = [x for x in res if x["covered"]]
    log(f"\n{a.tag}: {len(rows)} rows, {uncovered} not on the menu (a literal operand or a value the host lacks)")
    log(summarize(cov, "all"))
    log(summarize([x for x in cov if x["is_stop"]], "stop rows"))
    steps = [x for x in cov if not x["is_stop"]]
    log(summarize(steps, "step rows"))
    for k in (0, 1, 2):
        log(summarize([x for x in steps if x["k"] == k], f"  step {k}"))
    log(summarize([x for x in steps if x["k"] >= 3], "  step 3+"))
    log("  reference, teacher-forced greedy writing (same rows' family): step 0 0.758, later 0.549, stop 0.932")
    # is the margin a usable confidence? top-1 accuracy by the gap between the first and second choice
    m = sorted((x for x in cov if x["top_margin"] is not None), key=lambda x: x["top_margin"])
    for q in range(4):
        part = m[q * len(m) // 4:(q + 1) * len(m) // 4]
        if part:
            log(f"  margin quartile {q + 1} (gap {part[0]['top_margin']:.2f}..{part[-1]['top_margin']:.2f}): "
                f"top-1 {sum(x['rank'] == 0 for x in part) / len(part):.3f}")
    out = os.path.join(HERE, "logs", f"exp_he19_rank_{a.tag}.json")
    json.dump({"args": vars(a), "rows": res}, open(out, "w", encoding="utf-8"), indent=1)
    log(f"  -> {os.path.relpath(out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
