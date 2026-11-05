"""validation/system_eval/gpu.py -- order 0054's GPU part (Colab, Nick's run): the 450M's programs for the prompts
the stack handed its seat at home, and arm M's answers. No store, no VM, no network beyond Drive.

  programs   ``prompts.jsonl`` (from ``run_c.py --seat record``) -> ``programs.jsonl`` ({prompt, program}): greedy,
             as ``Cubby450mEmitter.emit`` writes them (``seats.programs``), the programs adapter on its own base
  m          the questions (``questions.jsonl``, ``bench.py --export``) -> ``m_<bench>.jsonl``: the 450M talk adapter
             alone, closed book, greedy, as the talk seat renders and cuts its answer (``talk_prompt`` /
             ``talk_answer``); CLUTRR's relations given as its facts. M never refuses: an empty answer is kept
             and graded as a refusal, and counted apart
Both resume: a prompt or question already written is skipped. ``--n`` (the preflight): the first n per benchmark,
then the time per question and the projected time for the full sets are printed.

    python validation/system_eval/gpu.py programs --prompts P --out O --ckpt C --adapter A --tokenizer T
    python validation/system_eval/gpu.py m --questions Q --out O --ckpt C --adapter A --tokenizer T [--n 50]

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

CHUNK = 256
M_MAX_NEW = 160                                          # the talk seat's own budget (``Cubby450mTalk.emit``)


def _done(path: str, key: str) -> set:
    return {json.loads(x)[key] for x in open(path, encoding="utf-8") if x.strip()} if os.path.exists(path) else set()


def m_prompt(row: dict) -> str:
    from cubbyllm.emit.talk import talk_prompt
    facts = (row.get("meta") or {}).get("facts")
    if row["bench"] == "clutrr" and facts:
        return talk_prompt(row["question"], {"facts": facts, "question": row["question"]})
    return talk_prompt(row["question"])


def m_answers(trunk, rows: list[dict], role: str = "talk") -> list[str]:
    """Each row's answer, as the talk seat cuts it."""
    from cubbyllm.emit.talk import talk_answer
    ids = [trunk.tokenizer.encode(m_prompt(r)).ids[-trunk.max_ctx:] for r in rows]
    outs = trunk.generate(role, ids, M_MAX_NEW)
    return [talk_answer(trunk.tokenizer.decode(o)) for o in outs]


def run_programs(a, trunk) -> dict:
    from system_eval.seats import programs
    todo = [r for r in map(json.loads, open(a.prompts, encoding="utf-8")) if r]
    have = _done(a.out, "prompt")
    todo = [r for r in todo if r["prompt"] not in have][: a.n or None]
    t0, n = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), CHUNK):
            part = todo[s:s + CHUNK]
            for r, p in zip(part, programs(trunk, [r["prompt"] for r in part], [r["max_new"] for r in part])):
                f.write(json.dumps({"prompt": r["prompt"], "program": p}, ensure_ascii=False) + "\n")
            n += len(part)
            print(f"  programs {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {"programs": n, "seconds": round(time.time() - t0, 1), "s_per": round((time.time() - t0) / max(1, n), 3)}


def run_m(a, trunk) -> dict:
    from system_eval.grade import verdict
    rows = [r for r in map(json.loads, open(a.questions, encoding="utf-8")) if r]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["bench"]].append(r)
    out, t_all = {}, time.time()
    for bench, rs in by.items():
        path = os.path.join(a.out, f"m_{bench}.jsonl")
        have = _done(path, "id")
        todo = [r for r in rs if r["id"] not in have][: a.n or None]
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for s in range(0, len(todo), CHUNK):
                part = todo[s:s + CHUNK]
                for r, ans in zip(part, m_answers(trunk, part)):
                    f.write(json.dumps({"id": r["id"], "bench": bench, "question": r["question"], "golds": r["golds"][:8],
                                        "seen": r["seen"], "answer": ans, "spoken": bool(ans.strip()),
                                        "verdict": verdict(bool(ans.strip()), ans, r["golds"], bench)},
                                       ensure_ascii=False) + "\n")
                print(f"  m {bench} {s + len(part)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        dt = time.time() - t0
        out[bench] = {"answered": len(todo), "rows": len(rs), "s_per": round(dt / max(1, len(todo)), 3),
                      "projected_min": round(dt / max(1, len(todo)) * len(rs) / 60, 1)}
    out["seconds"] = round(time.time() - t_all, 1)
    return out


def main(argv=None) -> None:
    from system_eval.seats import TorchTrunk
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("programs", "m"))
    ap.add_argument("--prompts", help="programs: prompts.jsonl")
    ap.add_argument("--questions", help="m: questions.jsonl")
    ap.add_argument("--out", required=True, help="programs: programs.jsonl; m: a folder")
    ap.add_argument("--ckpt", required=True, help="the adapter's base checkpoint")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=0, help="the preflight: the first n (per benchmark for m)")
    ap.add_argument("--rows", type=int, default=64, help="rows decoded together")
    a = ap.parse_args(argv)
    t0 = time.time()
    role, name = ("programs", "emitter_lora") if a.step == "programs" else ("talk", "talk_lora")
    trunk = TorchTrunk({role: (a.ckpt, a.adapter, name)}, a.tokenizer, rows=a.rows)
    print(f"{a.step}: loaded in {time.time() - t0:.0f}s on {trunk.dev}", flush=True)
    if a.step == "m":
        os.makedirs(a.out, exist_ok=True)
    res = run_programs(a, trunk) if a.step == "programs" else run_m(a, trunk)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
