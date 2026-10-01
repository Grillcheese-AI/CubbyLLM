"""exp_he19_next_step - H-E19's memory read: is the 450M a good LOCAL decider when the state is held for it?

Teacher-forced: for every arithmetic record, the prompt plus the GOLD program up to the end of step k-1 is
the prefix, and the adapter generates only the next block. Scored per block: the next step's operation
sequence (`op`), its operands (`operands`), both (`exact`), and the STOP decision (the closing
`sum/query/return` where the plan is complete, another step where it is not). Read by block index k and
by the plan's length K, so a fall with k under a gold prefix means the model is not reading its own
context (the window), not that it lost state it never had to hold.

The whole-program read (exp_he19_gate_a) is the same adapter in one pass. If per-block accuracy is high
where whole-program is near zero, the calling convention -- one pass, the plan in hidden state -- was
the failure, not the model (H-P3: the recurrence's time constant is ~1 step).

    python validation/exp_he19_next_step.py --export <grilly export of the base> --adapter <adapter dir> \
        --tokenizer <bbpe128k json> --data standin/data/out/pf_heldout_eval_w_slots.jsonl --tag tg_held
    python validation/exp_he19_next_step.py --gguf standin/models/emitter_v12e.Q4_K_M.gguf \
        --data standin/data/out/pf_heldout_eval.jsonl --tag standin_held        # the 2.6B, same read
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CREATE = re.compile(r"^\s*create\s+(s\d+)\s*:")
_OP = re.compile(r"^\s*(assign)\s+(s\d+)\s*=\s*([^;]+);|^\s*(add|sub|mul|div)\s+(s\d+)\s*,\s*([^;]+);")
_CLOSE = re.compile(r"^\s*(sum|query|return)\b")
CLOSE = "<close>"


def blocks_of(program: str) -> tuple[list[str], list[list[tuple[str, str]]], int]:
    """(prefix lines before the first step, the blocks as op lists, index of the first solve-body line).
    A block is a `create` line and the ops that follow it; the last block is CLOSE (the sum/query/return)."""
    lines = program.splitlines()
    starts = [i for i, l in enumerate(lines) if _CREATE.match(l)]
    close = next((i for i, l in enumerate(lines) if _CLOSE.match(l)), None)
    if not starts or close is None:
        return lines, [], 0
    bounds = starts + [close]
    blocks = []
    for a, b in zip(bounds, bounds[1:]):
        blocks.append(ops_of(lines[a:b]))
    return lines, blocks, starts[0]


def ops_of(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for l in lines:
        m = _OP.match(l)
        if m:
            out.append((m.group(1), m.group(3).strip(), m.group(2)) if m.group(1) else (m.group(4), m.group(6).strip(), m.group(5)))
    return out


def first_block(text: str) -> tuple[str, list[tuple[str, str]], bool]:
    """What the model emitted first: ('step', its ops, complete?) or (CLOSE, [], True) or ('other', [], False).
    Complete means a second `create` or the closing followed, so the block was not cut by the token budget."""
    lines = text.splitlines()
    if not lines:
        return "other", [], False
    i = next((i for i, l in enumerate(lines) if l.strip()), None)
    if i is None:
        return "other", [], False
    if _CLOSE.match(lines[i]):
        return CLOSE, [], True
    if not _CREATE.match(lines[i]):
        return "other", ops_of(lines), False
    j = next((j for j in range(i + 1, len(lines)) if _CREATE.match(lines[j]) or _CLOSE.match(lines[j])), None)
    return "step", ops_of(lines[i + 1:(j if j is not None else len(lines))]), j is not None


def simulate(ops: list[tuple[str, str]], regs: dict, slots: dict, target: str | None = None):
    """Run one block's ops over the register values; the block's register is the one its first op names.
    Returns (register, value) or (None, None) on a malformed op or an unknown reference."""
    regs = dict(regs)
    reg = target
    for o, v, *_ in ops:
        try:
            x = Fraction(slots[v]) if v.startswith("$") else regs[v] if re.fullmatch(r"s\d+", v) else Fraction(v.replace(",", ""))
        except (KeyError, ValueError, ZeroDivisionError):
            return None, None
        if o == "assign":
            regs[reg] = x
        elif reg not in regs:
            return None, None
        elif o == "add":
            regs[reg] += x
        elif o == "sub":
            regs[reg] -= x
        elif o == "mul":
            regs[reg] *= x
        elif o == "div":
            if x == 0:
                return None, None
            regs[reg] /= x
    return reg, regs.get(reg)


def canon_ops(ops):
    """The ops with s-registers kept and every value as written ($N1, 2, 3.5): order matters."""
    return [(o, re.sub(r"\s+", "", v)) for o, v, *_ in ops]


def score_block(gold: list, got_kind: str, got_ops: list, complete: bool, regs: dict | None = None,
                slots: dict | None = None) -> dict:
    if gold == CLOSE:
        return {"kind": "close", "stop_right": got_kind == CLOSE, "ran_on": got_kind == "step"}
    g, h = canon_ops(gold), canon_ops(got_ops)
    out = {"kind": "step", "stopped_early": got_kind == CLOSE, "other": got_kind == "other",
           "op": [o for o, _ in g] == [o for o, _ in h], "operands": sorted(v for _, v in g) == sorted(v for _, v in h),
           "exact": g == h, "complete": complete}
    if regs is not None:
        greg = gold[0][2] if gold and len(gold[0]) > 2 else None
        hreg = got_ops[0][2] if got_ops and len(got_ops[0]) > 2 else None
        _, gv = simulate(gold, regs, slots, greg)
        _, hv = simulate(got_ops, regs, slots, hreg)
        out["value"] = gv is not None and hv == gv        # the step computed the right number, however written
    return out


def load(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == "arithmetic" and r.get("split", "val") == "val":
                out.append(r)
    return out


def steps_of(r: dict) -> int | None:
    m = re.search(r"steps=(\d+)", r.get("subtype") or "")
    return int(m.group(1)) if m else None


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(Counter)
    for row in rows:
        for key in ("_all", f"k={row['k']}", f"K={row['K']}"):
            c = by[key]
            s = row["score"]
            if s["kind"] == "step":
                c["steps"] += 1
                for f in ("op", "operands", "exact", "stopped_early", "other", "complete", "value"):
                    c[f] += int(bool(s.get(f)))
            else:
                c["closes"] += 1
                c["stop_right"] += int(s["stop_right"])
                c["ran_on"] += int(s["ran_on"])
    out = {}
    for key, c in by.items():
        d = {"steps": c["steps"], "closes": c["closes"]}
        if c["steps"]:
            for f in ("op", "operands", "exact", "stopped_early", "other", "complete", "value"):
                d[f] = c[f] / c["steps"]
        if c["closes"]:
            d["stop_right"] = c["stop_right"] / c["closes"]
            d["ran_on"] = c["ran_on"] / c["closes"]
        out[key] = d
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", default="", help="grilly export of the 450M base (with --adapter and --tokenizer)")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--gguf", default="", help="the stand-in instead: a GGUF (llama.cpp), raw prompts, programs with literals")
    ap.add_argument("--data", required=True, help="a *_slots.jsonl; its arithmetic val records are read")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=56)
    ap.add_argument("--replay", help="score the gold program itself as the generation (no model; the parser's round trip)")
    ap.add_argument("--inject-state", action="store_true",
                    help="the zero-training probe of the write-back: every value the gold prefix computed is ALSO placed "
                         "in the question as a further number slot (`So far: [N6: 154]`), so a later step can chain "
                         "from it the way it chains from a question number; scored by value")
    a = ap.parse_args(argv)
    recs = load(a.data)
    if a.limit:
        recs = recs[:a.limit]
    out_path = os.path.join(HERE, "logs", f"exp_he19_next_step_{a.tag}.json")
    emitter = None
    if a.replay:
        pass
    elif a.gguf:
        from standin.emitter import LlamaCppEmitter
        emitter = LlamaCppEmitter(a.gguf)
    else:
        from standin.emitter import Cubby450mEmitter
        emitter = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    rows, t0 = [], time.perf_counter()
    n_gen = 0
    for i, r in enumerate(recs):
        lines, blocks, first = blocks_of(r["program"])
        if not blocks:
            continue
        K = len(blocks)
        gold_blocks = blocks + [CLOSE]
        starts = [j for j, l in enumerate(lines) if _CREATE.match(l)]
        close = next(j for j, l in enumerate(lines) if _CLOSE.match(l))
        cut_points = starts + [close]                     # the prefix for block k ends just before line cut_points[k]
        slots = {}
        for sp in r.get("spans", []):
            try:
                slots[sp["id"]] = Fraction(str(sp.get("value") or sp["text"]).replace(",", "").replace("$", ""))
            except (ValueError, ZeroDivisionError):
                pass
        regs: dict = {}
        computed: list = []                               # (register, value) in order, for --inject-state
        n_slots = len({sp["id"] for sp in r.get("spans", []) if sp["id"].startswith("$N")})
        for k, gold in enumerate(gold_blocks):
            prefix = "\n".join(lines[:cut_points[k]]) + "\n"
            prompt = r["prompt"]
            if a.inject_state and computed:
                extra = " ".join(f"[N{n_slots + j + 1}: {str(v) if v.denominator == 1 else float(v)}]"
                                 for j, (_, v) in enumerate(computed))
                head, tail = prompt.rsplit("\nProgram:\n", 1)
                prompt = head + "\nSo far: " + extra + "\nProgram:\n" + tail
                for j, (_, v) in enumerate(computed):        # j, not i: i is the record index the progress line reads
                    slots[f"$N{n_slots + j + 1}"] = v
            if a.replay:
                text = "\n".join(lines[cut_points[k]:]) + "\n"
            else:
                text = emitter.emit(prompt, max_new_tokens=a.max_new, prefix=prefix)[len(prefix):]
                n_gen += 1
            kind, ops, complete = first_block(text)
            rows.append({"id": r["id"], "k": k, "K": K, "steps": steps_of(r), "gold": gold if gold == CLOSE else canon_ops(gold),
                         "got": {"kind": kind, "ops": canon_ops(ops)},
                         "score": score_block(gold, kind, ops, complete, regs, slots), "text": text[:400]})
            if gold != CLOSE:                             # the gold state moves on whatever the model said
                greg, gv = simulate(gold, regs, slots, gold[0][2] if gold else None)
                if greg is not None and gv is not None:
                    regs[greg] = gv
                    computed.append((greg, gv))
        if (i + 1) % 20 == 0 or i + 1 == len(recs):
            s = summarize(rows)["_all"]
            print(f"  {i + 1}/{len(recs)} records, {n_gen} generations ({time.perf_counter() - t0:.0f}s)  "
                  f"exact={s.get('exact', 0):.3f} value={s.get('value', 0):.3f} op={s.get('op', 0):.3f} operands={s.get('operands', 0):.3f} "
                  f"stop_right={s.get('stop_right', 0):.3f} stopped_early={s.get('stopped_early', 0):.3f}", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"adapter": a.adapter or a.gguf, "data": a.data, "n_records": i + 1, "summary": summarize(rows), "rows": rows}, f, indent=1)
    s = summarize(rows)
    print("\nby block index k (0 = first step):")
    for key in sorted((k for k in s if k.startswith("k=")), key=lambda x: int(x[2:])):
        d = s[key]
        print(f"  {key:5s} steps={d['steps']:4d} exact={d.get('exact', 0):.3f} value={d.get('value', 0):.3f} op={d.get('op', 0):.3f} "
              f"operands={d.get('operands', 0):.3f}  closes={d['closes']:3d} stop_right={d.get('stop_right', 0):.3f}")
    print("by plan length K:")
    for key in sorted((k for k in s if k.startswith("K=")), key=lambda x: int(x[2:])):
        d = s[key]
        print(f"  {key:5s} steps={d['steps']:4d} exact={d.get('exact', 0):.3f} value={d.get('value', 0):.3f} op={d.get('op', 0):.3f}"
              f"  closes={d['closes']:3d} stop_right={d.get('stop_right', 0):.3f}")
    d = s["_all"]
    print(f"ALL steps={d['steps']} exact={d.get('exact', 0):.3f} value={d.get('value', 0):.3f} op={d.get('op', 0):.3f} operands={d.get('operands', 0):.3f} "
          f"stopped_early={d.get('stopped_early', 0):.3f} incomplete={1 - d.get('complete', 1):.3f} | closes={d['closes']} "
          f"stop_right={d.get('stop_right', 0):.3f} ran_on={d.get('ran_on', 0):.3f}  -> {out_path}")


if __name__ == "__main__":
    main()
