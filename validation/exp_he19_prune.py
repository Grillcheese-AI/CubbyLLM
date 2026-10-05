"""H-E19, model-free step pruning: which VM-side rules could cut wrong menu entries without ever cutting the gold step?

Gemini's suggestions for a no-LLM verifier, made concrete on our step menu (exp_he19_rank.menu: one op on two held
values, or a stop). Candidate rules, all domain-general (no unit knowledge):
  neg       the result is negative
  frac      the result is fractional while both operands are whole
  noop      the result equals a value already held (no progress: x*1, x/1, x+0, or a recomputation)
  zero      the result is 0 (x - x)
  self      both operands are the same slot (x op x)
and at a stop: unused = some question number never used by the trace (soft: questions carry distractors).

GENERALISATION PROTOCOL: rules are judged on the TRAIN rows (GSM8K-train step rows, the emitter's own data) and a
rule is kept only if it removes the gold step in <= 1% of train rows; the held-out pf rows (other wordings, other
numbers, the 236-question family) only CONFIRM -- nothing is tuned on them. Both are reported by step depth, so a
rule that only holds for shallow steps shows. Also checks whether held-out problem templates (program GSM<n>) occur
in training -- whether the held-out set tests new structure or new surface only. CPU, no model.

    python validation/exp_he19_prune.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_he19_rank import OPS, ROOT, _near  # noqa: E402  (also sets sys.path for emitter_data / cubbyllm)

LOG: list[str] = []
RULES = ("neg", "frac", "noop", "zero", "self")


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def whole(x: float) -> bool:
    return abs(x - round(x)) <= 1e-9 * max(1.0, abs(x))


def candidates(row: dict):
    """(gold_found, [(op, a, b, value, correct)], held values) -- the step menu with operands kept."""
    from emitter_data import table_of
    from cubbyllm.reasoning.slots import num_value
    table = table_of(row["spans"], row.get("question", ""))
    vals: dict[str, float] = {}
    for s in table.spans:
        if s.kind in ("N", "S") and s.id not in vals:
            v = num_value(s.filled)
            if v is not None:
                vals[s.id] = v
    gold = num_value(str(row["gold"]))
    out = []
    for op in OPS:
        for a in vals:
            for b in vals:
                x, y = vals[a], vals[b]
                if op == "div" and y == 0:
                    continue
                v = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y if y else None}[op]
                out.append((op, a, b, v, _near(v, gold)))
    return any(c[4] for c in out), out, vals


def flags(op, a, b, v, vals) -> dict:
    x, y = vals[a], vals[b]
    return {"neg": v < -1e-12,
            "frac": whole(x) and whole(y) and not whole(v),
            "noop": any(_near(v, h) for h in vals.values()),
            "zero": abs(v) <= 1e-12,
            "self": a == b}


def depth(row: dict) -> str:
    m = re.match(r"step (\d+)/", row.get("subtype") or "")
    if not m:
        return "?"
    d = int(m.group(1))
    return str(d) if d < 3 else "3+"


def screen(rows: list[dict], label: str) -> dict:
    """Per rule: gold-step violation rate, and the share of wrong candidates (and wrong value groups) it removes."""
    gold_hit = defaultdict(lambda: defaultdict(int))
    wrong_cut = defaultdict(lambda: defaultdict(int))
    n_rows = defaultdict(int); n_wrong = defaultdict(int); no_gold = 0
    for row in rows:
        found, cands, vals = candidates(row)
        if not found:
            no_gold += 1
            continue
        d = depth(row)
        for key in (d, "all"):
            n_rows[key] += 1
        gold_flags = [flags(*c[:4], vals) for c in cands if c[4]]
        for r in RULES:                                  # the gold step survives if ANY spelling of it passes the rule
            if all(f[r] for f in gold_flags):
                for key in (d, "all"):
                    gold_hit[key][r] += 1
        if all(any(f[r] for r in RULES) for f in gold_flags):
            for key in (d, "all"):
                gold_hit[key]["ANY"] += 1
        for c in cands:
            if c[4]:
                continue
            f = flags(*c[:4], vals)
            for key in (d, "all"):
                n_wrong[key] += 1
                for r in RULES:
                    wrong_cut[key][r] += f[r]
                wrong_cut[key]["ANY"] += any(f.values())
    log(f"\n{label}: {len(rows)} step rows ({no_gold} whose gold is not on the menu, skipped)")
    log(f"  {'rule':<6} {'depth':<5} {'rows':>6} {'gold cut':>9} {'wrong cut':>10}")
    res = {}
    for r in RULES + ("ANY",):
        for key in ("0", "1", "2", "3+", "all"):
            if not n_rows[key]:
                continue
            g = gold_hit[key][r] / n_rows[key]; w = wrong_cut[key][r] / max(1, n_wrong[key])
            res.setdefault(r, {})[key] = {"rows": n_rows[key], "gold_cut": g, "wrong_cut": w}
            if key == "all" or r == "ANY":
                log(f"  {r:<6} {key:<5} {n_rows[key]:>6} {g:>9.4f} {w:>10.4f}")
    return res


def unused_at_stop(rows: list[dict], label: str) -> dict:
    """Full-program rows: share of gold traces that leave a question number unused (the 'use every number' rule)."""
    n = unused = 0
    for row in rows:
        prog = row.get("program") or ""
        ids = {s["id"] for s in row.get("spans", []) if s.get("kind") == "N"}
        if not ids or "return" not in prog:
            continue
        used = set(re.findall(r"\$N\d+", prog))
        n += 1
        unused += bool(ids - used)
    log(f"  {label}: {unused}/{n} gold traces leave a question number unused ({unused / max(1, n):.3f})")
    return {"n": n, "unused": unused}


def read(path: str, pred, limit: int) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == "arithmetic" and pred(r):
                out.append(r)
                if len(out) >= limit:
                    break
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    d = os.path.join(ROOT, "standin", "data", "out")
    ap.add_argument("--train", default=os.path.join(d, "emitter_sft_v12e_w_tg30_step.jsonl"))
    ap.add_argument("--train-full", default=os.path.join(d, "emitter_sft_v12e_w_slots.jsonl"))
    ap.add_argument("--held", default=os.path.join(d, "pf_heldout_eval_w_step.jsonl"))
    ap.add_argument("--held-full", default=os.path.join(d, "pf_heldout_eval_w_slots.jsonl"))
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args(argv)
    t0 = time.time()
    is_step = lambda r: (r.get("subtype") or "").startswith("step ")
    train = read(a.train, lambda r: r.get("split") == "train" and is_step(r), a.n_train)
    held = read(a.held, is_step, 10 ** 9)
    log(f"he19 prune | train step rows {len(train)} (GSM8K-train), held-out step rows {len(held)} (pf) "
        f"({time.time() - t0:.0f}s)")
    report = {"train": screen(train, "TRAIN (rules are judged here)"),
              "held": screen(held, "HELD-OUT pf (confirmation only)")}
    log("\nstop rule 'every question number used':")
    tf = read(a.train_full, lambda r: r.get("split") == "train", 20000)
    hf = read(a.held_full, lambda r: True, 10 ** 9)
    report["unused_train"] = unused_at_stop(tf, "train")
    report["unused_held"] = unused_at_stop(hf, "held-out")
    tmpl = lambda rows: {m for r in rows for m in re.findall(r"program (GSM\d+)", (r.get("reference") or "") + (r.get("program") or ""))}
    th, tt = tmpl(hf), tmpl(tf)
    log(f"\ntemplates: held-out uses {len(th)} GSM templates; {len(th & tt)} of them also name a train program "
        f"(train sample names {len(tt)})")
    report["templates"] = {"held": len(th), "overlap": len(th & tt), "train": len(tt)}
    out = os.path.join(HERE, "logs", f"exp_he19_prune_{a.tag}")
    json.dump({"args": vars(a), **report}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"\n  -> {os.path.relpath(out, ROOT)}.{{log,json}} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
