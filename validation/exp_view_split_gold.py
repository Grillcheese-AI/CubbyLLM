"""Order 0073, step 4: the reader's form, its grammar and the host's accounting, checked on every row of
`long_rows_0073` (counts only; no row text is printed or kept).

For each row and each arm:
- **the form:** `view_split.target` parses back (`parse`, strictly, and with ``own_text``) to the row's kinds, tags,
  said forms and spans;
- **the accounting:** `view_account.account` of those lines gives the row's groups and, byte for byte, the row's
  program (the join step 3 ran in the VM), with nothing asked or flagged;
- **the machine:** `view_machine.Machine` takes the target whole in both variants (so each of its prefixes is
  allowed), and arm M's machine refuses arm W's target wherever the two differ.
On dev and a seeded sample of train (the rows one may look at when something differs):
- **two implementations agree:** on one-slot changes of the gold answer (`view_split_probe.mutants`) the machine
  takes an answer whole if and only if `parse` reads it, in both arms and both variants.
With the base's tokenizer (skipped, and said so, when it is not found):
- **in ids:** gold targets walk to the end token through `view_tokens.Reader`, with the seconds an id; the machines
  of ``--machines`` dev instructions are walked at random in ids: every state reached has an id allowed or is a
  whole answer, and from a sample of them the plainest way on reaches a whole answer.

  python validation/exp_view_split_gold.py [--set $CUBBY_DATASETS/long_rows_0073] [--tokenizer <json>]
      [--mutant-rows 300] [--machines 50] [--id-rows 100] [--out validation/logs/<name>.json]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation")]
import view_split_probe as P  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import view_account as A  # noqa: E402
from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402
from cubbyllm.speech.view_machine import Machine  # noqa: E402

__wiring__ = "STANDALONE"

VARIANTS = ((False, "free"), (True, "own text"))


def gold(rows: list[dict]) -> dict:
    """The form, the accounting and the machine on every row's target."""
    c = Counter()
    for r in rows:
        text = r["instruction"]
        want = [(s["kind"], s.get("join"), V.span(text, s["start"], s["end"]),
                 r["said"][s["part"]] if s["kind"] == "check" else None) for s in r["segments"]]
        for arm in V.ARMS:
            t = V.target(r, arm)
            c[f"arm {arm}: targets"] += 1
            for own, name in VARIANTS:
                got = V.parse(t, text, arm, own_text=own).get("lines")
                c[f"arm {arm}, {name}: parsed back to the row's lines"] += got is not None and [
                    (x["kind"], x["join"], (x["start"], x["end"]), x["form"]) for x in got] == [
                    (k, j, sp, f and f[arm]) for k, j, sp, f in want]
                c[f"arm {arm}, {name}: taken whole by the machine"] += Machine(text, arm, own).accepts(t)
            acc = A.account(V.parse(t, text, arm)["lines"], text)
            c[f"arm {arm}: the accounting gives the row's program"] += acc["program"] == r["join"]
            c[f"arm {arm}: the accounting gives the row's groups"] += [list(g) for g in acc["groups"]] == r["groups"]
            c[f"arm {arm}: something asked or flagged"] += bool(acc["asked"] or acc["flagged"])
        if V.target(r, "m") != V.target(r, "w"):
            c["rows where the arms differ"] += 1
            c["arm W's target refused by arm M's machine"] += not Machine(text, "m").accepts(V.target(r, "w"))
    return dict(sorted(c.items()))


def agree(rows: list[dict], seed: int) -> dict:
    """The machine against `parse` on one-slot changes of the gold answer."""
    rng, c = random.Random(seed), Counter()
    for r in rows:
        text = r["instruction"]
        for arm in V.ARMS:
            lines = V.parse(V.target(r, arm), text, arm)["lines"]
            for what, answer in P.mutants(lines, text, rng):
                for own, name in VARIANTS:
                    a, b = Machine(text, arm, own).accepts(answer), "lines" in V.parse(answer, text, arm, own_text=own)
                    c[f"{name}: changes"] += 1
                    c[f"{name}: taken by both"] += a and b
                    c[f"{name}: refused by both"] += not a and not b
                    c[f"{name}: the two differ"] += a != b
                    if a != b:
                        c[f"{name}: the two differ on {what}"] += 1
                    c[f"kind of change: {what}"] += own
    return dict(sorted(c.items()))


def in_ids(rows: list[dict], vocab: T.Vocab, tk, eos: int) -> dict:
    """Gold targets through `Reader`, in the ids the trainer teaches (`talk_data`: one space, then the answer)."""
    c, secs, sizes = Counter(), [], []
    for r in rows:
        for arm in V.ARMS:
            ids = tk.encode(" " + V.target(r, arm), add_special_tokens=False).ids
            for own, name in VARIANTS:
                rd, t0 = T.Reader(Machine(r["instruction"], arm, own), vocab, eos), time.time()
                for i in ids:
                    allowed = rd.allowed()
                    sizes.append(len(allowed))
                    if i not in allowed:
                        break
                    rd.push(i)
                else:
                    if eos in rd.allowed():
                        rd.push(eos)
                secs.append((time.time() - t0) / max(1, len(ids)))
                c["targets"] += 1
                c["ended `finished`"] += rd.end == "finished"
                c["ids"] += len(ids)
    return dict(c, **{"seconds an id, median": round(statistics.median(secs), 5), "seconds an id, slowest target": round(max(secs), 5),
                      "ids allowed at a step, median": statistics.median(sizes), "ids allowed at a step, most": max(sizes)})


def walked(rows: list[dict], vocab: T.Vocab, seed: int, walks: int = 6, steps: int = 400, sample: int = 40) -> dict:
    """The machines of `rows` walked at random in ids."""
    rng, c = random.Random(seed), Counter()
    for r in rows:
        for arm in V.ARMS:
            for own, _ in VARIANTS:
                m, ids, states = Machine(r["instruction"], arm, own), {}, set()
                for _ in range(walks):
                    seen, stuck = P.walk_ids(m, vocab, rng, steps, ids)
                    states.update(seen)
                    c["states with no id allowed that are not a whole answer"] += stuck
                picked = rng.sample(sorted(states, key=repr), min(sample, len(states)))
                c["machines"] += 1
                c["states reached"] += len(states)
                c["states tried for a way on"] += len(picked)
                c["states with no way on to a whole answer"] += sum(not P.way_on(m, s) for s in picked)
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--mutant-rows", type=int, default=300)
    ap.add_argument("--machines", type=int, default=50)
    ap.add_argument("--id-rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(a.set or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0073")
    rows = [json.loads(ln) for name in ("rows.jsonl", "pair_rows.jsonl") for ln in open(ds / name, encoding="utf-8")]
    dev, train = [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "train"]
    rng = random.Random(a.seed)
    look = dev + rng.sample(train, min(a.mutant_rows, len(train)))
    out = {"order": "0073", "what": "step 4: the reader's form, its grammar and the host's accounting on long_rows_0073",
           "rows": len(rows), "sha256": json.loads((ds / "manifest.json").read_text(encoding="utf-8"))["sha256"],
           "every row": gold(rows), "the machine against parse": dict(agree(look, a.seed), rows=len(look))}
    tok = Path(a.tokenizer or Path(local_paths.value("CUBBY_MODELS") or ".") / "base450m" / "grilly" / "tokenizer.json")
    if tok.is_file():
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(tok))
        vocab, eos = T.Vocab.of(tk), tk.token_to_id("</s>")
        out["the vocabulary"] = {"ids": tk.get_vocab_size(), "pieces that print nothing": tk.get_vocab_size() - len(vocab.pieces),
                                 "characters of the instructions with no piece of their own": len(
                                     {ch for r in rows for ch in r["instruction"]} - {p for p in vocab.pieces.values() if len(p) == 1})}
        out["gold targets in ids"] = dict(in_ids(rng.sample(dev, min(a.id_rows, len(dev))), vocab, tk, eos), rows=min(a.id_rows, len(dev)))
        out["machines walked in ids"] = walked(rng.sample(dev, min(a.machines, len(dev))), vocab, a.seed)
    else:
        out["in ids"] = "the tokenizer was not found: the checks in ids were not run"
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
