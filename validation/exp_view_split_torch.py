"""Order 0073, step 6: the reader's read on a CUDA card, one decoder for every arm (the answers only; they are
scored at home on the shared VM).

A batch of instructions decodes in lockstep on `CubbyModel.step`, as round 2's read did
(`exp_view_reader_torch.py`). What differs is how a row's next token is chosen:
- **arm m, arm w** (this order's adapters): the answer is held by the machine of its instruction
  (`cubbyllm/speech/view_machine.py`; in ids, `view_tokens.Reader`). The ids a state allows are looked up, and the
  next token is the best-scored of them. ``--own-text`` holds each description to its line's own text (point 7's
  variant: decided on dev, then frozen).
- **arm p** (round 2's adapter with its own form and grammar, rules 1 to 9): round 2's chooser, with an id whose
  piece prints nothing never offered.
Every answer records how it ended (``end``: finished, error, limit) and how many ids it took (``ids``). The rows of
a split are read in the file's order (``--n``: the first so many; 300 is a whole split of `long_rows_0073`).

  python validation/exp_view_split_torch.py --arm m|w|p --ckpt <base.pt> --tokenizer <json> --adapter <dir>
      --rows <rows.jsonl> [--splits dev,test,test_long] [--n 300] [--batch 32] [--max-new 900] [--own-text]
      --out <answers.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from cubbyllm.speech import view_split as V  # noqa: E402
from cubbyllm.speech import view_tokens as T  # noqa: E402
from cubbyllm.speech.view_machine import Machine  # noqa: E402

__wiring__ = "STANDALONE"

WIDE = 2048            # above this many allowed ids, the best-scored 64 are tried before the whole set is scored


def best(scores, ids: frozenset) -> int:
    """The best-scored of `ids` (`scores`: one row's logits)."""
    import torch
    if len(ids) > WIDE:
        for t in torch.topk(scores, 64).indices.tolist():
            if t in ids:
                return t
    idx = torch.tensor(sorted(ids), device=scores.device)
    return int(idx[torch.argmax(scores[idx])])


def decode(model, tk, vocab: T.Vocab, eos: int, batch: list, arm: str, own_text: bool, max_new: int, dev, clock=None) -> list[dict]:
    """One batch of rows -> [{"id", "split", "arm", "answer", "end", "ids"}], arms m and w."""
    import torch
    clock = {} if clock is None else clock
    prompts = [tk.encode(V.prompt(r["instruction"])).ids for r in batch]
    readers = [T.Reader(Machine(r["instruction"], arm, own_text), vocab, eos) for r in batch]
    gen = [[] for _ in batch]
    state = model.init_state(len(batch), dev)
    for t in range(max(len(p) for p in prompts) + max_new):
        if all(rd.end for rd in readers):
            break
        t0 = time.time()
        feed = [p[t] if t < len(p) else (g[-1] if g else eos) for p, g in zip(prompts, gen)]
        with torch.no_grad():                              # float32 (TF32 matmuls): the decode state's rings are float32
            logits, state = model.step(torch.tensor(feed, device=dev), state)
        logits = logits.float()
        t1 = time.time()
        for i, (p, rd) in enumerate(zip(prompts, readers)):
            if rd.end or t < len(p) - 1:
                continue
            ids = rd.allowed()
            if not ids:                                    # `end` is "error": by the machine's construction, never
                continue
            tok = best(logits[i], ids)
            rd.push(tok)
            if tok != eos:
                gen[i].append(tok)
                if len(gen[i]) >= max_new:
                    rd.stop("limit")
        clock["model"] = clock.get("model", 0.0) + t1 - t0
        clock["grammar"] = clock.get("grammar", 0.0) + time.time() - t1
    return [{"id": r["id"], "split": r["split"], "arm": arm, "answer": "".join(vocab.pieces[t] for t in g),
             "end": rd.end or "limit", "ids": len(g)} for r, g, rd in zip(batch, gen, readers)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("m", "w", "p"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--splits", default="dev,test,test_long")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--own-text", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)),
                    help="arm p: round 2's grammar worker processes (1: checked in this process)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.arm == "p" and a.own_text:
        ap.error("--own-text is arms m and w's; arm p reads under round 2's rules 1 to 9")
    max_new = a.max_new or (1800 if a.arm == "p" else 900)
    import torch
    from tokenizers import Tokenizer
    from train_base import load_base
    from train_talk_torch import load_adapter
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":                                 # no bf16 autocast: the step writes into float32 rings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    vocab, pool = (T.Vocab.of(tk) if a.arm != "p" else None), None
    if a.arm == "p" and a.workers > 1:                     # forked before the model touches the card
        from view_read_pool import GrammarPool
        pool = GrammarPool(min(a.workers, a.batch))
    model = load_base(a.ckpt, str(dev))
    meta = load_adapter(model, a.adapter)
    rows = [json.loads(ln) for ln in open(a.rows, encoding="utf-8")]
    todo = [r for s in a.splits.split(",") for r in [x for x in rows if x["split"] == s][:a.n]]
    done = {json.loads(ln)["id"] for ln in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    todo = sorted((r for r in todo if r["id"] not in done), key=lambda r: len(r["instruction"]))   # like lengths together
    print(f"arm {a.arm}{' (own text)' if a.own_text else ''}: {len(todo)} rows to read ({len(done)} already), "
          f"adapter r={meta['rank']} on {dev}, at most {max_new} ids an answer", flush=True)
    t0, clock, ends = time.time(), {}, Counter()
    with open(a.out, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch):
            part, tb = todo[b:b + a.batch], time.time()
            if a.arm == "p":
                import exp_view_reader_torch as P
                recs = [dict(rec, arm="p") for rec in P.decode(model, tk, eos, part, "typed", max_new, dev, pool, clock, skip_empty=True)]
            else:
                recs = decode(model, tk, vocab, eos, part, a.arm, a.own_text, max_new, dev, clock)
            for rec in recs:
                rec.update(own_text=a.own_text, seconds=round((time.time() - tb) / len(part), 2))
                ends[f"{rec['split']} {rec['end']}"] += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(b + a.batch, len(todo))}/{len(todo)} rows, {time.time() - t0:.0f} s "
                  f"(model {clock.get('model', 0):.0f} s, grammar {clock.get('grammar', 0):.0f} s)", flush=True)
    if pool:
        pool.close()
    print("ended:", dict(sorted(ends.items())), flush=True)          # per split: finished, error, limit
    print("read:", a.out, flush=True)


if __name__ == "__main__":
    main()
