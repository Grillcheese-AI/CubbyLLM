"""Order 0067, the view reader's read on a CUDA card (Nick, 2026-11-13: "why dont we run the tests on colab might be
faster?"): the answers only, decoded in torch; the VM scores them at home (`exp_view_reader.py --score`).

The local read decodes one instruction at a time on grilly2 (about 25 s an instruction on the local card; the grammar
is 0.3 s of it). Here a batch of instructions decodes in lockstep on `CubbyModel.step` (a fixed-size state a row, the
O(1) decode path): each step feeds every row its next token, its prompt's first, then its own answer's; each row's next
answer token is the best-scored one its grammar allows (`view_grammar.chooser` over the top 32, then 512, then every
id: the local read's `ranked_processor`, the same greedy choice). The same rows as the local read (each split's seeded
sample), the same grammar (``--grammar``). Floating point differs from grilly2's, so a greedy answer can differ.

  python validation/exp_view_reader_torch.py --ckpt <base.pt> --tokenizer <json> --adapter <dir> --rows <rows.jsonl>
      [--splits dev,test,test_long] [--n 100] [--batch 32] [--max-new 1800] [--grammar typed|lines] --out <answers.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from cubbyllm.speech import view_grammar as VG  # noqa: E402
from cubbyllm.speech import view_reader as VR  # noqa: E402
from cubbyllm.speech import view_types as VT  # noqa: E402

__wiring__ = "STANDALONE"
WIDTHS = (32, 512)


def sample(rows: list, split: str, n: int) -> list:
    pool = [r for r in rows if r["split"] == split]
    return random.Random(0).sample(pool, min(n, len(pool)))         # exp_view_reader.py's draw


def decode(model, tk, eos, batch: list, grammar: str, max_new: int, dev) -> list[dict]:
    """One batch of rows -> [{"id", "split", "answer"} (+ "error" when no token fits its grammar)]."""
    import torch
    pieces: dict = {}

    def piece(t):
        if t not in pieces:
            pieces[t] = tk.decode([t])
        return pieces[t]
    prompts = [tk.encode(VR.prompt(r["instruction"])).ids for r in batch]
    gram = [(VT.TypedGrammar if grammar == "typed" else VG.Grammar)(r["instruction"]) for r in batch]
    choose = [VG.chooser(g, piece, eos, len(p)) for g, p in zip(gram, prompts)]
    gen, done, err = [[] for _ in batch], [False] * len(batch), [None] * len(batch)
    state = model.init_state(len(batch), dev)
    for t in range(max(len(p) for p in prompts) + max_new):
        if all(done):
            break
        feed = [p[t] if t < len(p) else (g[-1] if g else eos) for p, g in zip(prompts, gen)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, state = model.step(torch.tensor(feed, device=dev), state)
        logits = logits.float()
        top = {w: torch.topk(logits, min(w, logits.shape[-1]), dim=-1).indices.tolist() for w in WIDTHS}
        for i, (p, g) in enumerate(zip(prompts, gen)):
            if done[i] or t < len(p) - 1:
                continue
            ids, pick = p + g, []
            for w in WIDTHS:
                pick = choose[i](ids, top[w][i])
                if pick:
                    break
            if not pick:
                pick = choose[i](ids, torch.argsort(logits[i], descending=True).tolist())
            if not pick:
                done[i], err[i] = True, "no token keeps the answer in its grammar"
                continue
            if pick[0] == eos or len(g) + 1 >= max_new:
                done[i] = True
            if pick[0] != eos:
                g.append(pick[0])
    out = []
    for r, g, e in zip(batch, gen, err):
        rec = {"id": r["id"], "split": r["split"], "answer": tk.decode(g)}
        if e:
            rec["error"] = e
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--splits", default="dev,test,test_long")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=1800)
    ap.add_argument("--grammar", default="typed", choices=("typed", "lines"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from tokenizers import Tokenizer
    from train_base import load_base
    from train_talk_torch import load_adapter
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    model = load_base(a.ckpt, str(dev))
    meta = load_adapter(model, a.adapter)
    rows = [json.loads(ln) for ln in open(a.rows, encoding="utf-8")]
    todo = [r for s in a.splits.split(",") for r in sample(rows, s, a.n)]
    done_ids = set()
    if os.path.exists(a.out):                              # resumable: rows already answered are kept
        done_ids = {json.loads(ln)["id"] for ln in open(a.out, encoding="utf-8")}
    todo = [r for r in todo if r["id"] not in done_ids]
    todo.sort(key=lambda r: len(r["instruction"]))         # like lengths together: a batch waits on its longest
    print(f"{len(todo)} rows to read ({len(done_ids)} already), adapter r={meta['rank']} on {dev}", flush=True)
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch):
            tb = time.time()
            for rec in decode(model, tk, eos, todo[b:b + a.batch], a.grammar, a.max_new, dev):
                rec["seconds"] = round((time.time() - tb) / len(todo[b:b + a.batch]), 2)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(b + a.batch, len(todo))}/{len(todo)} rows, {time.time() - t0:.0f} s", flush=True)
    print("read:", a.out, flush=True)


if __name__ == "__main__":
    main()
