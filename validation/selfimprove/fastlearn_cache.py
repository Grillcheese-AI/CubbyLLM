"""validation/selfimprove/fastlearn_cache.py -- order 0053's cache, on the GPU (a cell of 0049's notebook, Nick's run).

For 0049's wd/en and wd/fr problems (its stream order, then its held problems; the held hash is checked against
0049's base read), with the base adapter on the frozen trunk:
  program    the greedy program (as 0049 serves it); its HOP1 slot, ``bind frame, HOP1, "``; a problem whose program
             has no HOP1 slot is kept and counted, with no candidates
  state      the trunk's final hidden state (after the adapter, before the head) at the slot: the position whose
             next token is the word's first, from ``features`` over the prompt and the program up to the slot
  candidates the property labels of the problem's language whose property is a gold of some wd train record in
             that language (the task's vocabulary: 137 EN, 159 FR; a choice, recorded), each scored by the trunk as
             log p(prefix + word + '"') - log p(prefix): the word and its closing quote, exact whatever the BPE
             does at the boundary; the top K = 32 kept with their scores
  the rest   the program after the slot's closing quote, so a candidate's program is prefix + word + '"' + rest
             (the rest of a wd program does not depend on the word: ``return recover(frame, SEED)``)
The VM marks each candidate at home (``fastlearn_mark.py``). Writes <out>/fastlearn_cache.jsonl (one problem a line)
and <out>/fastlearn_states.npy (float32, row i = line i).

    python validation/selfimprove/fastlearn_cache.py --records R --ckpt C --tokenizer T --adapter A --props P --out O

Wired: STANDALONE (order 0053's harness).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

SLOT = 'bind frame, HOP1, "'
HELD_SHA = "ad968cc8de05bf69"                       # 0049's held problems (its base read)


def vocabulary(recs: list[dict], props: dict) -> dict:
    """{lang: [(word, pid)]}: the label in that language of every property that is a wd train gold in it."""
    out = {}
    for lang in ("en", "fr"):
        golds = sorted({r["gold"] for r in recs if r["task"] == "wd" and r.get("lang") == lang and r["split"] == "train"})
        out[lang] = [(props[p]["label"][lang], p) for p in golds if p in props and props[p]["label"].get(lang)]
    return out


def seq_logp(trunk, rows: list[tuple[str, str]], batch: int = 32) -> list[float]:
    """Each (prompt, program part)'s summed token log-probability of the program part."""
    import torch
    from talk_data import make_batch
    enc = [{"p": trunk.tk.encode(p).ids, "t": trunk.tk.encode(t).ids} for p, t in rows]
    out = []
    with torch.no_grad():
        for s in range(0, len(enc), batch):
            part = enc[s:s + batch]
            x, y, m, _ = make_batch(part)
            xt, yt, mt = (torch.from_numpy(a).to(trunk.dev) for a in (x, y, m))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trunk.amp):
                h = trunk.model.features(xt)
                lg = trunk.model.logits_from(h)
            lp = torch.log_softmax(lg.float(), -1).gather(-1, yt.unsqueeze(-1))[..., 0]
            out += (lp * mt).sum(1).tolist()
    return out


def state_at(trunk, prompt: str, prefix: str):
    """The final hidden state at the slot (the last prefix position), float32 (d,)."""
    import torch
    ids = trunk.tk.encode(prompt).ids + trunk.tk.encode(prefix).ids
    with torch.no_grad():
        h = trunk.model.features(torch.tensor([ids], device=trunk.dev))
    return h[0, -1].float().cpu().numpy()


def main(argv=None) -> None:
    import numpy as np
    from selfimprove import stream as S
    from selfimprove.trunk import Trunk
    ap = argparse.ArgumentParser()
    for k in ("records", "ckpt", "tokenizer", "adapter", "props", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="a smoke: the first N problems of each language (0: all)")
    a = ap.parse_args(argv)
    t0 = time.time()
    recs = [json.loads(x) for x in open(a.records, encoding="utf-8") if x.strip()]
    props = json.load(open(a.props, encoding="utf-8"))["properties"]
    _, stream, held = S.build(recs, 10, "arithmetic/steps=2", 300, 50, 500, a.seed)     # 0049's second T1's stream
    sha = hashlib.sha256("\n".join(sorted(r["id"] for r in held)).encode()).hexdigest()[:16]
    if sha != HELD_SHA:
        raise SystemExit(f"the held problems' hash {sha} is not 0049's {HELD_SHA}: the records or the seed changed")
    probs = [dict(r, held=False) for r in stream if r["family"] in ("wd/en", "wd/fr")] + \
            [dict(r, held=True) for r in held if r["family"] in ("wd/en", "wd/fr")]
    if a.limit:
        probs = [r for fam in ("wd/en", "wd/fr") for r in [x for x in probs if x["family"] == fam][:a.limit]]
    vocab = vocabulary(recs, props)
    trunk = Trunk(a.ckpt, a.tokenizer, a.adapter, seed=a.seed)
    print(f"{len(probs)} problems; vocabulary EN {len(vocab['en'])}, FR {len(vocab['fr'])}; "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    progs = trunk.programs([r["prompt"] for r in probs], [160] * len(probs))
    os.makedirs(a.out, exist_ok=True)
    states, n_slot = [], 0
    with open(os.path.join(a.out, "fastlearn_cache.jsonl"), "w", encoding="utf-8") as f:
        for i, (r, p) in enumerate(zip(probs, progs)):
            prog = p[0]
            rec = {"id": r["id"], "family": r["family"], "lang": r.get("lang") or "en", "held": r["held"],
                   "gold": r["gold"], "order": i, "prompt": r["prompt"], "spans": r["spans"],
                   "question": r.get("question", ""), "greedy": prog}
            j = prog.find(SLOT)
            if j < 0:
                states.append(np.zeros(trunk.model.config.d_model, dtype=np.float32))
                f.write(json.dumps({**rec, "slot": False}, ensure_ascii=False) + "\n")
                continue
            n_slot += 1
            prefix = prog[:j + len(SLOT)]
            q = prog.find('"', j + len(SLOT))
            rest = prog[q + 1:] if q >= 0 else ""
            words = vocab[rec["lang"]]
            base = seq_logp(trunk, [(r["prompt"], prefix)])[0]
            sc = seq_logp(trunk, [(r["prompt"], prefix + w + '"') for w, _ in words])
            top = sorted(range(len(words)), key=lambda k: -sc[k])[:a.k]
            states.append(state_at(trunk, r["prompt"], prefix))
            f.write(json.dumps({**rec, "slot": True, "prefix": prefix, "rest": rest,
                                "greedy_word": prog[j + len(SLOT):q] if q >= 0 else None,
                                "words": [words[k][0] for k in top], "word_pid": [words[k][1] for k in top],
                                "scores": [round(sc[k] - base, 5) for k in top]}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(probs)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(a.out, "fastlearn_states.npy"), np.stack(states))
    meta = {"problems": len(probs), "with_slot": n_slot, "vocab": {k: len(v) for k, v in vocab.items()},
            "k": a.k, "held_sha": sha, "minutes": round((time.time() - t0) / 60, 1),
            "families": {fam + ("/held" if h else "/stream"): sum(1 for r in probs if r["family"] == fam and r["held"] == h)
                         for fam in ("wd/en", "wd/fr") for h in (False, True)}}
    json.dump(meta, open(os.path.join(a.out, "fastlearn_cache_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
