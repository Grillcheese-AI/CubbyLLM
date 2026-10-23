"""make_scene_shard -- order 0035 C: a token shard of villages from scene files (``cubby.scene/1``, a world's own
exporter's), in the far facts' shape (``make_spatial_shard.render(lead=True)``) or 0034's, for the copy to train on.

Each village, cut as 0034 reads it (``exp_s12_scene_files.views``), gives ``--per-village`` documents of ``--doc``
tokens. Each document has its own draw of placed questions (neighbour, offset, at: the shapes run 5 was trained on),
as many as fit. They alternate between a cell that holds something (a villager or another thing) and an empty cell, so
half the answers are "nothing" and the gate learns to shut there. A village whose listing alone does not fit is
skipped and counted.

The files go to ``--cache``: ``<name>.u32``, ``.pos.i16`` and ``.role.u8``, with the cache's ``_sizes``, ``_align``
and ``_holdout`` entries. The cache is outside this repo when the scenes are private. The held-out tail is the last
villages' documents. This repo's log gets only the counts (``validation/logs/make_scene_shard_<name>_<date>.json``).

    python validation/make_scene_shard.py --scenes <folder> --cache <folder> --name <shard>
        [--per-village 10] [--doc 512] [--shape lead|0034] [--holdout-frac 0.02] [--seed 0]

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_s12_scene_files import SPM, files, views  # noqa: E402
from make_spatial_shard import render  # noqa: E402
from s12_questions import PLACED, VILLAGER, _box, _draw  # noqa: E402

__wiring__ = "STANDALONE"
DRAWS = 240                                              # placed questions drawn per document before the balance


def asked(sc, rng: random.Random) -> list[tuple]:
    """``(question, answer, cell, class)`` for one document: placed draws, deduplicated, alternating a cell that holds
    something and an empty one (from a random side), each side shuffled; among the things, villagers and other things
    alternate while both last (0034's draws give villagers half the occupied picks when one is in reach)."""
    names = sorted(sc.things)
    people = [n for n in names if sc.kinds.get(n) == VILLAGER]
    kinds = sorted({k for k in sc.kinds.values() if k})
    lo, hi = _box(sc)
    seen, full, empty = set(), [], []
    for _ in range(DRAWS):
        q = _draw(sc, rng, rng.choice(PLACED), names, people, kinds, lo, hi)
        text = "What is at here?" if q["shape"] == "at" else q["question"]
        key = (text, tuple(q["address"]))
        if key in seen:
            continue
        seen.add(key)
        (empty if q["cls"] == "nothing" else full).append((text, q["answer"], tuple(q["address"]), q["cls"]))
    rng.shuffle(full)
    rng.shuffle(empty)
    named, other = [q for q in full if q[3] == "name"], [q for q in full if q[3] != "name"]
    full = [q for pair in zip(named, other) for q in pair] + named[len(other):] + other[len(named):]
    sides = [full, empty] if rng.random() < 0.5 else [empty, full]   # villagers half the things, as 0034 draws them
    out = []
    for a, b in zip(*sides):
        out += [a, b]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", required=True, help="the folder of scene files")
    ap.add_argument("--cache", required=True, help="the token cache folder the shard is written into")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default=SPM)
    ap.add_argument("--per-village", type=int, default=10)
    ap.add_argument("--doc", type=int, default=512)
    ap.add_argument("--shape", choices=["lead", "0034"], default="lead")
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from exp_hypertoken_recall import answers
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    rng = random.Random(a.seed)
    t0 = time.time()
    os.makedirs(a.cache, exist_ok=True)
    paths = {ext: os.path.join(a.cache, f"{a.name}{ext}") for ext in (".u32", ".pos.i16", ".role.u8")}
    outs = {ext: open(p + ".part", "wb") for ext, p in paths.items()}
    docs, skipped, by_cls, villages = 0, 0, collections.Counter(), files(a.scenes)
    for f in villages:
        sc = views(os.path.join(a.scenes, f))["cut"]
        for _ in range(a.per_village):
            qa = asked(sc, rng)
            try:
                ids, places, roles = render(tk, sc, [q[:3] for q in qa], "places", eos, a.doc, a.shape == "lead")
            except ValueError:                           # the listing alone is longer than a document
                skipped += 1
                break
            for q, _span in zip(qa, answers(tk, ids)):   # the questions that fit, by class
                by_cls[q[3]] += 1
            for ext, arr in zip((".u32", ".pos.i16", ".role.u8"), (ids, places, roles)):
                outs[ext].write(arr.tobytes())
            docs += 1
    for ext, fh in outs.items():
        fh.close()
        os.replace(paths[ext] + ".part", paths[ext])
    hold = max(1, int(docs * a.holdout_frac)) * a.doc
    total = docs * a.doc
    for fn, val in (("_holdout.json", hold), ("_sizes.json", total - hold), ("_align.json", a.doc)):
        p = os.path.join(a.cache, fn)
        d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        d[a.name] = int(val)
        json.dump(d, open(p, "w", encoding="utf-8"), indent=1)
    rep = {"what": "order 0035 C: a village shard from scene files", "shard": a.name, "shape": a.shape,
           "villages": len(villages), "villages_skipped": skipped, "documents": docs, "doc": a.doc,
           "tokens": total, "holdout_tokens": hold, "questions": dict(by_cls), "per_village": a.per_village,
           "seed": a.seed, "seconds": round(time.time() - t0, 1)}
    logp = os.path.join(HERE, "logs", f"make_scene_shard_{a.name}_{_dt.date.today().isoformat()}.json")
    json.dump(rep, open(logp, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep), "->", logp)


if __name__ == "__main__":
    main()
