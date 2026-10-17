"""make_spatial_shard -- villages the trunk is shown as places, with questions about them: the
token shard that trains the spatial link (H-S12, gates 1 and 2).

Each document is one village. Its things are listed along the Hilbert curve, and every token
of a thing's name carries the thing's cell in the ``.pos.i16`` side file (``NO_PLACE`` on every
other token), so the trunk sees the village *as places* -- bound into the embeddings by
``cubbyllm.model.spatial`` -- and not as text. Then questions in the ask loop's own shapes,
each answered from ``Scene``'s exact reading, never written by hand:

    Question: What is the position of forge?            Answer: 10 9 0
    Question: What is the east neighbour of well?       Answer: oak
    Question: What is the nearest building of well?     Answer: forge
    Question: What is 3 east of well?                   Answer: hut | nothing
    Question: What is at here?                          Answer: forge   (``here`` is a placed token:
                                                                          the question itself is a place)

``--mode text`` writes the control: the same villages and questions with each thing's cell
spelled out in text ("forge at 10 9 0") and no places at all, for the arm without the link.

Documents are padded with </s> to ``--doc`` tokens and the shard is listed in ``_align.json``,
so ``train_base.Corpus`` never cuts a village in two. The held-out tail, ``_holdout.json``
and ``_sizes.json`` follow ``make_decay_shards.py``.

    python validation/make_spatial_shard.py --cache <token_cache> [--villages 20000] [--doc 512]
        [--mode places|text] [--tokenizer <bbpe128k tokenizer.json>] [--seed 0]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from cubbyllm.spatial.scene import DIRECTIONS, Scene  # noqa: E402

MODELS = os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS")
KINDS = {"water": ["well", "pond", "spring"], "tree": ["oak", "pine", "birch", "apple tree"],
         "building": ["hut", "forge", "barn", "mill", "church", "shop", "house", "tower", "inn"],
         "light": ["lantern", "torch", "lamp"], "way": ["road", "bridge", "gate", "fence", "wall"],
         "stone": ["stone", "boulder", "rock"]}
REGIONS = ["the square", "the orchard", "the yard", "the green"]


def village(rng: random.Random, n: int, side: int = 24, height: int = 3) -> Scene:
    """``n`` things at distinct cells of a ``side x side x height`` box (most on the floor),
    with one or two named regions."""
    sc = Scene()
    cells: set = set()
    while len(cells) < n:
        z = 0 if rng.random() < 0.8 else rng.randrange(1, height)
        cells.add((rng.randrange(side), rng.randrange(side), z))
    used: dict = {}
    for c in sorted(cells):
        kind = rng.choice(list(KINDS))
        base = rng.choice(KINDS[kind])
        used[base] = used.get(base, 0) + 1
        name = base if used[base] == 1 else f"{base} {used[base]}"
        sc.place(name, c, kind)
    for name in rng.sample(REGIONS, rng.randint(1, 2)):
        x, y = rng.randrange(side - 6), rng.randrange(side - 6)
        sc.region(name, (x, y, 0), (x + rng.randint(3, 6), y + rng.randint(3, 6), height - 1))
    return sc


def questions(sc: Scene, rng: random.Random, k: int) -> list:
    """``k`` (question, answer, place)`` triples from the exact reading; ``place`` is the cell a
    ``here`` token in the question carries, or None."""
    names = sorted(sc.things)
    out = []
    for _ in range(k):
        name = rng.choice(names)
        c = sc.where(name)
        kind = rng.choice(["position", "neighbour", "nearest", "offset", "at", "region"])
        if kind == "position":
            out.append((f"What is the position of {name}?", " ".join(map(str, c)), None))
        elif kind == "neighbour":
            d = rng.choice(list(DIRECTIONS))
            out.append((f"What is the {d} neighbour of {name}?", sc.adjacent(name).get(d, "nothing"), None))
        elif kind == "nearest":
            want = rng.choice(sorted({k for k in sc.kinds.values() if k}))
            out.append((f"What is the nearest {want} of {name}?", sc.nearest(name, want) or "nothing", None))
        elif kind == "offset":
            d, n = rng.choice(list(DIRECTIONS)), rng.randint(1, 4)
            out.append((f"What is {n} {d} of {name}?", sc.at(sc.offset(name, d, n)) or "nothing", None))
        elif kind == "region":
            rs = sc.regions_of(name)
            out.append((f"What is the region of {name}?", rs[0] if rs else "nothing", None))
        else:
            lo, hi = _box(sc)
            cell = c if rng.random() < 0.7 else tuple(int(rng.randint(int(lo[i]), int(hi[i]))) for i in range(3))
            out.append(("What is at here?", sc.at(cell) or "nothing", cell))
    return out


def render(tk, sc: Scene, qa: list, mode: str, eos: int, doc: int):
    """One document: ``(ids, places)`` of exactly ``doc`` tokens (questions dropped from the
    end when the village alone would not fit them; </s>-padded otherwise). In ``places`` mode
    a thing's name tokens carry its cell and a question's ``here`` carries the asked cell;
    in ``text`` mode cells are words and nothing is placed."""
    ids, pos = [], []

    def put(text: str, cell=None):
        t = tk.encode(text, add_special_tokens=False).ids
        ids.extend(t)
        pos.extend([cell if (cell is not None and mode == "places") else None] * len(t))

    put("Village:\n")
    for cell, name in sc.serialize(*_box(sc)):
        if mode == "text":
            put(f"{name} at {' '.join(map(str, cell))}\n")
        else:
            put(name, cell)
            put("\n")
    for r, (lo, hi) in sorted(sc.regions.items()):
        put(f"Region {r}: {' '.join(map(str, lo))} to {' '.join(map(str, hi))}\n")
    head = len(ids)
    for q, a, cell in qa:
        start = len(ids)
        if cell is not None:
            put("Question: What is at ")
            put("here" if mode == "places" else " ".join(map(str, cell)), cell)
            put("?\n")
        else:
            put(f"Question: {q}\n")
        put(f"Answer: {a}\n")
        if len(ids) + 1 > doc:
            del ids[start:], pos[start:]
            break
    if head + 1 > doc:
        raise ValueError(f"a village of {len(sc.things)} things needs {head + 1} tokens, --doc is {doc}")
    ids.append(eos)
    pos.append(None)
    ids.extend([eos] * (doc - len(ids)))
    pos.extend([None] * (doc - len(pos)))
    places = np.full((doc, 3), NO_PLACE, dtype=np.int16)
    for i, c in enumerate(pos):
        if c is not None:
            places[i] = c
    return np.asarray(ids, dtype=np.uint32), places


def _box(sc: Scene):
    cs = np.array(list(sc.cells))
    return cs.min(axis=0), cs.max(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(MODELS, "base450m", "grilly", "tokenizer.json"))
    ap.add_argument("--villages", type=int, default=20_000)
    ap.add_argument("--doc", type=int, default=512, help="tokens per document; divides CB_S")
    ap.add_argument("--things", type=int, nargs=2, default=(8, 32))
    ap.add_argument("--questions", type=int, nargs=2, default=(6, 12))
    ap.add_argument("--mode", choices=["places", "text"], default="places")
    ap.add_argument("--name", default="", help="shard name (default: spatial | spatial_text)")
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    name = args.name or ("spatial" if args.mode == "places" else "spatial_text")
    rng = random.Random(args.seed)
    t0 = time.time()
    ids_path, pos_path = os.path.join(args.cache, f"{name}.u32"), os.path.join(args.cache, f"{name}.pos.i16")
    n_q = n_placed = 0
    with open(ids_path + ".part", "wb") as fi, open(pos_path + ".part", "wb") as fp:
        for v in range(args.villages):
            sc = village(rng, rng.randint(*args.things))
            qa = questions(sc, rng, rng.randint(*args.questions))
            ids, places = render(tk, sc, qa, args.mode, eos, args.doc)
            fi.write(ids.tobytes())
            fp.write(places.tobytes())
            n_q += len(qa)
            n_placed += int((places[:, 0] != NO_PLACE).sum())
            if (v + 1) % 5000 == 0:
                print(f"  {v + 1:,} villages, {time.time() - t0:.0f}s", flush=True)
    os.replace(ids_path + ".part", ids_path)
    os.replace(pos_path + ".part", pos_path)
    total = args.villages * args.doc
    hold_docs = max(1, int(args.villages * args.holdout_frac))
    hold = hold_docs * args.doc
    for fn, val in (("_holdout.json", hold), ("_sizes.json", total - hold), ("_align.json", args.doc)):
        p = os.path.join(args.cache, fn)
        d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        d[name] = int(val)
        json.dump(d, open(p, "w", encoding="utf-8"), indent=1)
    rep = {"args": vars(args), "shard": name, "villages": args.villages, "tokens": total, "train_tokens": total - hold,
           "holdout_tokens": hold, "questions": n_q, "placed_tokens": n_placed, "seconds": round(time.time() - t0, 1)}
    logp = os.path.join(HERE, "logs", f"make_spatial_shard_{name}_{_dt.date.today().isoformat()}.json")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    json.dump(rep, open(logp, "w", encoding="utf-8"), indent=1)
    print(f"{name}.u32: {args.villages:,} villages x {args.doc} tokens, {n_q:,} questions, "
          f"{n_placed:,} placed tokens, {hold_docs} villages held out; wrote {logp}")


if __name__ == "__main__":
    main()
