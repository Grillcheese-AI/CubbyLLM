"""make_fact_shard -- wiki-world facts as loci tokens: the token shard that trains the fact link
(H-K10 gate 4, H-K7 gate 5) on the same channel as the spatial link.

A fact "o is the r of s" has an *address*: the subject's cell (a hash of its name into a
1,024^3 lattice, offset from the villages' box) plus the relation's step (a lattice step the
kernel separates, 6 cells; 2,902 relations on a 54 x 54 grid of steps in x, y). A name is a
span: its i-th token sits 6 i cells up the z axis from the address -- a bag with order, read
piece by piece. Each document is 8-16 facts of the wiki world
(``standin/data/out/wikikg/triplets.parquet``, every fact used once, so a held-out document's
facts are never seen in training), each rendered as a sentence whose OBJECT tokens carry the
fact's address (piece i at +6 i in z) in the ``.pos.i16`` side file, then questions in the ask
loop's own shape: ``here`` carries the address (it asks for piece 0), and each answer piece, as
input, carries the address of the next piece (the last asks for one that is not there: an empty
read means the answer is over):

    Keiko Matsuzaka works for Daiei.                    (Daiei's tokens at cell(Keiko Matsuzaka) + step(WORKS_FOR) + 6 i z)
    Question: What is the works for of Keiko Matsuzaka? here        -> Answer: Daiei

so with ``SpatialLink(query="placed")`` the trunk reads the scene at the fact's address and the
route is the host's (the entity linker: name -> cell, relation -> step), exactly as the ask
loop's ``route()`` picks the scene for a spatial question. ``--mode text`` is the control: the
same documents, no ``here``, nothing placed -- attention has to find the answer in the text.

``--far`` (the memory's shard, H-E20 gate 2): the document is filled with real corpus text between
the facts and the questions (windows of the cache's own shards, as the needle probe takes them),
so at ``--doc 2048`` the questions sit ~1,500 tokens past the facts -- beyond the 512 window, where
only a read of the past can answer. In ``text`` mode that read is the episodic memory's: the
training signal it never had (``facts_far``).

Documents are padded with </s> to ``--doc`` tokens and listed in ``_align.json``; the held-out
tail, ``_holdout.json`` and ``_sizes.json`` follow ``make_spatial_shard.py``.

``--absent q`` (order 0022 C; 0 = off, the shard byte for byte as before): a question is, with probability ``q``, about
a fact the document does not hold -- one from the far end of the shuffled facts, which no document writes -- asked in
the same shape (its address is empty: the read finds nothing). ``<name>.absent.json`` lists, per document, which of its
questions are absent. ``--skip-facts n`` starts the documents after the first ``n`` shuffled facts (a probe shard whose
facts no earlier shard of the same seed used).

    python validation/make_fact_shard.py --cache <token_cache> [--docs 20000] [--doc 512]
        [--mode places|text] [--names v1|v2] [--tokenizer <bbpe128k tokenizer.json>] [--seed 0]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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
from fact_names import doubled, question, words  # noqa: E402

MODELS = os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS")
WIKI = os.path.join(ROOT, "standin", "data", "out", "wikikg", "triplets.parquet")
ORIGIN, SIDE, STEP, GRID = 4096, 1024, 6, 54      # the lattice of entity cells, the step grid of relations (x, y)


def cell(entity: str) -> tuple:
    """The entity's cell: a hash into the lattice, the same for every document and every run."""
    h = hashlib.blake2b(entity.encode("utf-8"), digest_size=6).digest()
    return tuple(ORIGIN + int.from_bytes(h[2 * i:2 * i + 2], "little") % SIDE for i in range(3))


def step(rel_index: int) -> tuple:
    return (STEP * (rel_index % GRID), STEP * (rel_index // GRID), 0)


def address(subject: str, rel_index: int, piece: int = 0) -> tuple:
    """The fact's address, and the i-th piece of its object's name ``6 i`` cells up z."""
    c, d = cell(subject), step(rel_index)
    return (c[0] + d[0], c[1] + d[1], c[2] + d[2] + STEP * piece)


def hr_answers(tk, ids):
    """The answer spans the probe finds in a rendered document (``exp_hypertoken_recall.answers``)."""
    sys.path.insert(0, HERE)
    from exp_hypertoken_recall import answers
    return answers(tk, ids)


def load(rng: random.Random):
    import pyarrow.parquet as pq
    t = pq.read_table(WIKI, columns=["subject", "relation", "object"])
    tri = sorted(set(zip(t.column("subject").to_pylist(), t.column("relation").to_pylist(), t.column("object").to_pylist())))
    rels = {r: i for i, r in enumerate(sorted({r for _s, r, _o in tri}))}
    rng.shuffle(tri)
    return tri, rels


def document(tri: list, rels: dict, rng: random.Random, n_ask: int, v2: bool = False, absent: float = 0.0,
             reserve=None) -> list:
    """``(question, answer, address)`` for ``n_ask`` of the document's facts, plus the facts. With ``absent``, a
    question is replaced, at that rate, by one about ``reserve``'s next fact (one the document does not hold); the
    fourth element then marks it."""
    asked = rng.sample(tri, min(n_ask, len(tri)))
    if absent <= 0:
        return [(question(s, r, v2), words(o, v2=v2), address(s, rels[r])) for s, r, o in asked]
    out = []
    for s, r, o in asked:
        gone = rng.random() < absent
        if gone:
            s, r, o = next(reserve)
        out.append((question(s, r, v2), words(o, v2=v2), address(s, rels[r]), gone))
    return out


class Filler:
    """Random windows of real text from the cache's ``.u32`` shards (the needle probe's filler)."""

    def __init__(self, cache: str, names: str, seed: int) -> None:
        self.arrays = [np.memmap(os.path.join(cache, f"{n}.u32"), dtype=np.uint32, mode="r")
                       for n in names.split(",") if os.path.exists(os.path.join(cache, f"{n}.u32"))]
        if not self.arrays:
            raise SystemExit(f"no filler shard among {names} in {cache}")
        self.rng = np.random.default_rng(seed)

    def take(self, n: int) -> list:
        arr = self.arrays[self.rng.integers(len(self.arrays))]
        i = int(self.rng.integers(0, len(arr) - n - 1))
        return [int(t) for t in arr[i:i + n]]


WRITE, ASK = 1, 2                                   # role flags (train_base.Tokens): into the scene / reads there


def render(tk, facts: list, rels: dict, qa: list, mode: str, eos: int, doc: int, filler=None, lead: bool = False,
           v2: bool = False):
    """One document of exactly ``doc`` tokens, ``(ids, places, roles)``: the facts as sentences
    (an object's i-th token at the fact's address + 6 i z in ``places`` mode, WRITING into the
    scene), then the questions (``here`` at the address, each answer piece at the next piece's,
    all ASKING -- a question reads the scene, it does not write into it; in ``text`` mode no
    ``here`` and nothing placed), questions dropped from the end to fit. With ``filler`` the gap
    between the facts and the questions is real text that fills the document (``--far``). With
    ``lead`` the token right before each answer (the space after ``Answer:``) also asks for the
    first piece, so every token that predicts a piece asks for it (``--lead``; the far facts'
    second 450M run read nothing on the first piece, 2026-10-20). The ids are unchanged. ``v2``: the
    names as ``words(v2=True)`` renders them."""
    ids, pos, roles = [], [], []

    def put(text: str, at=None, span: bool = False):
        t = tk.encode(text, add_special_tokens=False).ids
        ids.extend(t)
        if at is None or mode != "places":
            pos.extend([None] * len(t))
            roles.extend([0] * len(t))
        else:
            pos.extend([(at[0], at[1], at[2] + STEP * (i + 1 if span else 0)) for i in range(len(t))])
            roles.extend([ASK] * len(t))

    put("Facts:\n")
    for s, r, o in facts:
        put(f"{words(s, v2=v2)} {words(r, True, v2)} ")
        a0 = address(s, rels[r])
        for i, tok in enumerate(tk.encode(words(o, v2=v2), add_special_tokens=False).ids):   # piece i at +6 i z
            ids.append(tok)
            pos.append((a0[0], a0[1], a0[2] + STEP * i) if mode == "places" else None)
            roles.append(WRITE if mode == "places" else 0)
        put(".\n")
    head = len(ids)
    if head + 1 > doc:
        raise ValueError(f"{len(facts)} facts need {head + 1} tokens, --doc is {doc}")
    for q, a, at, *_absent in qa:
        start = len(ids)
        put(f"Question: {q}")
        if mode == "places":
            put(" ")
            put("here", at)                       # asks for piece 0
        put("\nAnswer: ")
        if lead and mode == "places":             # the token before the answer asks for piece 0
            pos[-1], roles[-1] = at, ASK
        put(a, at, span=True)                     # piece j, as input, asks for piece j + 1
        put("\n")
        if len(ids) + 1 > doc:
            del ids[start:], pos[start:], roles[start:]
            break
    if filler is not None:                        # --far: real text fills the gap, the questions land at the end
        nl = tk.encode("\n", add_special_tokens=False).ids
        gap = doc - 1 - len(ids) - len(nl)
        ids[head:head] = filler.take(gap) + nl
        pos[head:head] = [None] * (gap + len(nl))
        roles[head:head] = [0] * (gap + len(nl))
    ids.append(eos)
    pos.append(None)
    roles.append(0)
    ids.extend([eos] * (doc - len(ids)))
    pos.extend([None] * (doc - len(pos)))
    roles.extend([0] * (doc - len(roles)))
    places = np.full((doc, 3), NO_PLACE, dtype=np.int16)
    for i, c in enumerate(pos):
        if c is not None:
            places[i] = c
    return np.asarray(ids, dtype=np.uint32), places, np.asarray(roles, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(MODELS, "base450m", "grilly", "tokenizer.json"))
    ap.add_argument("--docs", type=int, default=20_000)
    ap.add_argument("--doc", type=int, default=512, help="tokens per document; divides CB_S")
    ap.add_argument("--facts", type=int, nargs=2, default=(8, 16))
    ap.add_argument("--questions", type=int, nargs=2, default=(4, 8))
    ap.add_argument("--mode", choices=["places", "text"], default="places")
    ap.add_argument("--far", action="store_true", help="real text between the facts and the questions fills the document")
    ap.add_argument("--filler", default="books,wikipedia,fineweb_edu", help="--far: the cache shards the text comes from")
    ap.add_argument("--lead", action="store_true", help="the token right before each answer asks for its first piece")
    ap.add_argument("--names", choices=["v1", "v2"], default="v1",
                    help="v2: names in readable words (event ids split, case kept; no 'of of'); v1: every shard before 0011")
    ap.add_argument("--name", default="", help="shard name (default: facts | facts_text, _far with --far)")
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent", type=float, default=0.0, help="order 0022: the rate of questions about a fact not held")
    ap.add_argument("--skip-facts", type=int, default=0, help="the documents start after this many shuffled facts")
    ap.add_argument("--out", default=None, help="write the shard and its index files here (default: --cache)")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    name = args.name or (("facts" if args.mode == "places" else "facts_text") + ("_far" if args.far else ""))
    rng = random.Random(args.seed)
    t0 = time.time()
    tri, rels = load(rng)
    reserve = iter(tri[::-1])                                      # absent questions: facts no document writes
    used = args.skip_facts
    absent_of = []
    filler = Filler(args.cache, args.filler, args.seed) if args.far else None
    out = args.out or args.cache                                   # --out: the shard beside, the cache only read
    os.makedirs(out, exist_ok=True)
    ids_path, pos_path = os.path.join(out, f"{name}.u32"), os.path.join(out, f"{name}.pos.i16")
    n_q = n_placed = 0
    v2, n_double, n_any, double = args.names == "v2", 0, 0, []
    role_path = os.path.join(out, f"{name}.role.u8")
    with open(ids_path + ".part", "wb") as fi, open(pos_path + ".part", "wb") as fp, open(role_path + ".part", "wb") as fr:
        for d in range(args.docs):
            k = rng.randint(*args.facts)
            facts = tri[used:used + k]
            used += k
            if len(facts) < args.facts[0]:
                raise SystemExit(f"the wiki world ran out of facts after {d:,} documents")
            qa = document(facts, rels, rng, rng.randint(*args.questions), v2, args.absent, reserve)
            ids, places, roles = render(tk, facts, rels, qa, args.mode, eos, args.doc, filler, args.lead, v2)
            if args.absent > 0:                                    # the questions render kept, in order
                kept = len(hr_answers(tk, ids))
                absent_of.append([bool(x[3]) for x in qa][:kept])
                qa = [x for x in qa if not x[3]]
            subj = {address(s, rels[r]): s for s, r, _o in facts}
            for q, _a, at, *_x in qa:                              # a word the template or the relation doubled
                n_any, bad = n_any + bool(doubled(q)), doubled(q, words(subj[at], v2=v2))
                n_double, double = n_double + bool(bad), double + [q] * (bool(bad) and len(double) < 10)
            fr.write(roles.tobytes())
            fi.write(ids.tobytes())
            fp.write(places.tobytes())
            n_q += len(qa)
            n_placed += int((places[:, 0] != NO_PLACE).sum())
            if (d + 1) % 5000 == 0:
                print(f"  {d + 1:,} documents, {time.time() - t0:.0f}s", flush=True)
    os.replace(ids_path + ".part", ids_path)
    if args.mode == "places":
        os.replace(pos_path + ".part", pos_path)
        os.replace(role_path + ".part", role_path)
    else:
        os.remove(pos_path + ".part")
        os.remove(role_path + ".part")
    total = args.docs * args.doc
    hold_docs = max(1, int(args.docs * args.holdout_frac))
    hold = hold_docs * args.doc
    for fn, val in (("_holdout.json", hold), ("_sizes.json", total - hold), ("_align.json", args.doc)):
        p = os.path.join(out, fn)
        dd = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        dd[name] = int(val)
        json.dump(dd, open(p, "w", encoding="utf-8"), indent=1)
    rep = {"args": vars(args), "shard": name, "documents": args.docs, "facts_used": used, "relations": len(rels),
           "tokens": total, "train_tokens": total - hold, "holdout_tokens": hold, "questions": n_q,
           "placed_tokens": n_placed, "questions_with_a_doubled_word": n_double, "doubled_examples": double,
           "questions_with_a_repeat_in_or_at_a_name": n_any,
           "seconds": round(time.time() - t0, 1)}
    if args.absent > 0:
        json.dump(absent_of, open(os.path.join(out, f"{name}.absent.json"), "w", encoding="utf-8"))
        rep["absent_questions"] = sum(map(sum, absent_of))
    rep["args"].update(cache="<token cache>", out=args.out and "<probe folder>", tokenizer=os.path.basename(args.tokenizer))     # no local path in a log
    logp = os.path.join(HERE, "logs", f"make_fact_shard_{name}_{_dt.date.today().isoformat()}.json")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    json.dump(rep, open(logp, "w", encoding="utf-8"), indent=1)
    print(f"{name}.u32: {args.docs:,} documents x {args.doc} tokens, {used:,} facts, {n_q:,} questions, "
          f"{n_placed:,} placed tokens, {hold_docs} documents held out, {n_double} questions with a doubled word; "
          f"wrote {logp}")


if __name__ == "__main__":
    main()
