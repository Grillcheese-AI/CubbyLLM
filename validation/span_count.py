"""validation/span_count.py -- order 0051 A: what share of the training mix's tokens would fold into fact-bearing spans.

H-K10: a span of n pieces as one input unit, ``sum_i e(piece_i) (x) S(i)``, at most ~55 pieces at d = 1,024 (the
capacity law); the boundary from the host; "language words stay words". Counted here on token windows (no training):
  names    the longest exact matches in R1b's names segment (``cubelang knowledge batch``), word-aligned, leftmost
           first, at most MAX_WORDS words and CAP pieces, no overlaps. Two readings, both reported:
           (a) every match: an upper bound (common words are some item's alias: "the", "is", years);
           (b) proper names: a match whose first and last words are capitalised as written (the general sign of a
               name rather than a language word; no word list).
  folded   pieces inside spans of two or more pieces (a one-piece span saves nothing)
  saved    positions saved, sum(pieces - 1): the positions left per window, and the loss positions the trunk would no
           longer predict (inside a span nothing is predicted; the price, by design, is that it stops spelling names)
Sources: the 13 held-out tails (``_holdout.json``) and a sample of 0045's slice (evenly spaced windows).

    python validation/span_count.py [--windows 4 --len 1024] [--slice $CUBBY_DRIVE/cubbyllm/pod/inputs/corpus]
-> validation/logs/span_count_<date>.json

Wired: STANDALONE (an experiment; 0051's report reads it).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

CAP, MAX_WORDS = 55, 8


def word_starts(pieces: list[str]) -> list[bool]:
    """A piece starts a word when it is the first, begins with whitespace, or follows one ending in whitespace."""
    out = []
    for i, t in enumerate(pieces):
        out.append(i == 0 or t[:1].isspace() or pieces[i - 1][-1:].isspace() or not t[:1].isalnum()
                   or not pieces[i - 1][-1:].isalnum())
    return out


def candidates(pieces: list[str], starts: list[bool]) -> list[tuple[int, int, str]]:
    """(i, j, text) for every word-aligned run of 1..MAX_WORDS words and at most CAP pieces, starting and ending on a
    letter or digit (punctuation is never folded into a name)."""
    bounds = [i for i, s in enumerate(starts) if s] + [len(pieces)]
    out = []
    for a in range(len(bounds) - 1):
        for b in range(a + 1, min(a + 1 + MAX_WORDS, len(bounds))):
            i, j = bounds[a], bounds[b]
            if j - i > CAP:
                break
            if not pieces[i].strip() or not pieces[j - 1].strip():   # a span starts and ends on a real piece
                continue
            text = "".join(pieces[i:j]).strip()
            if text and text[0].isalnum() and text[-1].isalnum():     # a name starts and ends on a letter or digit
                out.append((i, j, text))
    return out


def proper(text: str, initial: bool = False) -> bool:
    """First and last words capitalised as written; a single word opening a sentence or a line is no sign of a name
    (every sentence's first word is capitalised), so it needs a second word."""
    words = text.split()
    if not words or not (words[0][:1].isupper() and words[-1][:1].isupper()):
        return False
    return len(words) >= 2 or not initial


def initial_at(pieces: list[str], i: int) -> bool:
    """Whether piece i opens a sentence or a line: nothing before it, a line break, or . ! ? : before it."""
    k = i - 1
    while k >= 0 and not pieces[k].strip():
        if "\n" in pieces[k]:
            return True
        k -= 1
    if k < 0 or "\n" in pieces[i][:len(pieces[i]) - len(pieces[i].lstrip())]:
        return True
    return pieces[k].rstrip()[-1:] in ".!?:" or "\n" in pieces[k][len(pieces[k].rstrip()):]


def spans(cands: list[tuple[int, int, str]], known) -> list[tuple[int, int]]:
    """Leftmost-longest, no overlaps: scanning from the left, at each start the longest known run, then past it."""
    by_start: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for i, j, t in cands:
        by_start[i].append((j, t))
    out, pos = [], 0
    for i in sorted(by_start):
        if i < pos:
            continue
        best = max((j for j, t in by_start[i] if known(t)), default=None)
        if best is not None:
            out.append((i, best))
            pos = best
    return out


def fold(text: str) -> str:
    from retrieval_sets.names_segment import fold as f
    return f(text)


def windows_of(cache: str, name: str, n: int, length: int, tail: bool) -> np.ndarray:
    arr = np.memmap(os.path.join(cache, f"{name}.u32"), dtype=np.uint32, mode="r")
    hp = os.path.join(cache, "_holdout.json")
    hold = int(json.load(open(hp, encoding="utf-8")).get(name, 0)) if os.path.exists(hp) else 0
    lo, hi = (len(arr) - hold, len(arr)) if tail else (0, len(arr) - hold)
    span = hi - lo - length
    starts = [lo + (span * i) // max(n - 1, 1) for i in range(n)] if span > 0 else [lo]
    return np.stack([np.array(arr[s:s + length], dtype=np.int64) for s in starts])


def count(groups: dict, tk, known_keys: set | None = None) -> dict:
    """{source: {reading: numbers}}; `known_keys` given: the segment's answer already in hand (tests, reruns)."""
    from retrieval_sets.names_segment import exact_items
    prepared = {}
    for src, wins in groups.items():
        prepared[src] = []
        for w in wins:
            pieces = [tk.decode([int(t)]) for t in w]
            prepared[src].append((pieces, candidates(pieces, word_starts(pieces))))
    if known_keys is None:
        keys = sorted({fold(t) for v in prepared.values() for _, cs in v for _, _, t in cs} - {""})
        known_keys = {k for k, items in exact_items(keys).items() if items}
    out = {}
    for src, v in prepared.items():
        res = {}
        for reading in ("a_every_match", "b_proper_names"):
            toks = folded = saved = n_sp = 0
            for pieces, cs in v:
                if reading == "a_every_match":
                    use = cs
                else:
                    use = [(i, j, t) for i, j, t in cs if proper(t, initial_at(pieces, i))]
                sp = [(i, j) for i, j in spans(use, lambda t: fold(t) in known_keys) if j - i >= 2]
                toks += len(pieces)
                n_sp += len(sp)
                folded += sum(j - i for i, j in sp)
                saved += sum(j - i - 1 for i, j in sp)
            res[reading] = {"tokens": toks, "spans": n_sp, "folded": folded, "saved": saved,
                            "folded_share": round(folded / max(1, toks), 4),
                            "saved_share": round(saved / max(1, toks), 4),
                            "positions_left_per_1024": round(1024 * (1 - saved / max(1, toks)), 1),
                            "unpredicted_share": round(saved / max(1, toks), 4)}
        out[src] = res
    return out


def mix_weights(cache: str) -> dict:
    """The stable phase's mix (``train_base.Corpus``): a shard's size times its QUALITY multiplier. QUALITY is read from
    train_base.py's source (importing it would import torch)."""
    import ast
    src = open(os.path.join(HERE, "train_base.py"), encoding="utf-8").read()
    quality = next(ast.literal_eval(n.value) for n in ast.parse(src).body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "QUALITY")
    sp = os.path.join(cache, "_sizes.json")
    sizes = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    return {n: float(sizes.get(n, 0)) * quality.get(n, 1.0) for n in quality}


def main() -> None:
    from tokenizers import Tokenizer

    from cubbyllm.local_paths import expand
    from head_dump import SHARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="$CUBBY_TRAINING_DATA/token_cache_cont")
    ap.add_argument("--slice", default="$CUBBY_DRIVE/cubbyllm/pod/inputs/corpus")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--len", type=int, default=1024)
    a = ap.parse_args()
    cache, sl = expand(a.cache), expand(a.slice)
    tk = Tokenizer.from_file(os.path.join(cache, "tokenizer", "grillcheese_bbpe128k.json"))
    groups = {f"heldout/{n}": windows_of(cache, n, a.windows, a.len, tail=True) for n in SHARDS}
    for n in SHARDS:
        if os.path.exists(os.path.join(sl, f"{n}.u32")):
            groups[f"slice0045/{n}"] = windows_of(sl, n, a.windows, a.len, tail=False)
    res = count(groups, tk)
    whole, mixes = {}, {"heldout": mix_weights(cache), "slice0045": mix_weights(sl)}
    for part in ("heldout", "slice0045"):
        rows = {k: v for k, v in res.items() if k.startswith(part + "/")}
        if not rows:
            continue
        whole[part] = {}
        for reading in ("a_every_match", "b_proper_names"):
            t = sum(r[reading]["tokens"] for r in rows.values())
            sv = sum(r[reading]["saved"] for r in rows.values())
            fd = sum(r[reading]["folded"] for r in rows.values())
            w = {k: mixes[part].get(k.split("/", 1)[1], 0.0) for k in rows}
            tw = sum(w.values()) or 1.0
            mix_f = sum(w[k] * rows[k][reading]["folded_share"] for k in rows) / tw
            mix_s = sum(w[k] * rows[k][reading]["saved_share"] for k in rows) / tw
            whole[part][reading] = {"tokens": t, "folded_share": round(fd / t, 4), "saved_share": round(sv / t, 4),
                                    "positions_left_per_1024": round(1024 * (1 - sv / t), 1),
                                    "mix_weighted_folded_share": round(mix_f, 4),
                                    "mix_weighted_saved_share": round(mix_s, 4)}
    rep = {"what": "order 0051 A: the span count (names; CPU)", "date": _dt.date.today().isoformat(),
           "cap": CAP, "max_words": MAX_WORDS, "windows": a.windows, "len": a.len,
           "line_for_C": "at least 10% of the mix's tokens folded", "whole": whole, "per_source": res}
    path = os.path.join(HERE, "logs", f"span_count_{rep['date']}.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(whole, indent=1), "->", path)


if __name__ == "__main__":
    main()
