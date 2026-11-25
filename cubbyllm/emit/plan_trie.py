"""cubbyllm/emit/plan_trie.py -- the programs seat may name only relations the store holds (order 0081).

The host knows every relation its store holds (`KnownRelations`, the set the gate checks a plan against). While the seat
writes a plan's relation literal (``bind frame, HOPk, "...";``) the decoder is held to the labels the store holds, and
while it writes the seed literal (``bind frame, SEED, "...";``) to the slot ids the host put in the prompt (``$E1``,
``$E2``...). The seat still chooses which label; the host only narrows the choice to what exists. Outside those two
literals nothing is constrained. The gate (coverage, the walk, the VM) does not change.

**The token boundary.** bbpe128k merges the opening quote with what follows it (``Ġ"$`` for a slot) and the closing
quote with the semicolon (``";``), and a label ending in punctuation can merge with the closing quote too. So each
option is the tokens the tokenizer writes *in context*: ``encode(CTX + ', "label";')`` minus ``encode(CTX)``, with
``CTX`` the role just written (``bind frame, HOP1``); an option whose tokens do not split cleanly there is left out
(counted in ``skipped``). The constraint starts right after the role name (the comma is part of the option) and ends
with the option's last token (its ``";``).

``CB_EMIT_TRIE`` (read by `mode`, each emission): ``off`` (default: the seat decodes exactly as before), ``rel`` (the
relation literals), ``rel+slot`` (the seed literal too). The CotChain form (``Hk_<RELATION>`` role names) is not
constrained here: the served programs adapter writes the CotPlan form on questions without a facts block.

Wired: WIRED (`SlotEmitter.emit` builds a `PlanTrie` when the flag is on; `Cubby450mEmitter.emit` hands it to the
decoder; the ask loop hands the seat its `known`).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

MODES = ("off", "rel", "rel+slot")
TRIGGER = re.compile(r"bind\s+frame\s*,\s*(?P<role>HOP\d+|SEED)$")
CTX = {"HOP": "bind frame, HOP1", "SEED": "bind frame, SEED"}
TAIL = 12                                             # tokens decoded to look for the role just written


def mode() -> str:
    m = (os.environ.get("CB_EMIT_TRIE") or "off").strip().lower()
    if m not in MODES:
        raise ValueError(f"CB_EMIT_TRIE={m!r}: one of {', '.join(MODES)}")
    return m


def _encode(tk, text: str) -> list:
    return list(tk.encode(text).ids)


def _decode(tk, ids) -> str:
    return tk.decode(list(ids), skip_special_tokens=False)


class Options:
    """The token sequences of a set of literals, written in context after ``role``, as a trie. ``skipped``: the
    literals whose tokens did not split cleanly after the context (left out)."""

    def __init__(self, tk, literals, role: str = "HOP") -> None:
        ctx = CTX[role]
        base = _encode(tk, ctx)
        lits = sorted({str(x) for x in literals if x and '"' not in str(x) and "\\" not in str(x)})
        texts = [f'{ctx}, "{lit}";' for lit in lits]
        enc = getattr(tk, "encode_batch", None)
        rows = [list(e.ids) for e in enc(texts)] if enc else [_encode(tk, t) for t in texts]
        self.root: dict = {}
        self.n, self.skipped = 0, []
        for lit, ids in zip(lits, rows):
            if ids[:len(base)] != base or len(ids) == len(base):
                self.skipped.append(lit)
                continue
            node = self.root
            for t in ids[len(base):]:
                node = node.setdefault(t, {})
            node[None] = lit                              # an option ends here
            self.n += 1

    def node(self, ids):
        node = self.root
        for t in ids:
            node = node.get(t)
            if node is None:
                return None
        return node


_CACHE: dict = {}


def relation_options(tk, known) -> Options:
    """The relation labels ``known`` holds (normalized, as the gate reads them), built once per tokenizer and
    vocabulary size (a relation the loop learns grows the vocabulary: the options are rebuilt then)."""
    key = (id(tk), id(known), len(known))
    got = _CACHE.get(key)
    if got is None:
        for k in [k for k in _CACHE if k[:2] == key[:2]]:
            del _CACHE[k]
        got = _CACHE[key] = Options(tk, list(known), "HOP")
    return got


class PlanTrie:
    """``allowed(new_ids) -> list | None``: the ids the seat may write next, given the ids it wrote so far (the
    prompt cut off); None outside a constrained literal (no constraint). ``rel`` / ``seed``: `Options` (None: that
    literal is left free). One emission at a time: the state follows the ids it is shown, and resets when they are
    not a continuation of the last call's."""

    def __init__(self, tk, rel: Options | None = None, seed: Options | None = None) -> None:
        self.tk, self.opts = tk, {"HOP": rel, "SEED": seed}
        self.entered = {"HOP": 0, "SEED": 0}
        self.written: list = []                          # (role kind, the literal) of each option completed
        self.left = 0                                    # times the ids left the trie (never, under the mask)
        self.reset()

    def reset(self) -> None:
        self._seen: list = []
        self._lit = None                                 # (kind, the ids' index where the option starts)

    def _step(self, ids: list, i: int) -> None:
        """Token ``ids[i]`` was written."""
        if self._lit is not None:
            kind, start = self._lit
            node = self.opts[kind].node(ids[start:i + 1])
            if node is None:
                self.left += 1
                self._lit = None
            elif None in node:
                self.written.append((kind, node[None]))
                self._lit = None
            return
        m = TRIGGER.search(_decode(self.tk, ids[max(0, i + 1 - TAIL):i + 1]))
        if m:
            kind = "SEED" if m.group("role") == "SEED" else "HOP"
            if self.opts[kind] is not None and self.opts[kind].n:
                self._lit = (kind, i + 1)
                self.entered[kind] += 1

    def allowed(self, new_ids) -> "list | None":
        ids = [int(t) for t in new_ids]
        if ids[:len(self._seen)] != self._seen:
            self.reset()
        for i in range(len(self._seen), len(ids)):
            self._step(ids, i)
        self._seen = ids
        if self._lit is None:
            return None
        kind, start = self._lit
        node = self.opts[kind].node(ids[start:])
        return [t for t in node if t is not None] if node else None

    def stats(self) -> dict:
        return {"entered": dict(self.entered), "written": list(self.written), "left": self.left}


def for_seat(inner, known=None, slots=()) -> "PlanTrie | None":
    """The trie a seat decodes under, by ``CB_EMIT_TRIE``; None when off, when the seat cannot take one (it has no
    tokenizer: the GGUF stand-in), or when there is nothing to constrain."""
    m = mode()
    tk = getattr(getattr(inner, "trunk", None), "tokenizer", None)
    if m == "off" or tk is None:
        return None
    rel = relation_options(tk, known) if known is not None and len(known) else None
    seed = Options(tk, slots, "SEED") if m == "rel+slot" and slots else None
    return PlanTrie(tk, rel, seed) if rel is not None or seed is not None else None


def hand(emitter, known) -> int:
    """The ask loop hands its ``known`` to every slotted seat behind ``emitter`` (a `ContextualEmitter`'s
    adapters, a wrapper's ``inner``); returns how many took it."""
    seen, todo, n = set(), [emitter], 0
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "relations"):
            e.relations = known
            n += 1
        todo.extend(getattr(e, "adapters", {}).values())
        todo.append(getattr(e, "inner", None))
    return n


__all__ = ["MODES", "Options", "PlanTrie", "for_seat", "hand", "mode", "relation_options"]
