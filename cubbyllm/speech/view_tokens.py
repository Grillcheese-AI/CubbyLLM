"""cubbyllm/speech/view_tokens.py -- the reader's machine in token ids (order 0073, "The grammar is a state machine",
points 3 to 5): which ids may come next, looked up and never judged one by one.

`Vocab(pieces)` holds the vocabulary once as a prefix tree of its pieces (``{id: the text it prints}``). A piece
that prints nothing is left out: it is never allowed. `allowed(machine, vocab, state)` walks the tree along the
machine's moves from a state and returns the ids whose whole piece the machine takes. Inside a free string almost
every piece fits, so the pieces with no quote, backslash or control character are kept by length once for the
vocabulary and only the others are walked: the same set.

`Reader(machine, vocab, eos)` is one answer being written: `allowed()` -> the ids it may take next (the end token
only when the answer is whole); `push(id)`; `end` is how it ended: ``finished`` (the end token, on a whole answer),
``error`` (no id is allowed: by the machine's construction this does not happen, and a read where it does is not
scored until the cause is found) or ``limit`` (`stop("limit")`, the decoder's token limit); `n` counts the ids.

Wiring: STANDALONE (the reader's read calls it; nothing in the serve path: CB_RULES stays off).
"""
from __future__ import annotations

from cubbyllm.speech import view_desc as D
from cubbyllm.speech.view_machine import START, Machine

__wiring__ = "STANDALONE"


class Vocab:
    def __init__(self, pieces: dict):
        self.pieces = {i: p for i, p in pieces.items() if p}
        self.root: dict = {}
        self.marked: dict = {}                               # the pieces a free string cannot simply take in
        by_len: list = [[] for _ in range(D.MAX_STRING + 1)]
        for i, p in self.pieces.items():
            self._add(self.root, p, i)
            if all(c >= " " and c not in '"\\' for c in p):
                if len(p) <= D.MAX_STRING:
                    by_len[len(p)].append(i)
            else:
                self._add(self.marked, p, i)
        self.plain, kept = [], set()                         # plain[k]: the plain pieces of at most k characters
        for ids in by_len:
            kept |= set(ids)
            self.plain.append(frozenset(kept))

    @staticmethod
    def _add(root: dict, piece: str, i: int) -> None:
        node = root
        for ch in piece:
            node = node.setdefault(ch, {})
        node.setdefault("", []).append(i)

    @classmethod
    def of(cls, tokenizer) -> "Vocab":
        """From a `tokenizers.Tokenizer`: each id's piece is what it decodes to alone."""
        return cls({i: tokenizer.decode([i]) for i in range(tokenizer.get_vocab_size())})


def walk(machine: Machine, state: tuple, root: dict) -> set:
    """The ids under `root` whose whole piece the machine takes from `state`."""
    out, todo = set(), [(state, root)]
    while todo:
        st, node = todo.pop()
        for ch, child in node.items():
            if ch == "":
                continue
            nxt = machine.step(st, ch)
            if nxt is not None:
                out.update(child.get("", ()))
                todo.append((nxt, child))
    return out


def allowed(machine: Machine, vocab: Vocab, state: tuple) -> frozenset:
    if state[0] == "str" and not state[3]:                   # inside a free string, not after a backslash
        return vocab.plain[D.MAX_STRING - len(state[2])] | walk(machine, state, vocab.marked)
    return frozenset(walk(machine, state, vocab.root))


class Reader:
    def __init__(self, machine: Machine, vocab: Vocab, eos: int):
        self.machine, self.vocab, self.eos = machine, vocab, eos
        self.state, self.end, self.n = START, None, 0
        self._ids: dict = {}

    def allowed(self) -> frozenset:
        """The ids the answer may take next; none once it has ended. No id allowed ends it with ``error``."""
        if self.end:
            return frozenset()
        if self.state not in self._ids:
            ids = allowed(self.machine, self.vocab, self.state) - {self.eos}
            self._ids[self.state] = ids | {self.eos} if self.machine.accepting(self.state) else ids
        if not self._ids[self.state]:
            self.end = "error"
        return self._ids[self.state]

    def push(self, token: int) -> None:
        if token not in self.allowed():
            raise ValueError(f"id {token} is not allowed here")
        self.n += 1
        if token == self.eos:
            self.end = "finished"
        else:
            self.state = self.machine.run(self.vocab.pieces[token], self.state)

    def stop(self, why: str = "limit") -> None:
        self.end = self.end or why


__all__ = ["Reader", "Vocab", "allowed", "walk"]
