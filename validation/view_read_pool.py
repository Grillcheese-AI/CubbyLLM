"""Order 0067, the view reader's Colab read: each row's grammar checked in its own worker process.

Nick's first counted read (2026-11-14) slowed from 53 s a batch of 32 short rows to 1,570 s a batch of long ones: the
grammar re-reads the whole answer at every token (``complete`` and ``viable``), so a row costs the square of its
length, and 32 rows were checked one after another on one core while the GPU waited. Here the rows of a batch are
dealt to worker processes (row i to worker i mod W); each worker keeps its rows' grammars and choosers
(`view_grammar.chooser`, the same code as the serial read) and answers a step's requests for its rows. The choice is
the same as the serial read's: the same ranked ids in the same order, the first one the grammar keeps.

Per step the host sends a worker, for each of its rows, the token appended since the last step and the ranked ids;
the pieces (token id -> text) a worker has not yet seen go with them, so nothing in a worker decodes.

Wiring: STANDALONE (`exp_view_reader_torch.decode` uses it when ``--workers`` > 1).
"""
from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict

__wiring__ = "STANDALONE"


def _grammar(name: str):
    from cubbyllm.speech import view_grammar as VG
    from cubbyllm.speech import view_types as VT
    return {"typed": VT.TypedGrammar, "typed8": VT.TypedGrammar8, "lines": VG.Grammar}[name]


def _worker(conn) -> None:
    from cubbyllm.speech import view_grammar as VG
    pieces: dict = {}
    rows: dict = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        if msg[0] == "rows":                                   # a new batch: [(i, grammar name, instruction, eos)]
            rows.clear()
            for i, name, instruction, eos in msg[1]:
                rows[i] = (VG.chooser(_grammar(name)(instruction), pieces.__getitem__, eos, 0), [])
            conn.send("ok")
        else:                                                  # ("step", new pieces, [(i, token appended or None, ranked)])
            pieces.update(msg[1])
            out = []
            for i, tok, ranked in msg[2]:
                choose, gen = rows[i]
                if tok is not None:
                    gen.append(tok)
                out.append((i, choose(gen, ranked)))
            conn.send(out)


class GrammarPool:
    """W worker processes; `rows(specs)` starts a batch, `choose(requests, piece)` -> {row: [pick] or []}."""

    def __init__(self, workers: int):
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else "spawn")
        self.conns, self.procs, self.sent = [], [], []
        for _ in range(workers):
            here, there = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(there,), daemon=True)
            p.start()
            self.conns.append(here)
            self.procs.append(p)
            self.sent.append(set())

    def rows(self, specs: list) -> None:
        w = len(self.conns)
        for k, c in enumerate(self.conns):
            c.send(("rows", [s for s in specs if s[0] % w == k]))
        for c in self.conns:
            assert c.recv() == "ok"

    def choose(self, requests: list, piece) -> dict:
        w, by = len(self.conns), defaultdict(list)
        for r in requests:
            by[r[0] % w].append(r)
        for k, rs in by.items():
            new = {}
            for _, tok, ranked in rs:
                for t in ranked if tok is None else [tok, *ranked]:
                    if t not in self.sent[k] and t not in new:
                        new[t] = piece(t)
            self.sent[k].update(new)
            self.conns[k].send(("step", new, rs))
        out: dict = {}
        for k in by:
            out.update(self.conns[k].recv())
        return out

    def close(self) -> None:
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


__all__ = ["GrammarPool"]
