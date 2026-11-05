"""cubbyllm/selfimprove/signature.py -- a request's skill signature (order 0049, the log's key).

What a request asks of the trunk, read from a program and its record (the stream's reference program where the record
has one, so that a wrong attempt does not scatter its own skill across signatures; else the trunk's attempt -- the
ledger says which), never from the gold answer:
  relations   the program's relation words (``bind frame, HOPk, "word"``), in order
  lang        the question's language
  shape       hops (the HOP count), arithmetic (the steps: ``add/sub/mul/div`` lines), kernel (a ``use`` other than vsa)
  inverse     asked from the value's side (the program walks back: ``recover(..., VALUE)`` or a record's flag)
  task        the job the host asks of the emitter (a record's task: ``wd``, ``plan``, ``arithmetic``, ...); read from
              the program only when no record says it
  steps       an arithmetic problem's step count (a record's ``stats.K``). One record is one step, a single operation,
              so no level shows in its program: every level would share one key (0049's stream, 2026-11-05)
Two requests with equal signatures exercise the same skill: a gap is a signature that keeps failing.

Wired: STANDALONE (order 0049).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__wiring__ = "STANDALONE"

_HOP = re.compile(r'bind\s+\w+\s*,\s*HOP\d+\s*,\s*"([^"]*)"')
_ARITH = re.compile(r"^\s*(add|sub|mul|div)\s", re.M)
_USE = re.compile(r"^\s*use\s+(\w+)", re.M)
_INVERSE = re.compile(r"recover\([^)]*VALUE")


@dataclass(frozen=True)
class Signature:
    task: str
    relations: tuple
    lang: str
    hops: int
    arith: int
    kernel: str
    inverse: bool
    steps: int = 0

    def key(self, coarse: bool = False) -> str:
        """The ledger's key. ``coarse``: the relation words left out (a skill as "relation words of this language and
        shape", not one per relation): 0049's planted gap "Wikidata's relation words in English" is one coarse key and
        hundreds of fine ones. Which grain the run uses is written down before it runs. A stepped problem's key carries
        its step count, and coarse leaves out the one step's operation count (one problem's steps share a key)."""
        rel = "*" if coarse and self.relations else ",".join(self.relations)
        arith = "a*" if coarse and self.steps else f"a{self.arith}"
        return "|".join([self.task, rel, self.lang, f"h{self.hops}", arith, self.kernel, "inv" if self.inverse else "fwd"]
                        + ([f"s{self.steps}"] if self.steps else []))


def of(program: str, lang: str = "en", task: str = "", inverse: bool | None = None, steps: int = 0) -> Signature:
    """The signature of a request: its program (what the trunk wrote), its language and task, a stepped problem's
    step count."""
    text = program or ""
    rel = tuple(w.strip().lower() for w in _HOP.findall(text))
    uses = [u for u in _USE.findall(text) if u != "vsa"]
    inv = bool(_INVERSE.search(text)) if inverse is None else bool(inverse)
    return Signature(task or ("arithmetic" if _ARITH.search(text) and not rel else "lookup"), rel, lang or "en",
                     len(rel), len(_ARITH.findall(text)), uses[0] if uses else "-", inv, int(steps or 0))


__all__ = ["Signature", "of"]
