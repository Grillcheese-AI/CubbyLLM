"""cubbyllm/speech/rule_stack.py -- a session's rules across the levels, and a system prompt compiled once into its
book (order 0067, step 1: roles in the API).

- `RuleStack(books, conversation)`: what a turn reads as its rule book (`chat.CubbyChat.rules`): the rules in force
  from every level's book (global, the company's, the session's system prompt), highest level first, then the
  conversation's own; rules said back in the conversation go to the conversation's book. Each book keeps its own
  rules (ids differ by the book's prefix), so a level's prompt replaced or corrected changes that book alone.
- `split(text)`: a system prompt's sentences, in order (a sentence ends at . ! ? or a line break).
- `compile_prompt(text, level, set_by, at, read, book, lang)`: each sentence read once (``read(sentence)`` ->
  {"how": "written" | "asked" | "none", "plan"}); a written plan becomes a rule of `level` in `book`, set by `set_by`
  at turn `at`; a plan the family does not run is kept as "asked". The book is then shown to whoever set it
  (`RuleBook.show_book`), so its rules may apply. Returns the entries and the lines shown. With no reader mounted
  (``read`` None) every sentence is kept as "asked": nothing is checked that nobody read.
  Stated facts and the voice (step 5's split) are not separated yet: a sentence is a rule, asked or none.

Wired: STANDALONE (`cubby_serve/rule_api.py` builds a stack per session behind ``CB_API_ROLES=1``).
"""
from __future__ import annotations

import re

from .rule_book import LEVELS, RuleBook

__wiring__ = "STANDALONE"

_END = re.compile(r"(?<=[.!?])\s+|\n+")


class RuleStack:
    def __init__(self, books: list, conversation: RuleBook) -> None:
        self.books = [b for b in books if b is not None]
        self.conversation = conversation

    def in_force(self, turn: int) -> list:
        got = [r for b in self.books + [self.conversation] for r in b.in_force(turn)]
        return sorted(got, key=lambda r: LEVELS.index(r.level))       # stable: within a level, each book's own order

    def unshown(self, turn: int) -> list:
        return self.conversation.unshown(turn)

    def show(self, *a, **k) -> str:
        return self.conversation.show(*a, **k)

    def get(self, rule_id: str):
        for b in self.books + [self.conversation]:
            for r in b.rules:
                if r.id == rule_id:
                    return r
        raise KeyError(rule_id)


def split(text: str) -> list[str]:
    return [s.strip() for s in _END.split(text or "") if s and s.strip()]


def compile_prompt(text: str, level: str, set_by: str, at: int, read, book: RuleBook, lang: str = "en") -> dict:
    entries = []
    for s in split(text):
        got = read(s) if read is not None else {"how": "asked", "plan": None}
        how = got.get("how")
        if how == "written" and got.get("plan"):
            try:
                r = book.set(got["plan"], level, set_by, at, text=s)
                entries.append({"sentence": s, "kind": "rule", "rule_id": r.id})
                continue
            except ValueError:                                         # not a plan the family runs: kept, never checked
                how = "asked"
        entries.append({"sentence": s, "kind": "none" if how == "none" else "asked"})
    lines = book.show_book(entries, set_by, at, lang)
    return {"entries": entries, "rule_book": lines,
            "rules": [e["rule_id"] for e in entries if e["kind"] == "rule"],
            "held": [e["rule_id"] for e in entries if e["kind"] == "rule" and book.get(e["rule_id"]).held]}


__all__ = ["RuleStack", "compile_prompt", "split"]
