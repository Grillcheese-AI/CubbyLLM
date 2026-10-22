"""standin/plugins_world.py -- a world plugin's in-fiction facts as an askable world (order 0028).

``FictionWorld`` covers a question only when the question names one of its works (0015's rule), and answers with the
facts that work states, never with one it would have to infer:

- "who is X's <relation>?" finds the stated fact of that relation (``Gertrude is the mother of Hamlet``);
- a spouse asked the other way gets the stated fact that entails it ("who is Adriana's husband?" -> ``Adriana is the
  wife of Antipholus``); a parent or sibling asked the other way is refused: the list's "X, daughter to Y" does not say
  whether Y is X's father or mother;
- anything else: the work's facts the question's words reach (none: a refusal).

Wired: WIRED (``plugins_load.Loader`` mounts one for a world plugin with ``route: fiction``).
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"

# Only an inversion the stated fact entails: "X is the wife of Y" says Y is X's husband. A parent or a sibling asked
# the other way would need the other person's sex, which a persons list rarely states -- "who is Juliet's mother?"
# answered with "Juliet is the daughter of Capulet" reads as Capulet being her mother (found on the real pack): refused.
INVERSE = {"husband": ("wife",), "wife": ("husband",), "spouse": ("wife", "husband")}


class FictionWorld:
    def __init__(self, name: str, domain: str, db_path) -> None:
        from ..worlds.fiction import FictionWorlds
        self.name, self.domain, self.fw = name, domain, FictionWorlds(db_path)

    def covers(self, question: str) -> float:
        return 1.0 if self.fw.work_named(question) else 0.0

    def _rows(self, sql: str, args) -> list[str]:
        return [f"{o} is the {r} of {s}" for s, r, o in self.fw.db.execute(sql, args)]

    def answer(self, question: str) -> list[str]:
        from ..worlds.fiction import KIN
        work = self.fw.work_named(question)
        if work is None:
            return []
        q = self.fw.strip(question, work)
        rels = sorted(set(KIN) | set(KIN.values()) | set(INVERSE))
        m = re.search(r"(?P<s>[A-Z][\w-]*(?:\s[A-Z][\w-]*)?)['’]s\s+(?P<r>" + "|".join(rels) + r")\b", q)
        if not m:
            return [f for s, f in self.fw.store(work)(q, 3) if s >= 0.5]
        who, asked = m["s"], m["r"].lower()
        rel = KIN.get(asked, asked)
        got = self._rows("SELECT subj, rel, obj FROM fiction_facts WHERE work = ? AND lower(subj) = lower(?) AND rel IN (?, ?)",
                         (work, who, asked, rel))
        if got:                                          # stated as asked: Hamlet's mother
            return got
        inv = INVERSE.get(asked, ())                     # stated the other way: the fact that answers it, as stated
        if not inv:
            return []
        marks = ",".join("?" * len(inv))
        return [f"{s} is the {r} of {o}" for s, r, o in self.fw.db.execute(
            f"SELECT obj, rel, subj FROM fiction_facts WHERE work = ? AND lower(obj) = lower(?) AND rel IN ({marks})",
            (work, who, *inv))]
