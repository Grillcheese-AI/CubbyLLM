"""cubbyllm/reasoning/cover_names.py -- order 0087: a plan must use the names the question holds.

With ``CB_COVER_NAMES=1`` (off by default; off, nothing here runs and every output is as before), a plan that
``covers()`` passed is disposed of three more ways, wherever a plan is disposed of (``pipeline.answer`` after
``verify_plan``: the emitter's plan, the grammar's parse, a remembered plan, a host shape's restatement alike):

1. ``seed_is_relation``: the seed, normalized, is made only of words of the relations the plan walks. The walked
   relations are every hop's relation in each wording ``covers()`` accepts (the plan's, the store's paraphrase, the
   words a relabelled hop was asked under), and for hop 0 every reading of its tail that names a relation (the store
   holds it, or the source's property table has it as a label or an alias), the tail as a whole included: the plan
   ``country`` of ``citizenship`` walks a tail that is itself `country of citizenship`, whatever the store has
   fetched so far. The seed "citizenship" is that relation's word, not a name (0083's seven "Ghana" rows).
2. ``name_left_over``: an entity span of the question's slot table (``slots.extract``: the names the host knows and
   the question's own capitalised runs, H-E15) that the plan neither seeds nor binds. A span made only of the walked
   relations' words is the relation, not a name; a span inside an aside (covers() v4's parenthesis) is not counted,
   unless the seed itself holds it ("Tombo (album)"). Needs the slot table; without one, this check does not run.
3. ``seed_is_pronoun``: a pronoun (``PRONOUNS``, a closed list) never seeds a plan, and with the flag on the slot
   table offers none as an entity span (``drop_pronouns``, in ``slots.extract``), even when the name index holds an
   item by that name ("who are you?" was answered with the profile of "You", a television personality). The host's
   profile shape is refused the same way (``AskProfile.profile``).

Nothing here changes ``plan_verify``, the walk or the VM. The checks run once ``covers()`` has passed, before the
unknown-relation refusal (whose fetches would otherwise be spent on a plan refused anyway); a refusal is returned
before any walk or fetch, with its reason.

Wired: WIRED (``pipeline.answer``, ``slots.extract``, ``AskProfile.profile``; the ask loop hands the slot table
through ``learn_and_answer(names=)`` from ``AskRecall._walk_setup``).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

PRONOUNS = frozenset("i me my mine you your yours he him his she her hers it its we us our ours they them their theirs "
                     "this that these those someone anyone everyone something who what".split())
_PAREN = re.compile(r"\([^)]*\)")                     # covers() v4's aside (plan_verify._PAREN)


def on() -> bool:
    """``CB_COVER_NAMES=1``: the three checks run. Off by default."""
    return os.environ.get("CB_COVER_NAMES") == "1"


def _words(text: str) -> list[str]:
    from .planner import normalize
    return normalize(text or "").split()


def is_pronoun(text: str) -> bool:
    """The text, normalized, is one pronoun of the closed list."""
    w = _words(text)
    return len(w) == 1 and w[0] in PRONOUNS


def drop_pronouns(spans: list) -> list:
    """`SpanIndex.spans`' [(text, start, end)] without the pronouns (``slots.extract``, flag on)."""
    return [s for s in spans if not is_pronoun(s[0])]


def hop0(plan, known, question: str) -> tuple[str, str]:
    """(relation, seed) of the plan's tail, read as ``covers()`` reads it: the longest held relation prefix, then
    the paraphrase tier, else the question decides the split."""
    from .plan_verify import split_tail, tail_match, tail_relation
    tr = tail_relation(plan.tail, known) if known is not None else None
    if tr is None and known is not None:
        tm = tail_match(plan.tail, known)
        tr = tm[0] if tm else None
    if tr is None:
        return split_tail(plan.tail, None, question)
    ent = plan.tail[len(tr):].strip()
    return tr, (ent[3:] if ent.startswith("of ") else ent)


def relation_words(plan, known, aliases: dict | None, question: str, holds=None) -> set[str]:
    """Every word of the relations the plan walks, in every wording covers() accepts for them, and of every
    reading of hop 0's tail that names a relation (the whole tail included). `holds`: does a text name a relation
    (the store's vocabulary or the property table; default: the store's, `known`)."""
    from .planner import normalize, of_prefixes
    rel1, _seed = hop0(plan, known, question)
    rels = [rel1] + [r for r in plan.relations[1:] if r]
    holds = holds or (lambda r: known is not None and r in known)
    rels += [r for r in of_prefixes(plan.tail) + [normalize(plan.tail)] if holds(r)]
    out: set[str] = set()
    for r in rels:
        out.update(_words(r))
        alt = known.match(r) if known is not None and hasattr(known, "match") and r not in known else None
        out.update(_words(alt or ""))
        for w in (aliases or {}).get(normalize(r), ()):
            out.update(_words(w))
    return out


def left_over(table, seed: str, rel_words: set[str], referenced=None) -> list[str]:
    """The entity spans of `table` the plan neither seeds (word for word inside the seed) nor binds (`referenced`:
    the slot ids its program names), that are not the relation's words and not inside an aside."""
    seed_s = " " + " ".join(_words(seed)) + " "
    asides = [(m.start(), m.end()) for m in _PAREN.finditer(table.question)]
    out = []
    for s in table.spans:
        if s.kind != "E" or s.id in (referenced or ()):
            continue
        w = _words(s.text)
        if not w or " " + " ".join(w) + " " in seed_s or set(w) <= rel_words:
            continue
        if any(a <= s.start and s.end <= b for a, b in asides):
            continue
        out.append(s.text)
    return out


def refusal(question: str, plan, known, aliases: dict | None = None, names=None) -> tuple[str, dict] | None:
    """(reason, detail) when the plan fails one of the three checks, else None. `names`: (slot table, the slot ids
    the program referenced or None, a `holds` for `relation_words` or None), or None (checks 1 and 3 only, over the
    store's vocabulary). Called with the flag on."""
    _rel, seed = hop0(plan, known, question)
    if is_pronoun(seed):
        return "seed_is_pronoun", {"seed": seed}
    table, referenced, holds = names if names else (None, None, None)
    rw = relation_words(plan, known, aliases, question, holds)
    sw = set(_words(seed))
    if sw and sw <= rw:
        return "seed_is_relation", {"seed": seed, "relation_words": sorted(rw)}
    if table is not None:
        over = left_over(table, seed, rw, referenced)
        if over:
            return "name_left_over", {"seed": seed, "spans": over}
    return None


def slots(loop, question: str):
    """(slot table, referenced ids, holds) for the ask loop's walk: the programs seat's own table when it was built
    for this question (and the slots its program named), else the table ``slots.extract`` builds over the seat's name
    index (a remembered plan, a host shape); the table is None when the loop has no slotted seat. `holds`: a text
    names a relation when the store holds it or the source's property table has it (``WikidataAliases.relations``)."""
    from .ask_words import strip_as_of
    from .slots import extract
    aliases = getattr(loop.source, "aliases", None)
    known = loop.known

    def holds(r: str) -> bool:
        return (known is not None and r in known) or bool(aliases is not None and aliases.relations(r))
    em = loop.emitter
    seat = em.adapters.get("programs", em) if isinstance(getattr(em, "adapters", None), dict) else em
    if not (hasattr(seat, "index") and hasattr(seat, "last")) or seat.index is None:
        return None, None, holds
    asked = strip_as_of(question)[0]
    last = seat.last or {}
    table = last.get("table")
    if table is not None and table.question == asked:
        verdict = last.get("verdict")
        return table, (list(verdict.referenced) if verdict is not None else None), holds
    return extract(asked, seat.index), None, holds
