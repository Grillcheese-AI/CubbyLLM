"""cubbyllm/reasoning/ask_namesakes.py -- a question about a name several items share (order 0033).

A world that holds a name only as several items' (the hdc world: three items are labelled "Globe Theatre") gives none
of their facts by the name alone (`worlds.hdc_namesakes`). Two things decide which item:

1. the asker: a clarify choice comes back with its item;
2. the question's own context, a qualifier the name carries. "Bambi (film)" names a value of the item; "Paris, Texas"
   and "the Globe Theatre in Boston" name where it is, and only the values that say where count. A qualifier
   decides when exactly one of the items has it.

Nothing else decides; the question's relations do not, in this snapshot (`worlds.hdc_namesakes` says why). Otherwise
the reply is `ambiguous_entity`, with the items named and a choice for each: the same clarify that "what is Ottawa?"
already gets from the online source.

A world that holds the name as one thing (the wiki world: one article per title) answers as before. A profile ask
comes here only when the stores hold no profile fact about the name.

Wired: WIRED (`ask_profile.AskProfile.profile`, `ask.AskLoop._ask`).
"""
from __future__ import annotations

import contextlib
import re as _re

from . import ask_namesake_answer as answer
from .ask_forms import PROFILE
from .ask_words import say

__wiring__ = "WIRED"

CHOICES_MAX = 8                                          # the clarify's choices, the most described first
DETAIL_FIRST = ("instance of", "occupation", "country", "located in the administrative territorial entity", "country of origin")

QUALIFIED = (                                            # (form, whether the qualifier says where the item is)
    (_re.compile(r"^(?P<name>.+?)\s*\((?P<q>[^()]+)\)$"), False),                 # Bambi (film)
    (_re.compile(r"^(?P<name>[^,]+?),\s+(?P<q>[^,]+)$"), True),                     # Paris, Texas
    (_re.compile(r"^(?P<name>.+?)\s+in\s+(?:the\s+)?(?P<q>.+)$", _re.I), True),   # the Globe Theatre in Boston
)


def readings(entity: str) -> list[tuple[str, str, bool]]:
    """(name, qualifier, place) readings of an entity that carries its own context."""
    out = []
    for rx, place in QUALIFIED:
        m = rx.match(" ".join((entity or "").split()))
        if m:
            name = m.group("name").strip()
            out.append((name[4:] if name.lower().startswith("the ") else name, m.group("q").strip(), place))
    return out


def detail(rows) -> str:
    """An item told apart by its own values, what kind of thing and where first: "located in: Boston"."""
    order = {r: i for i, r in enumerate(dict.fromkeys(DETAIL_FIRST + PROFILE))}
    by: dict = {}
    for _f, t in rows:
        by.setdefault(t.rel, []).append(t.obj)
    rels = sorted(by, key=lambda r: order.get(" ".join(r.lower().split()), len(order)))
    return "; ".join(f"{say(r)}: {', '.join(dict.fromkeys(by[r][:3]))}" for r in rels[:3])


def choices(by: dict) -> list[tuple[str, str, str]]:
    """(item, label, detail) per item, as the online source's `ambiguous` list. The most described come first: an
    order to read them in, never a pick."""
    ranked = sorted(by.items(), key=lambda kv: -len(kv[1]))[:CHOICES_MAX]
    return [(q, rows[0][1].subj, detail(rows)) for q, rows in ranked]


def _found(by: dict, q: str, how: str) -> dict:
    return {"item": q, "label": by[q][0][1].subj, "rows": by[q], "how": how}


def _decided(world, entity: str, item: str | None, held: bool):
    """(by, (item, how) or None) for the first reading of `entity` that is a shared name; (None, None) when none is.
    `held`: the stores hold the whole entity as one thing, so its qualified readings are not tried."""
    for name, qual, place in [(entity, None, False)] + ([] if held else readings(entity)):
        by = world.namesakes(name)
        if by:
            if item in by:
                return by, (item, "chosen by the asker")
            q = world.pick_by(name, qual, place) if qual else None
            return by, ((q, f"the question's own context ({qual})") if q else None)
    return None, None


def pick(loop, entity: str, item: str | None = None) -> dict | None:
    """A profile ask's entity held only as a shared name. Returns {"item", "label", "rows", "how"} when something
    decides; {"ambiguous": [(item, label, detail)], "of": n} when nothing does; None when it is no shared name."""
    if not hasattr(loop.world, "namesakes"):
        return None
    by, got = _decided(loop.world, entity, item, held=False)
    if by is None:
        return None
    return _found(by, *got) if got else {"ambiguous": choices(by), "of": len(by)}


def seed_for(loop, seed: str, item: str | None = None, question: str | None = None, rels=None):
    """The walk's context for a seed held only as a shared name: the asker's item or the qualifier's pick pins the
    seed to that one item (the world's `pinned`; the question's own words stay the plan's, so the plan still covers
    the question). Otherwise nothing, and the world's hop decides by its relation or withholds (`after_walk`); with
    ``CB_NAMESAKE_RELATION=1`` (order 0037) the hop reads the question's cue words (`ask_namesake_answer`)."""
    if not hasattr(loop.world, "namesakes"):
        return contextlib.nullcontext()
    loop.world.clear_withheld()
    by, got = _decided(loop.world, seed, item, held=bool(loop.facts_about(seed)))
    if got:
        return loop.world.pinned(seed, by[got[0]][0][1].subj, got[0])
    if question and answer.on() and hasattr(loop.world, "cued"):
        return loop.world.cued(answer.cue_words(question, seed, rels))
    return contextlib.nullcontext()


def after_walk(loop, rec: dict, seed: str, learn=None) -> None:
    """A walk refused because a world withheld facts its hop would have read on a shared name that nothing decided,
    and no world holds the name as one thing: `ambiguous_entity`, the items that held them first. Any other refusal
    keeps its own reason (no item held the fact: a source gap, not a question of which). With
    ``CB_NAMESAKE_RELATION=1`` (order 0037), when the question's relation decided: one item fits, its answer and the
    other namesakes named; two of a kind, each walked apart (`learn`) and answered, labelled (`ask_namesake_answer`)."""
    world = loop.world
    if not hasattr(world, "withheld") or loop.facts_about(seed):
        return
    d = world.decision(seed) if answer.on() and hasattr(world, "decision") else None
    if d and len(d["fit"]) == 1:
        answer.one_fits(loop, rec, seed, d, world.namesakes(seed))
        return
    if d and len(d["fit"]) > 1 and not rec.get("verified") and answer.several_fit(loop, rec, seed, d, world.namesakes(seed), learn):
        return
    if rec.get("verified"):
        return
    holders, by = world.withheld(seed), world.namesakes(seed)
    if holders and by:
        first = {q: by[q] for q in holders if q in by}
        refuse(rec, choices(first) + choices({q: r for q, r in by.items() if q not in first}), len(by))


def refuse(rec: dict, amb: list, of: int | None = None) -> None:
    """The record of a question whose name nothing decided: `ambiguous_entity`, the items named, a choice each."""
    rec["reason"] = "ambiguous_entity"
    rec["candidates"] = [[a[1], a[2], a[0]] for a in amb]
    rec["clarify"] = {"question": "Which one do you mean?", "choices": [{"label": a[1], "detail": a[2], "item": a[0]} for a in amb]}
    if of and of > len(amb):
        rec["clarify"]["more"] = of - len(amb)
