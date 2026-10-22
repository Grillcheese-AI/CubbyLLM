"""cubbyllm/reasoning/ask_words.py -- how the ask loop words what it found: the display names of relations, hedges,
dates and "as of", categories, the grounded prose of a profile, and whether a value is the one held. Split out of
ask.py (order 0031), verbatim; ask.py re-exports every name.

Wired: WIRED (the ask loop; the SFT builders read grounded_prose).
"""
from __future__ import annotations

import re as _re

__wiring__ = "WIRED"


SAY = {
    # A display rename may only DROP the source's schema words. It may never ADD one: Wikidata's P131 does
    # not say "province" and its P669 does not say "road", so speaking either would be a claim with no
    # provenance -- the one thing the kill line forbids. To earn the word "province", fetch Quebec's own
    # `instance of` and let THAT fact license it.
    "located in the administrative territorial entity": "located in",
    "contains the administrative territorial entity": "contains",
    "located in or next to body of water": "located on or next to",
    "located on street": "located on",
    "located in time zone": "in time zone",
}


def hedged(answer: str, lang: str = "en") -> str:
    """An answer one source alone states (order 0012), said as an account and never as fact; the source is not
    named (order 0016)."""
    return f"Selon une source, {answer}." if lang.startswith("fr") else f"One account says {answer}."


def say(rel: str) -> str:
    """A relation as it should READ, never as it is stored. The template is `<obj> is the <rel> of <subj>`,
    so a relation whose own label ends in 'of' renders 'capital of of Quebec City' (2026-09-14, live). The
    stored text is the key the gate, the walk and the VM all share and is never rewritten; this is the
    display layer only, and SAY above bounds how far it may go."""
    r = " ".join(str(rel or "").split())
    r = SAY.get(r.lower(), r)
    return r[:-3] if r.endswith(" of") else r


def when_text(w: str) -> str:
    """A stored time as it should read: the source writes an unknown month and day as 00, so
    '2024-00-00' is simply the year 2024 -- and a leading minus is the ERA, so -2560-01-01 is 2560 BC,
    not a negative number (Nick, 2026-09-14, on the Great Pyramid)."""
    out = []
    for part in str(w or "").split("\u2013"):
        q = part.strip()
        bce = q.startswith("-")                          # the sign is the era, not a minus sign
        if bce:
            q = q[1:]
        if bce or _re.fullmatch(r"\d{4}-00-00", q):
            q = q[:4]
        elif _re.fullmatch(r"\d{4}-\d{2}-00", q):
            q = q[:7]
        if q:
            out.append(q + " BC" if bce else q)
    return "\u2013".join(out)


_AS_OF_PHRASE = _re.compile(r"\s*\b(?:in|as of|during|back in)\s+(1[0-9]{3}|2[0-9]{3})\b\s*", _re.I)


def strip_as_of(question: str) -> tuple[str, str | None]:
    """(the question without its year phrase, the year) -- the host takes the time OFF the
    question before the emitter sees it.

    2026-09-14, live: "what was the position held of bill haslam in 2015?" made gen 2 emit the
    relation `position held in 2015`, and "the population of quebec city in 2011" made the seed
    `quebec city 2011 population`. Both are unknown_relation / does-not-cover before any walk
    runs, so the store's dates never got a chance to answer a question they can answer perfectly
    well. The year is not part of the relation; it is a constraint ON the answer, and the walk
    reads it from the original question (`pipeline._AS_OF`). Same move as `_strip_place_ask`:
    the model proposes, and the host hands it a question it can propose FOR.

    A year inside the entity's own name is left alone -- "what is the 2011 census" is not a
    question about 2011, it is a question about a thing called 2011 census."""
    m = _AS_OF_PHRASE.search(question or "")
    if not m:
        return question, None
    head, tail = question[:m.start()], question[m.end():]
    if not head.strip():                                 # the phrase IS the question's subject
        return question, None
    stripped = " ".join((head.rstrip() + " " + tail.lstrip()).split())
    return (stripped or question), m.group(1)


_TOKEN = _re.compile(r"[A-Za-zÀ-ÿ][\w'’\-]*|\d[\d.,:/\-]*")


_SMALL = frozenset("the a an and or of in on at to for by with from as is was were are be been being it its this that "
                   "he she they his her their who whom which what where when also known born died one two three four five".split())


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")


_ISO = _re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def date_words(facts: list[str]) -> list[str]:
    """The month name, day and year of every ISO date in the facts: '2015-12-11' is also 'December 11,
    2015' (2026-09-14, live: the grounding rule refused an exact paraphrase of OpenAI's inception over
    the word 'December'). A date the store holds, spelled the way people spell it, is not an addition."""
    out: list[str] = []
    for f in facts:
        for y, m, d in _ISO.findall(f):
            i = int(m)
            if 1 <= i <= 12:
                out.append(_MONTHS[i - 1])
            out.append(y)
            if d:
                out.append(str(int(d)))
    return out


_GROUP = _re.compile(r"[,\u00a0\u202f\u2009]")


def _num_key(tok: str) -> str:
    """A number as its digits alone, without thousands grouping: '574,482' and '574482' are the SAME
    number. 2026-09-14, live: a paraphrase of Quebec City that was faithful in every word was refused
    because the talk adapter grouped a population the store holds bare. The guard was right to check and
    wrong about the answer -- an over-refusal is not free, it is the loop declining to speak the truth."""
    return _GROUP.sub("", str(tok)).strip(".:/-")


_CATEGORY = _re.compile(
    r"\b(?:is|was|are|were)\s+(?:a|an|the)\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,3}?)"
    r"(?=\s*(?:[,.;:]|$|\bin\b|\bof\b|\bat\b|\bon\b|\bfor\b|\bthat\b|\bwhich\b|\band\b|\bwith\b|"
    r"\bbased\b|\blocated\b|\bheadquartered\b|\bknown\b|\bfounded\b|\bborn\b))")


def category_claims(text: str) -> list[str]:
    """The head noun of every 'X is a <kind>' in the paraphrase. A category is the one slot where the
    name-and-number guard is blind: every word of 'the Bank of England is a financial institution' is
    lowercase, so nothing flagged it -- and no fact in the store said 'institution'. 2026-09-14, live.
    A kind is a claim like any other and needs a fact behind it."""
    out = []
    for m in _CATEGORY.finditer(text or ""):
        head = m.group(1).split()[-1].strip("-")
        if head and head not in _SMALL:
            out.append(head)
    return out


def grounded_prose(text: str, facts: list[str], entity: str) -> tuple[bool, list[str]]:
    """Every capitalised word and every number in the paraphrase must occur in the facts (or the entity's
    name, or a date the facts hold written out): the talk adapter says what the store says, in its words,
    and nothing it adds is spoken. -> (ok, the offending tokens)."""
    hay = " ".join(facts + [entity] + date_words(facts)).lower()
    nums = {_num_key(t) for t in _TOKEN.findall(" ".join(facts + [entity])) if t[:1].isdigit()}
    bad: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower().strip("'’-.,:")
        # A possessive of a grounded name is the name: "Bill Haslam's father is Jim Haslam." was refused
        # over "Haslam's" (2026-09-24, the ground_sft pilot; the same guard serves the loop). The stem
        # must still occur in the facts, so nothing new is licensed.
        if low.endswith(("'s", "’s")):
            low = low[:-2]
        if not low or low in _SMALL:
            continue
        is_name = tok[0].isupper() and low not in _SMALL
        is_num = tok[0].isdigit()
        if is_num and _num_key(low) in nums:
            continue
        if (is_name or is_num) and low not in hay:
            bad.append(tok.strip("'’-.,:"))
    bad += [c for c in category_claims(text) if c not in hay]
    return (not bad), bad


ABSENT_REPLY = "The facts don't say."


def _vnorm(s) -> str:
    return " ".join(_re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def has_value(answer: str, value: str) -> bool:
    """The answer states the value: as words (punctuation and case aside), or an ISO date as its year
    plus its month or day spelled out ('1953-12-17' -> '... December 17, 1953'). The one definition --
    the talk data's checks and the H-E6 gate read values the way the host does."""
    a, v = _vnorm(answer), _vnorm(value)
    if v and f" {v} " in f" {a} ":
        return True
    if _re.fullmatch(r"-?\d{4}-\d{2}-\d{2}", str(value).strip()):
        y = str(value).strip().lstrip("-")[:4]
        words = [_vnorm(w) for w in date_words([value]) if w]
        return y in a.split() and any(w and w != y and f" {w} " in f" {a} " for w in words)
    return False


def value_check(answer: str, returned: list[str], others: list[str], entity: str = "") -> tuple[bool, list[str]]:
    """The talk adapter's reply to a question the VM answered is spoken only if it states a value the VM
    RETURNED and names nothing else from the facts block it was shown (another entity, another line's
    value). The name-and-number guard cannot see a binding error -- every name in it is grounded -- but
    the host knows what the VM returned, so this is a check, not a model (H-E6: it refuses all 10 of
    talk_v1's wrong-but-guarded answers). `others`: every other value and entity in the block -- never the
    entity asked about, which the answer is expected to name. -> (ok, what failed)."""
    problems = []
    if not any(has_value(answer, v) for v in returned):
        problems.append("returned value missing")
    rest = f" {_vnorm(answer)} "
    for v in sorted([*returned, entity], key=lambda s: -len(_vnorm(s))):   # 'Ranma12' inside the returned 'Ranma12
        if _vnorm(v):                                    # New Anime2024', or 'String Quintet' inside the asked 'String
            rest = rest.replace(f" {_vnorm(v)} ", " ")  # Quintet Schubert', is that value or that name, not another
    # one -- the longest first, so a returned '150' does not cut the asked 'Teotihuacan expansion by A.D. 150' apart
    keep = {_vnorm(v) for v in returned}
    for o in others:
        n = _vnorm(o)
        if n and n not in keep and f" {n} " in rest:
            problems.append(o)
    return (not problems), problems
