"""standin/time_rule.py -- the time rule (order 0012, Cowork's review of the books bench): a relation whose value
changes over a subject's life is said flat only with a date.

"The capital of the Kingdom of Kush is Kerma" was said flat on two books: true once, not for the kingdom's whole life
(Napata and Meroe came after). So for a relation in `CHANGES`, a fact the books witness is said flat only when every
witness dates it (a reign's years, an event's date: `book_frames.read`'s `when`), and then under the question's tense
and year as every dated fact is (`question_forms.time_check`). Undated, it is hedged in time ("Kerma was a capital of
the Kingdom of Kush at one point."), or refused when the question asks about a year the fact cannot be placed in.

`CHANGES` is seeded from the Wikidata properties whose statements take start / end time qualifiers (the hdc world's
`TIME_SENSITIVE`, order 0014) plus the review's list: capital, ruler and head of state, spouse, member of, the
country a place belongs to, population, an office's holder. Labels are Wikidata's, as the books and Wikidata write
them, with the wiki world's nouns for the same properties.

A world's own facts are dated by the world (the wiki world's present, hdc's "as of" snapshot); the rule reads the
books' witnesses. Wired: WIRED (`ask.AskLoop`, with the books on).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize

__wiring__ = "WIRED"

# Wikidata property -> label: TIME_SENSITIVE (P6 ... P749) and the review's (P36 capital, P17 country, P131, P361)
PROPERTIES = {"P6": "head of government", "P35": "head of state", "P39": "position held", "P54": "member of sports team",
              "P102": "member of political party", "P108": "employer", "P118": "league", "P127": "owned by",
              "P137": "operator", "P159": "headquarters location", "P169": "chief executive officer", "P286": "head coach",
              "P488": "chairperson", "P1082": "population", "P26": "spouse", "P463": "member of", "P1037": "director / manager",
              "P2828": "corporate officer", "P1308": "officeholder", "P451": "unmarried partner", "P551": "residence",
              "P937": "work location", "P1416": "affiliation", "P749": "parent organization", "P36": "capital",
              "P17": "country", "P131": "located in the administrative territorial entity", "P361": "part of"}
WIKI_NOUNS = {"position", "holder", "member", "group", "spouse", "capital", "ruler", "leader"}   # the wiki world's words
CHANGES = {normalize(v) for v in PROPERTIES.values()} | WIKI_NOUNS


def changes(relation: str) -> bool:
    return normalize(relation or "") in CHANGES


def dated(when) -> bool:
    return bool(when) and any(when.get(k) for k in ("point", "start", "end"))


def undated(relation: str, witnesses) -> bool:
    """A changing relation whose book witnesses do not all date it (no witness: not the books' fact, not ruled here)."""
    ws = list(witnesses or ())
    return changes(relation) and bool(ws) and not all(dated(w.get("when")) for w in ws)


def at_one_point(answer: str, relation: str, subject: str, lang: str = "en") -> str:
    """The answer hedged in time: true at some point of the subject's life, not said for all of it."""
    if lang.startswith("fr"):
        return f"{answer} a été, à un moment, {relation} de {subject}."
    return f"{answer} was a {relation} of {subject} at one point."
