"""Derive every name form from the export itself, instead of typing them out.

The redaction file was hand-written, so it caught "Nicolas Cloutier" and missed `Nic`
x18, `JF` x17, `Sabin` x16 alone, and the surname on its own. A hand list will always
miss the forms people actually use, because those are nicknames and initials nobody
thinks to write down.

So the participant list in the export's own metadata generates the forms:

  full          Nicolas Cloutier
  first / last  Nicolas, Cloutier
  hyphen parts  Jean-Francois -> Jean, Francois
  initials      JF, NC          (two letters, word-bounded)
  short first   Nic             (3-4 letter prefix of the first name)
  unaccented    Jean-Francois alongside Jean-Francois

Every candidate is then CHECKED AGAINST THE CORPUS and only used if it actually
occurs. That matters both ways: an unused form costs nothing to skip, and a form that
fires a lot was going to be a leak. Anything short enough to collide with a real word
or a SKU (`NC`, `SM`) is reported rather than replaced unless it clears a frequency
floor, because silently rewriting `SM` inside a product code damages the corpus for
nothing.

Third parties are not in the metadata - they are people the participants MENTION. Those
come from a small gazetteer of given names, matched only when capitalised mid-sentence,
and they are reported for confirmation rather than assumed.
"""
from __future__ import annotations

import json
import re
import unicodedata

# Given names seen in this corpus' review list plus common Quebec/English firsts. Only
# used capitalised and word-bounded; a miss here is reported, never silently kept.
GAZETTEER = {
    "audrey", "mike", "steve", "isa", "isabelle", "marc", "martin", "sylvain", "eric",
    "stephane", "patrick", "julie", "melanie", "caroline", "guillaume", "maxime",
    "alexandre", "sebastien", "francis", "mathieu", "simon", "david", "daniel",
    "pierre", "luc", "jonathan", "vincent", "olivier", "samuel", "antoine", "gabriel",
    "charles", "michel", "richard", "robert", "andre", "claude", "denis", "jacques",
    "john", "paul", "peter", "chris", "tom", "dave", "steph", "nath", "nathalie",
    "veronique", "genevieve", "josee", "manon", "chantal", "lucie", "sophie", "anne",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def forms_for(name: str) -> list[str]:
    """Every spelling one person is plausibly called, longest first."""
    out = {name}
    parts = [p for p in re.split(r"[\s]+", name) if p]
    for p in parts:
        out.add(p)
        for piece in p.split("-"):            # Jean-Francois -> Jean, Francois
            if len(piece) >= 3:
                out.add(piece)
    if len(parts) >= 2:
        out.add("".join(p[0] for p in parts))                       # NC, JD
        out.add("".join(p[0] for p in parts[0].split("-")) + parts[-1][0])   # JFD
    # A hyphenated first name is itself initialled: Jean-Francois -> JF. This is the
    # form that got missed first time round - `JF` x17 in the corpus while the derived
    # set produced only `JD` from first+last, because "Jean-Francois" is one part.
    if parts and "-" in parts[0]:
        out.add("".join(p[0] for p in parts[0].split("-") if p))
    first = parts[0].split("-")[0] if parts else ""
    for n in (3, 4):                                                # Nic, Nico
        if len(first) > n:
            out.add(first[:n])
    for f in list(out):                                             # unaccented twins
        out.add(strip_accents(f))
    return sorted({f for f in out if len(f) >= 2}, key=len, reverse=True)


def occurrences(form: str, texts: list[str]) -> int:
    pat = re.compile(rf"(?<![\w]){re.escape(form)}(?![\w])", re.I)
    return sum(1 for t in texts if pat.search(t))


def build_map(participants: list[str], texts: list[str], tag_for,
              short_floor: int = 6) -> tuple[dict, list]:
    """Returns (replacements, reported). Short/ambiguous forms need `short_floor` hits
    before they are trusted - `NC` appearing twice is more likely a SKU than a person."""
    repl, reported = {}, []
    for who in participants:
        tag = tag_for(who)
        for form in forms_for(who):
            n = occurrences(form, texts)
            if not n:
                continue
            risky = len(form) <= 2 or (len(form) <= 3 and form.isupper())
            if risky and n < short_floor:
                reported.append((form, n, tag, "too short / too rare to trust"))
                continue
            repl[form] = tag
    return repl, reported


def third_parties(texts: list[str]) -> list[tuple[str, int]]:
    """Given names the participants mention. Reported, never auto-replaced: the corpus
    owner knows which of these are people and which are a brand or a place."""
    seen = {}
    word = re.compile(r"\b([A-Z][a-zà-ÿ]{2,})\b")
    for t in texts:
        for w in word.findall(t):
            if w.lower() in GAZETTEER:
                seen[w] = seen.get(w, 0) + 1
    return sorted(seen.items(), key=lambda kv: -kv[1])


# A JWT that never got its second dot, a bearer token pasted half-copied, a base64 blob
# starting `ey` - the strict three-part pattern misses all of them, and the owner says
# one may still be in there.
LOOSE_TOKEN = re.compile(r"(?<![\w])ey[A-Za-z0-9_\-]{14,}")


def loose_tokens(text: str) -> list[str]:
    return LOOSE_TOKEN.findall(text)
