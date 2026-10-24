"""cubbyllm/reasoning/everyday.py -- everyday wording, read off the question by the host (2026-10-24, Nick: "how far
are we from a useful model?"; then "can you do it").

The real stack (the 450M seats, wiki + hdc, the ask loop offline) answered 2 of 20 common questions. 15 of the 18 it
did not were the wording, not the facts: the model wrote a plan that did not cover the question, or the loop could not
tell which relation "wrote", "painted", "flows through" or "mayor of" asks for. This module is the table from everyday
wording to the relation it asks, as `question_forms.Forms` reads it:

- `EVERYDAY`: hand-written forms, English and French, for the shapes people use ("Who wrote {S}?" asks `author`,
  "What time zone is {S} in?" asks `located in time zone`, "Qui a écrit {S} ?" asks `author`);
- `question_forms.FORMS` (order 0012's forms) and the hdc set's template for EVERY relation it lists
  (`relation_template_mapping.csv`; `question_forms.load_csv` adds only the relations FORMS already holds).

`EverydayForms.parse` reads (entity, relation, tense) off a question; two relations matching equally well are a refusal,
never a pick. `restate` gives the loop what the scene gives it for a spatial question (order 0035): the question in the
host's own grammar ("What is the author of Pride and Prejudice?", its year phrase kept), the relation and the seed. The
loop then walks that plan; the walk, the gate and the VM do exactly what they do for a model's plan, so a form only
proposes. Every question the table does not read goes to the model as before, through the same path, untouched.

Why restate rather than bind the question's own word: the loop resolves an unheld wording with the source's property
table first, and Wikidata's aliases name "wrote" as `notable work` -- "Who wrote Pride and Prejudice?" walked the wrong
relation and found nothing. The table already knows the relation, so it says it in the store's words.

`NOT_READ` leaves to the model every relation whose value changes over a term (the hdc world's TIME_SENSITIVE) or that
holds many values the worlds often keep one of (schools, teams): read by the table, those were spoken wrong.

Every verified answer through a form is one more (question, program) record in the loop's history, which is what the
emitter's next SFT round learns from: the table is scaffolding the trunk is meant to outgrow (the grammar rule).

Measured: `validation/exp_everyday_webq.py` (WebQuestions NA/EU, 2,216, and the stack's 20).

Wired: WIRED (`ask.AskLoop(everyday=)`; ``CB_EVERYDAY=1`` turns it on, off by default, the loop then unchanged).
"""
from __future__ import annotations

import csv
import os
import re

from .question_forms import AS_OF, DATED, FORMS, Forms

__wiring__ = "WIRED"

HDC_TEMPLATES = os.environ.get("CUBBY_HDC_TEMPLATES") or ""      # default: $CUBBY_DATASETS/hdc/relation_template_mapping.csv

# French articles before a name: "de {S}" is written "du {S}", "de la {S}", "de l'{S}", "des {S}" as well
_DE = ("de", "du", "de la", "de l'", "des")
_EN = ("en", "au", "aux", "à")


def _fr(form: str) -> list[str]:
    """A French form with `{de}` / `{en}` before its slot, in every article."""
    if "{de} " in form:
        return [form.replace("{de} {S}", a + ("" if a.endswith("'") else " ") + "{S}") for a in _DE]
    if "{en} " in form:
        return [form.replace("{en} {S}", a + " {S}") for a in _EN]
    return [form]


# relation -> (aspect, {tense: forms}); the relation is named as Wikidata names it, the loop's resolvers map it onto
# each store's own label. Only shapes whose relation is unambiguous are here: "Where is {S} from?" (born there, or a
# citizen?) and "What state is {S} in?" (the parent entity is often a county) are left to the model.
EVERYDAY: dict[str, tuple[str, dict[str, list[str]]]] = {
    "author": ("event", {"past": ["Who wrote {S}?", "Who was {S} written by?", "Who authored {S}?",
                                  "Qui a écrit {S} ?"] + _fr("Qui est l'auteur {de} {S} ?"),
                         "present": ["Who is the author of {S}?"]}),
    "creator": ("event", {"past": ["Who painted {S}?", "Who created {S}?", "Who was {S} painted by?", "Qui a peint {S} ?"]}),
    "composer": ("event", {"past": ["Who composed {S}?", "Who wrote the music for {S}?", "Qui a composé {S} ?"],
                           "present": ["Who is the composer of {S}?"]}),
    "director": ("event", {"past": ["Who directed {S}?", "Who was {S} directed by?", "Qui a réalisé {S} ?"]}),
    "discoverer or inventor": ("event", {"past": ["Who discovered {S}?", "Who invented {S}?", "Who was the inventor of {S}?",
                                                  "Qui a découvert {S} ?", "Qui a inventé {S} ?"]}),
    "architect": ("event", {"past": ["Who was the architect of {S}?"], "present": ["Who is the architect of {S}?"]}),
    "official language": ("state", {"present": [
        "What language is spoken in {S}?", "What languages are spoken in {S}?", "What language do they speak in {S}?",
        "What language do people speak in {S}?", "What languages do they speak in {S}?", "What do they speak in {S}?",
        "What do people speak in {S}?", "What is the official language of {S}?", "What is the language of {S}?",
        "What language is spoken in the {S}?", "Quelle langue parle-t-on {en} {S} ?"] + _fr("Quelle est la langue officielle {de} {S} ?")}),
    "located in time zone": ("state", {"present": [
        "What time zone is {S} in?", "What timezone is {S} in?", "What time zone is {S}?", "What timezone is {S}?",
        "What time zone am I in {S}?", "What timezone am I in {S}?", "What is the time zone of {S}?",
        "What is the time zone in {S}?", "What time zone is in {S}?", "What time zone is {S} under?",
        "What time zone is {S} on?", "What timezone is {S} on?", "What time zone is it in {S}?",
        "What timezone is it in {S}?"]}),
    "country": ("state", {"present": ["What country is {S} in?", "In which country is {S}?", "Which country is {S} in?",
                                      "What country does {S} belong to?", "Dans quel pays se trouve {S} ?"]}),
    "continent": ("state", {"present": ["What continent is {S} in?", "What continent is {S} on?", "On which continent is {S}?",
                                        "Which continent is {S} in?"]}),
    "currency": ("state", {"present": [
        "What currency does {S} use?", "What currency is used in {S}?", "What is the currency in {S}?",
        "What money does {S} use?", "What currency do they use in {S}?", "What kind of money does {S} use?",
        "What currency is used in the {S}?"] + _fr("Quelle est la monnaie {de} {S} ?")}),
    "country of citizenship": ("state", {"present": ["What is {S}'s nationality?", "What is {S} nationality?",
                                                     "What nationality is {S}?", "What is the nationality of {S}?"]}),
    "religion": ("state", {"present": ["What religion is {S}?", "What is {S}'s religion?"], "past": ["What religion was {S}?"]}),
    "cause of death": ("event", {"past": ["What did {S} die of?", "What was {S}'s cause of death?", "What killed {S}?",
                                          "What was the cause of death of {S}?"]}),
    "occupation": ("state", {"present": ["What does {S} do for a living?", "What is {S}'s job?", "What is {S}'s profession?",
                                         "What is the occupation of {S}?"],
                             "past": ["What did {S} do for a living?", "What was {S}'s job?", "What was {S}'s profession?"]}),
    "instrument": ("state", {"present": ["What instrument does {S} play?", "What instruments does {S} play?"],
                             "past": ["What instrument did {S} play?", "What instruments did {S} play?"]}),
    "genre": ("state", {"present": ["What genre is {S}?", "What kind of music does {S} play?", "What type of music does {S} play?",
                                    "What kind of music is {S}?", "What type of music is {S}?"]}),
    "located in or next to body of water": ("state", {"present": [
        "What river flows through {S}?", "What river runs through {S}?", "What river is {S} on?",
        "What body of water is {S} on?", "Which river flows through {S}?"]}),
    "height": ("state", {"present": ["How tall is {S}?"]}),
    "elevation above sea level": ("state", {"present": ["How high is {S}?", "What is the elevation of {S}?"]}),
    "end time": ("event", {"past": ["When did {S} end?", "What year did {S} end?", "In what year did {S} end?"]}),
    "start time": ("event", {"past": ["When did {S} start?", "When did {S} begin?", "What year did {S} start?",
                                      "What year did {S} begin?"]}),
    "basic form of government": ("state", {"present": ["What form of government does {S} have?",
                                                       "What type of government does {S} have?",
                                                       "What kind of government does {S} have?"]}),
    "named after": ("event", {"past": ["Who is {S} named after?", "What is {S} named after?", "Who was {S} named after?"]}),
    "capital": ("state", {"present": ["What is {S}'s capital?", "What is the capital city of {S}?"]
                                     + _fr("Quelle est la capitale {de} {S} ?")}),
    "place of birth": ("event", {"past": ["What city was {S} born in?", "Where was {S} born at?"] + ["Où est né {S} ?",
                                                                                                    "Où est née {S} ?"]}),
    "date of birth": ("event", {"past": ["Quand est né {S} ?", "Quand est née {S} ?"],
                                "present": ["What is {S}'s birthday?", "When is {S}'s birthday?"]}),
    "date of death": ("event", {"past": ["Quand est mort {S} ?", "Quand est morte {S} ?"]}),
    "child": ("state", {"present": ["Who are {S}'s children?", "Who are the children of {S}?"]}),
}
DATES = set(DATED) | {"end time", "start time"}
# Left to the model: a relation whose value changes over a life or a term (the hdc world's TIME_SENSITIVE, by label),
# and one that holds many values the worlds often keep only one of. Measured on WebQuestions (2026-10-24), the table
# spoke "As of 2019, Dallas Cowboys" for Troy Aikman, a high school for "what college did Joe Namath attend?" and a gas
# chamber for "where was Auschwitz located?": each a fact the world holds, none the answer to the question.
NOT_READ = {"head of government", "head of state", "position held", "member of sports team", "member of political party",
            "employer", "league", "owned by", "operator", "headquarters location", "chief executive officer", "head coach",
            "chairperson", "population", "spouse", "member of", "director / manager", "corporate officer", "officeholder",
            "unmarried partner", "residence", "work location", "affiliation", "parent organization",
            "educated at", "location", "home venue"}
# words that frame a question in either language and never name a relation (question_forms._FRAME is English only)
_FRAME = set("who what when where which how is are was were will be does do did has had have the a an of in on at by to "
             "from for with during year years city town empire whom what's s i they people quel quelle quels quelles qui "
             "que quand où est sont a ont le la les l' de du des d' en au aux à se trouve on il elle t".split())


def _merge(into: dict, more: dict) -> dict:
    for rel, (aspect, by) in more.items():
        _a, have = into.setdefault(rel, (aspect, {}))
        for tense, forms in by.items():
            for f in forms:
                if f not in have.setdefault(tense, []):
                    have[tense].append(f)
    return into


def hdc_forms(path: str) -> dict:
    """Every relation's template in the hdc set's csv, in the present tense (its words name the relation)."""
    out: dict = {}
    if not path or not os.path.isfile(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rel = (r.get("relation") or "").strip()
        t = (r.get("template") or "").replace("{relation}", (r.get("edited_relation") or rel).strip()).replace("{subject}", "{S}")
        if rel and "{S}" in t:
            out.setdefault(rel, ("state", {}))[1].setdefault("present", []).append(t)
    return out


def _names_relation(form: str) -> bool:
    """A form must name its relation in words: "When was the {S}?" or "Where is {S}?" names none, and those bare
    shapes stay the profile's (who / what / where / when is X)."""
    return any(w not in _FRAME for w in re.findall(r"[\w']+", form.lower().replace("{s}", " ")))


def everyday_forms(templates: str | None = None) -> dict:
    """FORMS, EVERYDAY and the hdc templates, merged (a copy: FORMS is never changed). A hand-written form wins over an
    hdc template that words another relation the same way (the csv words `capital of` as "What is the capital of X?")."""
    forms = {k: (kind, {t: list(v) for t, v in by.items()}) for k, (kind, by) in FORMS.items()}
    _merge(forms, EVERYDAY)
    if templates is None:
        from cubbyllm.local_paths import expand
        templates = HDC_TEMPLATES or expand("$CUBBY_DATASETS/hdc/relation_template_mapping.csv")
    hand = {f.lower(): rel for rel, (_k, by) in forms.items() for fs in by.values() for f in fs}
    more = {rel: (k, {t: [f for f in fs if hand.get(f.lower(), rel) == rel] for t, fs in by.items()})
            for rel, (k, by) in hdc_forms(templates).items()}
    _merge(forms, more)
    return {rel: (k, {t: [f for f in fs if _names_relation(f)] for t, fs in by.items()})
            for rel, (k, by) in forms.items() if rel not in NOT_READ}


# words a question carries that are not its entity's or its relation's ("what timezone is colorado in right now?",
# "what is the currency of spain called?": WebQuestions' wording), taken off the question's end before the forms read it
_TAIL = re.compile(r"\s+(?:on a map|on the map|wikipedia|right now|now|today|currently|called)(?=\s*\??\s*$)", re.I)
_NOISE = re.compile(r"\s+(?:located|on a map|on the map|wikipedia|right now|now|today|currently)\s*$", re.I)
# "alaska state": a state's name with the word after it
_US = set("alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois "
          "indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri "
          "montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington "
          "wisconsin wyoming".split()) | {"new hampshire", "new jersey", "new mexico", "new york", "north carolina",
                                          "north dakota", "rhode island", "south carolina", "south dakota", "west virginia"}


class EverydayForms(Forms):
    """`Forms` over the merged table, its relation words free of the French frame too."""

    def __init__(self, forms: dict | None = None) -> None:
        super().__init__(forms or everyday_forms())
        self._words = {rel: {w for fs in by.values() for f in fs
                             for w in re.findall(r"[\w']+", f.lower().replace("{s}", " ")) if w not in _FRAME}
                       for rel, (_k, by) in self.forms.items()}

    def kind(self, relation: str) -> str | None:
        return None if relation not in self.forms else "date" if relation in DATES else "name"

    def _match(self, question: str):
        while _TAIL.search(question):
            question = _TAIL.sub("", question)
        m = super()._match(question)
        if not m:
            return None
        ent = m[0]
        while _NOISE.search(ent):
            ent = _NOISE.sub("", ent)
        if ent.lower().endswith(" state") and ent[:-6].strip().lower() in _US:
            ent = ent[:-6].strip()
        return (ent,) + tuple(m[1:]) if ent.strip() else None

    def restate(self, question: str) -> tuple[str, str, str] | None:
        """(the question in the host's grammar, the relation, the seed), or None when no form reads it. The year phrase
        stays ("in 2010" is the walk's as-of constraint); nothing else of the question is dropped but its noise."""
        p = self.parse(question)
        if p is None:
            return None
        ent, rel, _tense = p
        y = AS_OF.search(question)
        return f"What is the {rel} of {ent}{' ' + y.group(0) if y else ''}?", rel, ent
