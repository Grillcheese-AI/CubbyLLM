"""standin/question_forms.py -- questions as people ask them (order 0012, Nick): `{wh} {tense} {verb form}
{subject} [in {year}]?` -- "Who founded Rome?", "Where was Napoleon born?", "Who was the ruler of England in
1160?" -- and the (entity, relation, tense) each one asks, read back.

- the wh-word is the answer's type (a person: Who; a place: Where; a date: When; else What / Which);
- the tense is the fact's: ended -> was / did, ongoing -> is / does, scheduled -> will;
- the verb form is the relation's: active ("Who founded Rome?"), passive ("Where was Napoleon born?") or a
  noun ("Who was the ruler of England?").

`FORMS` holds hand-written forms per relation and tense; `tensed` turns a present-tense template (the hdc
set's `relation_template_mapping.csv`, `load_csv`) into all three. `Forms.parse` reads a question back to
(entity, relation, tense): the most specific form that matches wins, and two relations matching equally well
are a refusal (None), never a pick. The year ("in 1160") is the walk's own as-of constraint
(`pipeline._AS_OF`). `time_check` is the host's rule on the answer: the tense and the year asked against the
dates the facts carry -- "Who rules England?" on a reign that ended is not answered. `What is the <relation>
of <entity>?` stays the host's internal plan string only.

Wired: WIRED (`ask.AskLoop` reads tense and year for `time_check`; `TemplateEmitter` proposes a plan where no
model does -- the books bench, `validation/exp_books_bench.py`).
"""
from __future__ import annotations

import csv
import re

__wiring__ = "WIRED"

TENSES = ("past", "present", "future")
# relation -> (kind, {tense: forms}). An EVENT is dated by when it happened (a birth, a founding, a battle):
# asked in the present or the future about a past one, the question is not about that fact. A STATE holds
# from a start to an end (a reign, a capital): asked in the present about one that ended, likewise.
FORMS: dict[str, tuple[str, dict[str, list[str]]]] = {
    "date of birth": ("event", {"past": ["When was {S} born?", "In what year was {S} born?", "What year was {S} born in?"],
                                "present": ["When is {S} born?"], "future": ["When will {S} be born?"]}),
    "place of birth": ("event", {"past": ["Where was {S} born?", "In which city was {S} born?", "In what town was {S} born?"],
                                 "present": ["Where is {S} born?"], "future": ["Where will {S} be born?"]}),
    "date of death": ("event", {"past": ["When did {S} die?", "In what year did {S} die?", "When did {S} pass away?"],
                                "present": ["When does {S} die?"], "future": ["When will {S} die?"]}),
    "place of death": ("event", {"past": ["Where did {S} die?", "In which city did {S} die?"],
                                 "present": ["Where does {S} die?"], "future": ["Where will {S} die?"]}),
    "founded by": ("event", {"past": ["Who founded {S}?", "Who was the founder of {S}?", "By whom was {S} founded?"],
                             "present": ["Who founds {S}?"], "future": ["Who will found {S}?"]}),
    "inception": ("event", {"past": ["When was {S} founded?", "In what year was {S} founded?", "When was {S} established?"],
                            "present": ["When is {S} founded?"], "future": ["When will {S} be founded?"]}),
    "point in time": ("event", {"past": ["When was the {S} fought?", "When did the {S} take place?", "In what year was the {S}?",
                                         "When was the {S}?"],
                                "present": ["When does the {S} take place?"], "future": ["When will the {S} take place?"]}),
    "place of burial": ("state", {"present": ["Where is {S} buried?"], "past": ["Where was {S} buried?"],
                                  "future": ["Where will {S} be buried?"]}),
    "location": ("state", {"present": ["Where is {S} located?", "Where is {S}?"], "past": ["Where was {S} located?", "Where was {S}?"],
                           "future": ["Where will {S} be located?"]}),
    "part of": ("state", {"present": ["What is {S} part of?"], "past": ["What was {S} part of?", "Which empire was {S} part of?"],
                          "future": ["What will {S} be part of?"]}),
    "capital": ("state", {"present": ["What is the capital of {S}?", "Which city is the capital of {S}?"],
                          "past": ["What was the capital of {S}?", "Which city was the capital of {S}?"],
                          "future": ["What will be the capital of {S}?"]}),
    "reign": ("state", {"past": ["When did {S} rule?", "When did {S} reign?", "During which years did {S} reign?"],
                        "present": ["When does {S} reign?"], "future": ["When will {S} reign?"]}),
    "head of state": ("state", {"past": ["Who ruled {S}?", "Who was the ruler of {S}?"],
                                "present": ["Who rules {S}?", "Who is the ruler of {S}?"], "future": ["Who will rule {S}?"]}),
}
_AUX = {"is": ("was", "will be"), "are": ("were", "will be"), "does": ("did", "will"), "do": ("did", "will"),
        "has": ("had", "will have")}
_TENSE_OF = [(re.compile(r"\b(will|shall)\b", re.I), "future"), (re.compile(r"\b(was|were|did|had)\b", re.I), "past"),
             (re.compile(r"\b(is|are|does|do|has|have)\b", re.I), "present")]
DATED = {"date of birth", "date of death", "inception", "point in time", "reign"}
_FRAME = set("who what when where which how is are was were will be does do did has had have the a an of in on at by "
             "to from for with during year years city town what's s".split())
AS_OF = re.compile(r"\b(?:in|as of|during|by|back in)\s+(1[0-9]{3}|2[0-9]{3})\b", re.I)   # pipeline._AS_OF


def tensed(template: str) -> dict[str, str]:
    """A present-tense template in the three tenses, by its first auxiliary: 'Who is the {relation} of
    {S}?' -> 'Who was ...', 'Who will be ...'; 'To which X does {S} belong?' -> '... did ...', '... will ...'."""
    m = re.search(r"\b(is|are|does|do|has)\b", template)
    if not m:
        return {"present": template}
    past, fut = _AUX[m.group(1)]
    swap = lambda w: template[:m.start()] + w + template[m.end():]               # noqa: E731
    return {"present": template, "past": swap(past), "future": swap(fut)}


def load_csv(path: str, forms=None) -> dict:
    """`forms` (default a copy of FORMS) with the hdc template of each relation it already holds, in the three
    tenses ({relation} filled with the edited label when the file gives one, {subject} the slot)."""
    forms = {k: (kind, {t: list(v) for t, v in by.items()}) for k, (kind, by) in (forms or FORMS).items()}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rel = r["relation"].strip()
        if rel in forms:
            t = r["template"].replace("{relation}", (r["edited_relation"] or rel).strip()).replace("{subject}", "{S}")
            for tense, form in tensed(t).items():
                if form not in forms[rel][1].setdefault(tense, []):
                    forms[rel][1][tense].append(form)
    return forms


def tense_of(question: str) -> str | None:
    for rx, tense in _TENSE_OF:
        if rx.search(question):
            return tense
    return None


def _pattern(form: str) -> re.Pattern:
    head, _, tail = form.partition("{S}")
    lit = lambda s: r"\s+".join(re.escape(w) for w in s.split())                  # noqa: E731
    return re.compile(rf"^\s*{lit(head)}\s*(?P<S>.+?)\s*{lit(tail.rstrip('?'))}\s*\??\s*$", re.I)


class Forms:
    def __init__(self, forms: dict | None = None) -> None:
        self.forms = forms or FORMS
        # specificity: the literal words, and half a word more for a form anchored after its slot ("When was
        # {S} founded?" outranks "When was the {S}?" on "When was the Roman Empire founded?")
        self._compiled = sorted(((len(f.replace("{S}", "").replace("?", "").split()) + 0.5 * bool(f.partition("{S}")[2].strip(" ?")),
                                  rel, tense, _pattern(f)) for rel, (_k, by) in self.forms.items()
                                 for tense, fs in by.items() for f in fs), key=lambda x: -x[0])
        self._words = {rel: {w for fs in by.values() for f in fs for w in re.findall(r"[\w']+", f.lower().replace("{s}", " "))
                             if w not in _FRAME} for rel, (_k, by) in self.forms.items()}

    name = "forms"

    def aspect(self, relation: str) -> str:
        """'event' (dated by when it happened) or 'state' (from a start to an end)."""
        return self.forms.get(relation, ("state", {}))[0]

    # as a resolver of the loop's levers 4 and 6: the words a relation is asked with name it ("ruled" ->
    # head of state; "born" -> date and place of birth, which the question's wh-word then narrows by `kind`)
    def relations(self, text: str) -> list[str]:
        t = " ".join(re.findall(r"[\w']+", text.lower()))
        return [rel for rel, words in self._words.items() if t and t in words]

    def kind(self, relation: str) -> str | None:
        """The answer's value kind (`plan_verify.KIND_OF_ASK`): a date or a name; None for a relation it lacks."""
        return None if relation not in self.forms else "date" if relation in DATED else "name"

    def parse(self, question: str) -> tuple[str, str, str] | None:
        """(entity, relation, tense) the question asks -- its year phrase taken off first -- or None: no form,
        or two relations equally specific."""
        q = " ".join(AS_OF.sub(" ", question).split()) if AS_OF.search(question) else question
        best: dict[str, tuple[str, str]] = {}
        top = None
        for n, rel, tense, rx in self._compiled:
            if top is not None and n < top:
                break
            m = rx.match(q)
            if m and m["S"].strip():
                top = n
                best.setdefault(rel, (re.sub(r"^the\s+", "", m["S"].strip().rstrip("?").strip(), flags=re.I), tense))
        if len(best) != 1:
            return None
        rel, (ent, tense) = next(iter(best.items()))
        return ent, rel, tense

    def render(self, relation: str, entity: str, tense: str = "past") -> list[str]:
        return [f.replace("{S}", entity) for f in self.forms.get(relation, ("", {}))[1].get(tense, [])]


def _year(v) -> int | None:
    m = re.match(r"^\s*(-?\d{1,4})(?:-\d\d){0,2}\s*(BC|BCE)?\s*$", str(v or ""))
    if not m:
        return None
    y = int(m.group(1))
    return -y if m.group(2) else y


def time_check(kind: str, tense: str | None, year: int | None, when: dict | None, value: str, now: int) -> str | None:
    """None when the question's time can be the fact's; else why not. `when` is the fact's {start, end, point}
    (an event with none is dated by its value when that is a date). Undated facts are not checked."""
    when = dict(when or {})
    if kind == "event":
        y = _year(when.get("point")) if when.get("point") else _year(value)
        if y is None:                                  # an undated event: only the past may ask about it
            return f"asked in the {tense}, the event carries no date" if tense in ("present", "future") else None
        if tense in ("present", "future") and y <= now:
            return f"asked in the {tense}, it happened in {y}"
        if tense == "past" and y > now:
            return f"asked in the past, it is set for {y}"
        if year is not None and year != y:
            return f"asked about {year}, it happened in {y}"
        return None
    y0, y1 = _year(when.get("start")), _year(when.get("end"))
    if y0 is None and y1 is None:                      # an undated state: now or before, unless a year or the future is asked
        if tense == "future":
            return "asked in the future, nothing is scheduled"
        return None if year is None else f"asked about {year}, the fact carries no dates"
    if tense == "present" and y1 is not None and y1 < now:
        return f"asked in the present, it ended in {y1}"
    if tense in ("present", "past") and y0 is not None and y0 > now:
        return f"asked in the {tense}, it starts in {y0}"
    if tense == "future" and (y0 is None or y0 <= now):
        return "asked in the future, it is not scheduled"
    if year is not None and not ((y0 is None or y0 <= year) and (y1 is None or year <= y1)):
        return f"asked about {year}, it held {y0 if y0 is not None else '?'}-{y1 if y1 is not None else '?'}"
    return None


class TemplateEmitter:
    """An emitter that writes the plan a parsed question asks for -- the program the ask loop reads from its
    model, without one (the bench)."""

    def __init__(self, forms: Forms | None = None) -> None:
        self.forms = forms or Forms()

    def emit(self, prompt: str, max_new_tokens: int = 300, **kw) -> str:
        p = self.forms.parse(prompt)
        if p is None:
            return "no plan\n"
        seed, rel, _tense = p
        return ("function plan(): str {\n" + f'    bind frame, SEED, "{seed}";\n' + f'    bind frame, HOP1, "{rel}";\n'
                + "    return recover(frame, SEED);\n}\n")
