"""The verified-program loop on ONE natural question, live -- what exp_r11 runs per row, as a service.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this). serve_api mounts it behind
`POST /ask` when started with `--ask`, and every step it takes is an event on `/loop/stream`,
so the control panel (dashboard/control_panel.html, served at /panel) draws the tree as it grows:
question -> plan (the emitter's proposal, the host's aliases) -> walk -> hops -> facts; the fetch
from the source and the gate's verdict on each fact; the answer or the refusal.

The seven invariants hold here exactly as in the benches: the emitter proposes a plan, the host
disposes (covers, lever 4/6/7), the VM verifies, the source's facts pass the gate with provenance,
and the answer is spoken only when a VM-verified chain reaches it -- never from the model.
"""
from __future__ import annotations

import collections
import inspect as _inspect
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TAU_VM = {1: 1.0, 2: 0.4736328125, 3: 0.22021484375}       # exp_r11's thresholds, by hop count

# "who is X?" / "what is X?" / "where is X?" -- no relation asked: a PROFILE of the entity
# (2026-09-14, Nick: "seems like it does not get who is" -- the emitter had planned a relation named
# 'who is'; then: "the model needs to learn to retrieve: if it's a question about who is an entity,
# it needs to call a program that will retrieve the details about the entity and paraphrase it";
# "who, what, where"). The shape is a PROGRAM the emitter can learn to emit -- `bind frame, SEED,
# "x"; bind frame, ASK, "who";` -- and the host executes it: the facts the store holds about X under
# the ask kind's relations, fetched through the gate first when the store has none, each line with its
# provenance; then the talk adapter paraphrases those lines and the host keeps the paraphrase only
# when every name and number in it is in the facts. Until gen 3 emits the program, the host reads the
# shape off the question itself (the same regex builds gen 3's records). Not a walk, not VM-verified --
# a readout of the store, and the record says so (`reason: profile`).
import re as _re
_PROFILE_ASK = _re.compile(r"^\s*(?P<kind>who|what|where)\s+(?:is|was|are|were)\s+(?:the\s+)?(?P<ent>[^?]*?)\s*\??\s*$", _re.I)
# the temporal ask: the entity is a DATE and the relation is 'what happened' -- the same retrieval
# program with ASK 'when' (2026-09-14, Nick: "if I ask what happened in 2026? it refuses it"; it was
# refused as `unknown_relation`, blaming the word 'happened', where the truth is that no source the
# host has indexes events BY DATE -- the wiki world holds 0 facts whose subject is a year)
_WHEN_ASK = _re.compile(r"^\s*(?:what|which)\s+(?:\w+\s+)?(?:happened|occurred|occured|took\s+place|went\s+on)"
                        r"(?:\s+(?:in|on|during|around))?\s+(?:the\s+)?(?P<ent>[^?]*?)\s*\??\s*$", _re.I)
_ASK_ROLE = _re.compile(r'bind\s+frame\s*,\s*ASK\s*,\s*"(?P<kind>who|what|where|when)"\s*;', _re.I)
# WORLD names the store the question is about: absent (or "world") is the world, "self" is the
# loop's own record. A question about what the loop KNOWS and a question about what IS are
# different questions, and the program says which one it is asking -- letting the second be
# answered by the first is how a system starts mistaking its notes for the world.
_WORLD_ROLE = _re.compile(r'bind\s+frame\s*,\s*WORLD\s*,\s*"(?P<world>self|world)"\s*;', _re.I)
_SEED_ROLE = _re.compile(r'bind\s+frame\s*,\s*SEED\s*,\s*"(?P<seed>(?:[^"\\]|\\.)*)"\s*;')
_HOP_ROLE = _re.compile(r'bind\s+frame\s*,\s*(?:HOP\d+|H\d+_\w+)\s*,')
PROFILE_KINDS = {
    "who": ("occupation", "country of citizenship", "date of birth", "place of birth", "date of death", "place of death",
            "position held", "notable work", "field of work", "award received", "educated at", "employer", "member of",
            "member of political party", "spouse", "father", "mother", "sex or gender", "instance of"),
    "what": ("instance of", "subclass of", "part of", "developer", "creator", "author", "manufacturer", "genre",
             "inception", "publication date", "country", "country of origin", "founded by", "headquarters location",
             "official language", "currency", "capital", "population", "occupation", "date of birth"),
    "where": ("instance of", "country", "located in the administrative territorial entity", "capital of", "continent",
              "part of", "capital", "population", "located in or next to body of water", "headquarters location",
              "located on terrain feature", "official language"),
    # 'when' asks for the EVENTS of a date. Only event relations count: a year item's trivia
    # ('instance of: year', 'part of: 2020s') would look like an answer to a question about events
    # and is not one, so the kind never falls back to the other relations -- it refuses instead.
    "when": ("timeline event", "significant event", "notable event", "event"),
}
PROFILE = tuple(dict.fromkeys(r for k in ("who", "what", "where") for r in PROFILE_KINDS[k]))
FALLBACK_KINDS = ("who", "what", "where")          # kinds that may also show any other profile relation
# the resolution hint: the kind's relations, specific first, with the classifiers dropped -- `instance of`
# and `part of` are carried by almost everything and decide nothing
RESOLVE_HINT = {k: tuple(r for r in v if r not in ("instance of", "subclass of", "part of", "sex or gender"))[:8] for k, v in PROFILE_KINDS.items()}
PROFILE_MAX_PER_RELATION = 3
from self_source import LOOP, SelfSource   # noqa: E402

PROFILE_MAX_LINES = 9
# Past this many dated facts, "what happened in 2026?" is not a question with an answer, it is a
# question with a library. Nick, 2026-09-14: "when ambiguous it should either select A based on
# context or B ask the user: do you want to know more about: choices."
NEWS_CLARIFY_MIN = 12
PARAPHRASE_SYSTEM = ("You are a careful writer. You restate given facts in plain sentences and never add anything. "
                     "The facts are written in a database's own vocabulary; say them in ordinary English.")


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


# "what do you know about X", "why did you refuse X", "how often have you been asked about X"
_SELF_ABOUT = _re.compile(
    r"^\s*(?:what|why|how(?:\s+\w+)?)\s+(?:do|did|have|has|are|is|was)\s+you\b[^?]*?\babout\s+"
    r"(?P<ent>[^?]+?)\s*\??\s*$", _re.I)
_SELF_ABOUT2 = _re.compile(
    r"^\s*why\s+(?:did|do)\s+you\s+(?:refuse|reject|fail\s+on)\s+(?P<ent>[^?]+?)\s*\??\s*$", _re.I)
# "what do you know", "how are you doing", "what do you refuse most", "what do you fail at"
_SELF_LOOP = _re.compile(
    r"^\s*(?:what|how)\s+(?:do|did|are|have|has|is)\s+you(?:r)?\b(?:(?!\babout\b)[^?])*\??\s*$", _re.I)


def self_ask(question: str) -> tuple[str, str] | None:
    """(kind, subject) when the question is about the LOOP rather than about the world -- the
    subject being either something the loop was asked about, or `this loop` itself.

    The host reads this shape off the question the same way it reads who/what/where, because
    gen 2 cannot emit `bind frame, WORLD, "self"` yet. When gen 3 can, this becomes a fallback
    and the model asks for its own record directly -- which is the point: a loop that can only
    be introspected by its operator is not aware of its surroundings, it is merely logged."""
    for rx in (_SELF_ABOUT2, _SELF_ABOUT):
        m = rx.match(question or "")
        if m:
            ent = " ".join(m.group("ent").split()).strip(".,")
            if ent and len(ent.split()) <= 6:
                return "what", ent
    if _SELF_LOOP.match(question or ""):
        return "what", LOOP
    return None


def profile_ask(question: str, is_relation=None) -> tuple[str, str] | None:
    """(kind, entity) of a 'who / what / where is X' or 'what happened in <date>' question, or None: a
    question with a relation in it ('the capital of France', "Canada's capital") is a plan for the
    emitter, not a profile.

    'of' alone does not make a plan. 2026-09-14, live: "what is the great pyramid of giza?" refused,
    and so would the Bank of England, the University of Toronto and the Isle of Man -- the guard threw
    out every entity whose own NAME contains 'of'. What separates the two is the head: 'capital of
    France' leads with a relation the store knows, 'great pyramid of giza' does not. `is_relation` is
    the store's own answer to that (AskLoop passes it); without one the old, safe guard stands, because
    guessing which head is a relation is exactly the guess that speaks a wrong answer."""
    m = _WHEN_ASK.match(question)
    if m:
        ent = m.group("ent").strip().strip(",")
        return ("when", ent) if ent and len(ent.split()) <= 5 else None
    m = _PROFILE_ASK.match(question)
    if not m:
        return None
    ent = m.group("ent").strip()
    if not ent or "'s " in ent or ent.endswith("'s") or len(ent.split()) > 6:
        return None
    if " of " in f" {ent} ":
        head = ent.split(" of ", 1)[0].strip()
        head = head[4:].strip() if head.lower().startswith("the ") else head
        if is_relation is None or not head or is_relation(head):
            return None                                  # a relation leads it: the emitter's plan, not a profile
    return m.group("kind").lower(), ent


# "how is B related to A?" -- CLUTRR's own question: the relation the chain of facts between the two
# COMPOSES to, which no single fact states (2026-09-24, the skill library: cubbyllm/reasoning/skills.py).
# Every form reads as "B is the R of A".
_RELATION_ASKS = (
    _re.compile(r"^\s*how\s+(?:is|was|are|were)\s+(?P<b>.+?)\s+related\s+to\s+(?P<a>.+?)\s*\??\s*$", _re.I),
    _re.compile(r"^\s*what\s+(?:relation|relationship|kin)\s+(?:is|was)\s+(?P<b>.+?)\s+to\s+(?P<a>.+?)\s*\??\s*$", _re.I),
    _re.compile(r"^\s*what\s+is\s+(?P<b>.+?)'s\s+(?:relation|relationship)\s+to\s+(?P<a>.+?)\s*\??\s*$", _re.I),
    _re.compile(r"^\s*what\s+is\s+(?:the\s+)?(?:relation|relationship)\s+of\s+(?P<b>.+?)\s+to\s+(?P<a>.+?)\s*\??\s*$", _re.I),
)


def relation_ask(question: str) -> tuple[str, str] | None:
    """(A, B) of a question asking how B is related to A, or None."""
    for rx in _RELATION_ASKS:
        m = rx.match(question or "")
        if m:
            a, b = (" ".join(m.group(k).split()).strip(".,") for k in ("a", "b"))
            if a and b and len(a.split()) <= 6 and len(b.split()) <= 6:
                return a, b
    return None


def profile_program(program: str) -> tuple[str, str] | None:
    """(kind, seed) when an emitted program is the retrieval shape -- SEED and ASK bound, no hop."""
    a, s = _ASK_ROLE.search(program or ""), _SEED_ROLE.search(program or "")
    if not a or not s or _HOP_ROLE.search(program or ""):
        return None
    return a.group("kind").lower(), s.group("seed").replace('\\"', '"')


def program_world(program: str) -> str:
    """Which store an emitted program asks about: 'self' or 'world' (the default)."""
    m = _WORLD_ROLE.search(program or "")
    return m.group("world").lower() if m else "world"


def cot_profile(seed: str, kind: str) -> str:
    """The retrieval program, in the emitter's CotPlan form: what gen 3 learns to emit for this shape."""
    esc = seed.replace("\\", "\\\\").replace('"', '\\"')
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n"
            "        create frame: number;\n"
            f'        bind frame, SEED, "{esc}";\n        bind frame, ASK, "{kind}";\n'
            "        return recover(frame, SEED);\n    }\n}\n")


def cot_self(seed: str, kind: str) -> str:
    """The introspection program: the retrieval shape with WORLD bound to "self". What gen 3
    learns to emit when the question is about the loop rather than about the world."""
    esc = seed.replace("\\", "\\\\").replace('"', '\\"')
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n"
            "        create frame: number;\n"
            f'        bind frame, SEED, "{esc}";\n        bind frame, ASK, "{kind}";\n'
            '        bind frame, WORLD, "self";\n'
            "        return recover(frame, SEED);\n    }\n}\n")


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


# ---------------------------------------------------------------- the claim check (H-E10)
# The value check keeps a reply to the one value the host returned: safe, and why Cubby answers like a form.
# The claim check lets a reply say more of the block -- "It began in 1914 in Sarajevo, when Gavrilo Princip
# shot the archduke" -- and still refuses every statement the block does not make: each thing the reply
# names from the block must be tied, in the same sentence, to something the block ties it to (or to the
# event asked about, which a reply may leave as "it"); a sentence that says one thing led to another must
# say it the way round the block does; a year said to begin or end something must be that line's year.
FORWARD_CUES = ("led to", "lead to", "leads to", "leading to", "caused", "causes", "resulted in", "results in",
                "brought about", "brings about", "triggered", "sparked", "gave rise to", "set off", "prompted",
                "provoked", "paved the way for", "contributed to", "precipitated", "opened the way to", "produced")
BACKWARD_CUES = ("because of", "due to", "as a result of", "in response to", "was caused by", "were caused by",
                 "was triggered by", "was sparked by", "was prompted by", "was provoked by", "stemmed from",
                 "came from", "resulted from", "followed from", "was a response to", "were a response to",
                 "in the wake of", "was brought about by", "grew out of")
_CAUSAL_REL = {"caused": 1, "contributed to": 1, "precursor of": 1, "response to": -1}   # +1: e -> v, -1: v -> e
_BEGIN = _re.compile(r"\b(?:began|begin|begins|started|start|starts|broke out|opened|commenced)\b", _re.I)
_END = _re.compile(r"\b(?:ended|end|ends|finished|concluded|closed|was over)\b", _re.I)
_SENT = _re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"'(])")
_PRONOUN = _re.compile(r"\b(?:it|this|that|these|they|its|their|there)\b", _re.I)


def _aliases(value: str) -> list[str]:
    """The ways a reply may write a block string: itself; an ISO date as 'June 28, 1914', '28 June 1914',
    'June 1914' or the year alone (a year is a less precise statement of the same date)."""
    v = str(value)
    m = _re.fullmatch(r"(-?)(\d{4})-(\d{2})(?:-(\d{2}))?", v.strip())
    if not m:
        return [v]
    y, mo = m.group(2).lstrip("0") or "0", int(m.group(3))
    out = [v]
    if 1 <= mo <= 12:
        name = _MONTHS[mo - 1]
        if m.group(4):
            d = str(int(m.group(4)))
            out += [f"{name} {d} {y}", f"{d} {name} {y}"]
        out.append(f"{name} {y}")
    out.append(y)
    return out


def _mentions(text: str, strings) -> list[tuple[int, int, frozenset]]:
    """(start, end, the block strings it may be) for every block string the text writes, longest first, not
    overlapping. One form can be several strings: '1914' is the July Crisis's year, the war's start year and
    the year of the assassination's date -- the sentence decides which, so all are kept."""
    t = f" {_vnorm(text)} "
    forms: dict[str, set] = {}
    for s in strings:
        for a in _aliases(s):
            if _vnorm(a):
                forms.setdefault(_vnorm(a), set()).add(s)
    found: list[tuple[int, int, frozenset]] = []
    for form in sorted(forms, key=len, reverse=True):
        for m in _re.finditer(r"(?<= )" + _re.escape(form) + r"(?= )", t):
            if not any(m.start() < b and a < m.end() for a, b, _ in found):
                found.append((m.start(), m.end(), frozenset(forms[form])))
    return sorted(found, key=lambda f: f[0])


def _cue(sentence: str, strings) -> tuple[int, int] | None:
    """(+1 | -1, position in the normalised sentence) of the first causal cue, if any."""
    t = f" {_vnorm(sentence)} "
    best = None
    for cues, sign in ((BACKWARD_CUES, -1), (FORWARD_CUES, 1)):          # "was caused by" before "caused"
        for c in cues:
            i = t.find(f" {_vnorm(c)} ")
            if i >= 0 and (best is None or i < best[1] or (i == best[1] and sign == -1)):
                best = (sign, i)
    return best


def claim_check(answer: str, lines: list, returned: list[str], entity: str | list[str]) -> tuple[bool, list[str]]:
    """Every statement the reply makes from the block is one the block makes. `lines`: the block as
    (entity, relation, value) triples; `entity`: the event asked about (its name, or all the names it is
    served under). The reply must state a returned value (as `value_check`). -> (ok, what failed)."""
    asked = {_vnorm(e) for e in ([entity] if isinstance(entity, str) else entity) if _vnorm(e)}
    problems: list[str] = []
    if returned and not any(has_value(answer, v) for v in returned):
        problems.append("returned value missing")
    triples = [(str(e), str(r), str(v)) for e, r, v in lines]
    strings = {x for e, _, v in triples for x in (e, v)} | set(asked)
    for sent in _SENT.split(answer or ""):
        ms = _mentions(sent, strings)
        t = f" {_vnorm(sent)} "
        said = {_vnorm(s) for _, _, ss in ms for s in ss}
        present = said | asked                           # the event asked about may stay "it"
        verbs = sorted([(m.start(), "begin") for m in _BEGIN.finditer(t)] + [(m.start(), "end") for m in _END.finditer(t)])
        for at_m, _, ss in ms:
            if any(_vnorm(s) in asked for s in ss):
                continue
            fits = []                                    # the readings of this mention the sentence supports
            for s in ss:
                n = _vnorm(s)
                ties = [(e, r, v) for e, r, v in triples if n in (_vnorm(e), _vnorm(v))
                        and (_vnorm(v) if n == _vnorm(e) else _vnorm(e)) in present]
                rels = {r for e, r, v in ties if _vnorm(v) == n}
                gov = next((kind for p, kind in reversed(verbs) if at_m - 40 <= p < at_m), None)
                if gov == "begin" and rels and rels <= {"end year"}:
                    continue                             # an end year said as a beginning
                if gov == "end" and rels and rels <= {"start year"}:
                    continue
                if ties:
                    fits.append(s)
            if not fits:
                shown = sorted(ss, key=len)[0]
                problems.append(f"'{shown}' is not tied, as said, to what the sentence says")
        cue = _cue(sent, strings)
        if cue:
            sign, at = cue
            events = {_vnorm(e) for e, _, _ in triples} | asked      # what can cause: events, not a year or a place
            left = {_vnorm(s) for a, _, ss in ms if a < at for s in ss if _vnorm(s) in events} or \
                (asked if _PRONOUN.search(t[:at]) else set())
            right = {_vnorm(s) for a, _, ss in ms if a > at for s in ss if _vnorm(s) in events} or \
                (asked if _PRONOUN.search(t[at:]) else set())
            edges = set()
            for e, r, v in triples:
                d = _CAUSAL_REL.get(r)
                if d:
                    a, b = (_vnorm(e), _vnorm(v)) if d == 1 else (_vnorm(v), _vnorm(e))
                    edges.add((a, b))
            for x in left:
                for y in right:
                    if x == y:
                        continue
                    cause, effect = (x, y) if sign == 1 else (y, x)
                    if (cause, effect) not in edges and not ({x, y} <= asked):
                        problems.append(f"the block does not say '{cause}' led to '{effect}'")
    return (not problems), list(dict.fromkeys(problems))


def talk_reply(returned: list[str], draft: str | None, facts: list[str], entity: str,
               others: list[str], lines: list | None = None, check: str = "value") -> tuple[str | None, str]:
    """What the host says for a question over a facts block. The VM returned nothing -> the host says
    ABSENT_REPLY itself (the model is not asked to decline). Otherwise the draft is spoken only if the
    name-and-number guard AND the value check pass -- or, with check="claims" and the block's `lines`,
    the claim check in its place (H-E10). -> (reply or None to refuse, reason)."""
    if not returned:
        return ABSENT_REPLY, "vm_empty"
    if not draft:
        return None, "no_draft"
    ok, bad = grounded_prose(draft, facts, entity)
    if not ok:
        return None, "ungrounded: " + ", ".join(bad[:4])
    if check == "claims" and lines is not None:
        ok, bad = claim_check(draft, lines, returned, entity)
        if not ok:
            return None, "claim: " + "; ".join(bad[:3])
        return draft, "spoken"
    ok, bad = value_check(draft, returned, others, entity)
    if not ok:
        return None, "value: " + ", ".join(bad[:4])
    return draft, "spoken"


def parse_choose(raw) -> dict[int, str]:
    """The asker's picks among offered branches, as the walk takes them: {hop: normalized object}.
    Accepts one {"hop", "object"} or a list of them (the panel sends back what a choice carried);
    anything malformed is dropped -- a choice can only narrow the store's options, so a bad one
    costs nothing but the narrowing."""
    from cubbyllm.reasoning.planner import normalize
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    out: dict[int, str] = {}
    for c in items:
        try:
            out[int(c["hop"])] = normalize(str(c["object"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def branch_clarify(plan, res, prior: dict[int, str] | None = None) -> dict | None:
    """The choices an ambiguous hop's branches offer the asker (graph-of-thought, 2026-09-24).

    Offered only when the ambiguity is MID-chain: which of Marie's two citizenships the question
    means is the asker's to say, and the rest of the chain is still walked and certified after
    they say it (chosen, not invented). A split at the LAST hop is a set of answers, not a choice
    of path -- offering it would let the asker pick the answer and have it come back 'verified'
    (the poisoned-capital case), so there the refusal names the values and asks nothing."""
    branches = getattr(res, "branches", None) or []
    if res.reason != "ambiguous_hop" or not branches or plan is None:
        return None
    first = min(b["via"][0]["hop"] for b in branches if b.get("via"))
    if first >= plan.n_hop - 1:
        return None
    rel = plan.tail.rsplit(" of ", 1)[0] if first == 0 else (plan.relations[first] or "")
    ent = (plan.tail.rsplit(" of ", 1)[1] if (first == 0 and " of " in plan.tail)
           else (res.trace[first - 1].triple.obj if 0 < first <= len(res.trace) and res.trace[first - 1].triple else ""))
    kept = [{"hop": h, "object": o} for h, o in sorted((prior or {}).items()) if h != first]
    groups: dict[str, list[dict]] = {}
    for b in branches:
        groups.setdefault(b["via"][0]["object"], []).append(b)
    choices = []
    for obj, bs in groups.items():
        answers = sorted({b["answer"] for b in bs if b["status"] == "verified"})
        if len(answers) == 1 and all(b["status"] == "verified" for b in bs):
            detail = f"leads to {answers[0]}"
        elif answers:
            detail = "leads to " + " or ".join(answers)
        elif any(b["status"] == "pruned" for b in bs):
            detail = "the store has no way on from " + next(b["stalled"] for b in bs if b["status"] == "pruned")
        else:
            detail = "the VM did not certify this path"
        choices.append({"label": obj, "detail": detail, "item": None,
                        "choose": kept + [{"hop": first, "object": obj}]})
    return {"question": f"{ent} has more than one {say(rel)}. Which one do you mean?",
            "hop": first, "choices": choices}


def _learned(p) -> dict:
    """A gated fact as the record keeps it: the gate's verdict and the provenance the sleep cycle needs to
    make it durable (when it was fetched, the store's snapshot hash before the write, what it clashed with)."""
    return {"fact": p.fact, "status": p.status, "source": p.source, "entity": p.entity,
            "fetched_at": getattr(p, "fetched_at", None), "snapshot_before": getattr(p, "snapshot_before", None),
            "clash": getattr(p, "clash", None)}


class AskLoop:
    """`ask(question)` -> the loop's record. `emitter` is any object with `.emit(prompt, max_new_tokens=)`
    (the serving emitter, shared); `world` a FactStore with a TripleIndex (the wiki world when None);
    `source` 'wikidata' | 'wikidata-offline' | 'lfm' | a Source object; `lexicon` adds lever 5."""

    def __init__(self, emitter, world=None, source="wikidata", lexicon: bool = True, exe: str | None = None,
                 max_new: int = 300, lfm_gguf: str | None = None, run_fn=None, tau_profile: float = TAU_VM[1],
                 news=None, history_path: str | None = None, web=None, skills=None):
        from cubbyllm.bridges import cubelang_client as cc
        from cubbyllm.reasoning.plan_verify import StoreRelations
        self.emitter = emitter
        if world is None:
            import wikikg as wk
            wk.ensure_data(("triplets",))
            world = wk.wiki_world()
        self.world = world
        self.known = StoreRelations(world.index._seen)
        if isinstance(source, str):
            if source.startswith("wikidata"):
                from sources import WikidataSource
                source = WikidataSource(offline=source.endswith("offline"), use_choices=True)   # serving learns what askers mean
            elif source == "lfm":
                from lfm_source import LfmSource
                source = LfmSource(lfm_gguf or str(ROOT / "standin" / "models" / "LFM2.5-2.6B.Q4_K_M.gguf"))
            else:
                raise ValueError(f"unknown source {source!r}")
        self.source = source
        # The date source. A `when` ask is the one question the wiki world provably cannot answer
        # (measured: 0 facts whose subject is a year), so it gets its own source -- and only a
        # `when` ask ever consults it. `news="rss"` builds the default feed set.
        if news == "rss":
            from news_source import NewsSource
            news = NewsSource()
        self.news = news
        # the web (2026-09-24): asked only after the source above has nothing more for the entity the walk
        # stalled on, and what it finds is held until two independent sites agree (standin/web_source.py)
        self.web = web
        self.fallbacks = [web] if web is not None else []
        if web is not None and getattr(web, "wikidata", None) is None and hasattr(source, "resolve"):
            web.wikidata = source                        # a Wikipedia hit hands over to Wikidata's own claims
        if web is not None and getattr(web, "containment", None) is None and hasattr(source, "_claims"):
            web.containment = source                     # which places contain which: Warsaw vouches for Poland
        if web is not None and getattr(web, "_aliases", None) is None and getattr(source, "aliases", None) is not None:
            web._aliases = source.aliases                # one property table for both
        self.resolvers = []
        if lexicon:
            from cubbyllm.reasoning.lexicon import Lexicon
            self.resolvers.append(Lexicon())
        self._vm = run_fn                                # tests: a stand-in VM; otherwise the resident CubeLang session
        self.tau_profile = float(tau_profile)            # a profile line is spoken only when the VM recovers it at this similarity
        self.session = None if run_fn else cc.CubelangSession(exe=exe)
        self.max_new = int(max_new)
        self.calls = collections.Counter()
        self.history: list[dict] = []
        # every record, one JSON line each, for the night's sleep cycle (cubbyllm/reasoning/sleep.py):
        # the day's refusals are its curriculum and the day's accepted facts what it makes durable
        self.history_path = pathlib.Path(history_path) if history_path else None
        # the skill library (cubbyllm/reasoning/skills.py): composition rules the sleep cycle adopted, read
        # from its ledger; empty without one, and then a relation ask is answered only by a direct fact
        from cubbyllm.reasoning.skills import Library
        self.skills = skills if isinstance(skills, Library) else Library(skills)
        self._lock = threading.Lock()                    # one question at a time: the emitter and the VM session are not re-entrant

    def _is_relation(self, head: str) -> bool:
        """Does the store know `head` as a relation? `self.known` is StoreRelations -- a set over parsed
        `Triple.rel`, which is the question being asked. (The first cut of this read `index._seen`, which
        holds whole FACT SENTENCES, not relations: every membership test was False, so every entity whose
        name contains 'of' stayed a profile AND every real plan became one too. Live, 2026-09-14.)"""
        if not head:
            return False
        try:
            if head in self.known:
                return True
        except Exception:                                # noqa: BLE001
            return True                                  # a store that cannot answer keeps the old, safe guard
        for res in self.resolvers:                       # the lexicon's own names for a relation count too
            fn = getattr(res, "relations", None)
            if not callable(fn):
                continue
            try:
                if any(c in self.known for c in (fn(head) or ())):
                    return True
            except Exception:                            # noqa: BLE001 -- a resolver that cannot answer is not a yes
                continue
        return False

    def _run_fn(self, source: str, fn: str) -> dict:
        self.calls["vm"] += 1
        return self._vm(source, fn) if self._vm else self.session.run(source, fn=fn)

    def _keep(self, rec: dict) -> None:
        """Keep the record in memory and, with a history path, append it as one JSON line: an id, the
        time, the store's snapshot hash, and for each learned fact its relation (as the index split it
        now, so a reload splits it the same way) and WHEN the source said it was true. Inside an ask,
        the record gets its thought graph first, so the history keeps how the answer came about."""
        g = getattr(self, "_graph", None)
        if g is not None and "graph" not in rec:
            try:
                rec["graph"] = g[0].graph(g[1])
            except Exception as e:                       # noqa: BLE001 -- a record is kept even if its graph is not
                rec["graph"] = {"error": str(e)[:200]}
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec.setdefault("ts", ts)
        rec.setdefault("id", f"{ts}-{len(self.history) + 1}")   # the handle an asker's feedback names
        self.history.append(rec)
        if self.history_path is None:
            return
        try:
            import json as _json

            from cubbyllm.reasoning.learn import snapshot
            from cubbyllm.reasoning.planner import parse_fact
            known = getattr(getattr(self.world, "index", None), "_reused", None)
            times = getattr(self.world, "times", None) or {}
            for l in rec.get("learned") or []:
                key = " ".join(str(l.get("fact", "")).split())
                if l.get("status") == "accepted" and "rel" not in l:
                    t = parse_fact(key, known=known)
                    if t is not None:
                        l["rel"] = t.rel
                if key in times:
                    l.setdefault("time", times[key])
            line = dict(rec, store_snapshot=snapshot(self.world))
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception as e:                           # noqa: BLE001 -- a history that cannot be written never costs an answer
            print(f"[ask] history not written: {type(e).__name__}: {e}", flush=True)

    def load_learned(self, path) -> int:
        """Facts the sleep cycle made durable (its ``learned_facts.jsonl``), back into the store at boot,
        with their relation declared, their source as provenance and their time. What the loop learned
        yesterday is looked up today like any other fact -- and verified by the VM like any other."""
        from cubbyllm.reasoning.sleep import load_facts
        p = pathlib.Path(path)
        if not p.exists():
            return 0
        idx = getattr(self.world, "index", None)
        prov = getattr(self.world, "provenance", None)
        times = getattr(self.world, "times", None)
        n = 0
        for key, r in load_facts(p).items():
            rel = r.get("rel")
            if rel:
                if idx is not None and hasattr(idx, "declare_relation"):
                    idx.declare_relation(rel)
                if hasattr(self.known, "declare"):
                    self.known.declare(rel)
            if self.world.add(key):
                n += 1
                if hasattr(self.known, "add"):
                    self.known.add(key)
            if prov is not None:
                prov.setdefault(key, r.get("source") or "learned")
            if times is not None and r.get("time"):
                times.setdefault(key, r["time"])
        return n

    def ask(self, question: str, item: str | None = None, asker: str | None = None, choose=None) -> dict:
        """One question through the loop; the record it returns (and keeps) carries the run's thought
        graph (`cubbyllm.reasoning.graph`), built from the events the ask emitted."""
        from cubbyllm.reasoning.graph import Collector
        with Collector() as col:
            self._graph = (col, " ".join(question.split()))
            try:
                return self._ask(question, item=item, asker=asker, choose=choose)
            finally:
                self._graph = None

    def _ask(self, question: str, item: str | None = None, asker: str | None = None, choose=None) -> dict:
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import learn_and_answer
        from cubbyllm.reasoning.planner import QuestionPlan, normalize
        from eval_emitter_vm import strip_fences
        from exp_r9_matched_pairs import emitted_plan
        t0 = time.perf_counter()
        question = " ".join(question.split())
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": None, "plan": None, "seed": None,
                     "learned": [], "entities": [], "aliased": [], "snapped": None, "trace": [], "wall_s": 0.0}
        with self._lock:
            self.calls["asked"] += 1
            sa = self_ask(question)                      # about the LOOP, not about the world
            if sa:
                rec = self.introspect(question, sa[0], sa[1], t0, how="the question's shape (about you)")
                self._keep(rec)
                return rec
            ra = relation_ask(question)                  # "how is B related to A?": composed, not looked up
            if ra:
                rec = self.relation(question, ra[0], ra[1], t0)
                self._keep(rec)
                return rec
            pa = profile_ask(question, self._is_relation)   # the host reads the shape off the question (gen 2 cannot emit it yet)
            if pa:
                rec = self.profile(question, pa[0], pa[1], t0, how="the question's shape (who / what / where / when)", item=item, asker=asker)
                self._keep(rec)
                return rec
            asked, _year = strip_as_of(question)         # the year is a constraint, not a relation
            try:
                raw = self.emitter.emit(asked, max_new_tokens=self.max_new)
                if program_world(strip_fences(raw)) == "self":
                    pp = profile_program(strip_fences(raw))
                    if pp:
                        rec = self.introspect(question, pp[0], pp[1], t0, how='the emitter\'s program (WORLD "self")')
                        self._keep(rec)
                        return rec
                program = strip_fences(raw)
                pp = profile_program(program)            # the emitter wrote the retrieval program: SEED + ASK, no hop
                if pp:
                    rec = self.profile(question, pp[0], pp[1], t0, how="the emitter's program (ASK role)", item=item, asker=asker)
                    self._keep(rec)
                    return rec
                ep = emitted_plan(program, normalize)
            except Exception as e:                       # noqa: BLE001 -- the proposer failed; the loop has nothing to dispose
                ep = None; rec["reason"] = f"emit_error: {str(e)[:120]}"
            if not ep or not ep[0] or not ep[1]:
                rec["reason"] = rec["reason"] or "no_plan"
                qid = ev.emit("question", text=question, source=getattr(self.source, "name", None))
                ev.emit("answer", qid, answer=None, verified=False, reason=rec["reason"])
                rec["wall_s"] = round(time.perf_counter() - t0, 3); self._keep(rec); return rec
            rels, seed = ep
            plan = QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))
            rec["plan"], rec["seed"] = rels, seed
            lr = learn_and_answer(question, lambda q, k: [], self._run_fn, store=self.world, known=self.known,
                                  source=self.source, tau_vm=TAU_VM.get(plan.n_hop, 0.2202), top_k=3, max_repairs=1,
                                  plan=plan, resolvers=self.resolvers, choose=parse_choose(choose) or None,
                                  fallbacks=self.fallbacks)
        rec.update(answer=lr.result.answer if lr.result.verified else None, verified=bool(lr.result.verified),
                   reason=None if lr.result.verified else lr.result.reason,
                   learned=[_learned(p) for p in lr.learned],
                   entities=list(lr.entities), aliased=[list(a) for a in lr.aliased], snapped=list(lr.snapped) if lr.snapped else None,
                   trace=[h.fact for h in lr.result.trace if h.fact], refused=lr.result.refused,
                   branches=lr.result.branches, chose=choose or None,
                   clarify=branch_clarify(lr.plan or plan, lr.result, parse_choose(choose)),
                   wall_s=round(time.perf_counter() - t0, 3))
        self._keep(rec)
        return rec

    def feedback(self, record_id: str, verdict: str | None = None, relation: str | None = None,
                 asker: str | None = None) -> dict:
        """The asker on one of the loop's records: `verdict` right / wrong, and for a relation ask the relation
        they say it is. Written to the history as its own line, which the sleep cycle folds into the record:
        a spoken answer called wrong is a defect, and a relation the asker states makes the ask an episode
        for the skill library -- the one way a live refusal teaches a rule. The loop's own answers never do.
        Several askers are needed: the library counts distinct askers as support, so one asker alone cannot
        put a rule in."""
        from cubbyllm.reasoning.planner import normalize
        rec = next((r for r in reversed(self.history) if r.get("id") == record_id), None)
        if verdict not in (None, "right", "wrong"):
            raise ValueError("verdict is 'right' or 'wrong'")
        if relation is not None and rec is not None and rec.get("kind") != "relation":
            raise ValueError("a relation corrects a relation ask ('how is B related to A?') only")
        stated = normalize(relation) if relation else None
        if stated and rec is not None and rec.get("verified"):
            verdict = "right" if normalize(rec.get("answer") or "") == stated else "wrong"
        if verdict is None and stated is None:
            raise ValueError("say right / wrong, or the relation it is")
        line = {"feedback_for": record_id, "verdict": verdict, "relation": stated, "asker": asker,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if rec is not None:
            if verdict:
                rec["feedback"] = verdict
            if stated:
                rec["gold"] = stated
        if self.history_path is not None:
            import json as _json
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(line, ensure_ascii=False) + "\n")
        return {"ok": True, "record": record_id, "known": rec is not None, "verdict": verdict, "relation": stated,
                "episode": bool(stated and (rec is None or rec.get("kind") == "relation"))}

    # -- the relation ask --------------------------------------------------------------------
    def relation(self, question: str, a: str, b: str, t0: float) -> dict:
        """How is B related to A: every chain of held facts from A to B, each composed by the skill
        library over every bracketing and certified in the VM with its facts; one relation is spoken only
        if every path that composes agrees (`skills.relate`). The loop's own answers never teach the
        library -- only gold or an asker does, at night."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        from cubbyllm.reasoning.skills import MAX_HOPS, MAX_PATHS, relate, relate_back
        qid = ev.emit("question", text=question, source="skills")
        learned: list = []
        asked: list = []
        paths, overflow = self.world.index.paths(a, b, max_hops=MAX_HOPS, limit=MAX_PATHS)
        if not paths and not overflow and hasattr(self.source, "facts"):
            # nothing joins them in the store: ask the source for the two families, through the gate
            learned, asked = self._fetch_family(a, b)
            paths, overflow = self.world.index.paths(a, b, max_hops=MAX_HOPS, limit=MAX_PATHS)
        res = relate(paths, self.skills, self._run_fn, tau_vm=TAU_VM, chunk=2, overflow=overflow)
        if not res["verified"] and res["reason"] in ("no_path", "no_rule") and self.skills.inverses:
            # the store may only lead the other way, from B to A: compose that, and turn it round with the
            # inverse rule for B's sex (skills.relate_back) -- certified with B's sex fact in the VM
            back, back_over = self.world.index.paths(b, a, max_hops=MAX_HOPS, limit=MAX_PATHS)
            if back and not back_over:
                turned = relate_back(back, self._sex_of(b), self.skills, self._run_fn, tau_vm=TAU_VM, chunk=2)
                if turned["verified"]:
                    res = turned
        first = res["paths"][0]["relations"] if res["paths"] else []
        ev.emit("plan", qid, seed=normalize(a), relations=first, tail=normalize(b), n_hop=len(first),
                how=f"relation ask: {len(paths)} path(s) from {a} to {b}, composed by {len(self.skills)} rule(s)")
        rec: dict = {"question": question, "kind": "relation", "answer": res["answer"], "verified": res["verified"],
                     "reason": res["reason"], "plan": first, "seed": normalize(a), "target": normalize(b),
                     "statement": f"{b} is the {res['answer']} of {a}" if res["verified"] else None,
                     "paths": [p["relations"] for p in res["paths"]], "relation_paths": res["paths"],
                     "direction": res.get("direction", "forward"),
                     "trace": [f for p in res["paths"] if p["status"] == "derived" for f in p["facts"]],
                     "learned": [_learned(p) for p in learned], "entities": asked,
                     "aliased": [], "snapped": None, "wall_s": round(time.perf_counter() - t0, 3)}
        ev.emit("answer", qid, answer=rec["answer"], verified=rec["verified"], reason=rec["reason"])
        return rec

    FAMILY_FETCH_MAX = 40                                # source calls one relation ask may spend

    def _sex_of(self, person: str):
        """(sex, the stored fact that says it) for a person the store knows the sex of, else None."""
        from cubbyllm.reasoning.planner import normalize
        for _f, t in self.world.index._by_subj.get(normalize(person), []):
            if t is not None and normalize(t.rel) == "sex or gender" and normalize(t.obj) in ("male", "female"):
                return normalize(t.obj), t
        return None

    def _fetch_family(self, a: str, b: str) -> list:
        """The two families from the source: each person's father, mother, children, siblings and spouses,
        then each of THOSE people once -- for their sex (a male child is a son) and their own family, which
        is what a path of up to four hops needs. Every fact passes the gate with its provenance; the edge back
        is the one Wikidata itself declares (father/mother inverse of child, sibling and spouse symmetric,
        `skills.family_facts`). Returns (the gate's records of what was accepted, the people asked about)."""
        from cubbyllm.reasoning.learn import gate
        from cubbyllm.reasoning.planner import normalize, parse_fact
        from cubbyllm.reasoning.skills import FAMILY, family_facts
        sex: dict[str, str] = {}
        edges: list[tuple[str, str, str]] = []
        budget = [self.FAMILY_FETCH_MAX]
        asked: list[str] = []

        def family(entity: str, via: str | None) -> list[tuple[str, str, str]]:
            if budget[0] <= 0:
                return []
            budget[0] -= 1
            asked.append(entity)
            try:
                try:
                    got = self.source.facts(entity, via=via) if via else self.source.facts(entity)
                except TypeError:                        # a source that takes the entity alone
                    got = self.source.facts(entity)
            except Exception:                            # noqa: BLE001 -- a source that fails states nothing
                return []
            out = []
            for t in got or []:
                t = parse_fact(t) if isinstance(t, str) else t
                if t is None:
                    continue
                r = normalize(t.rel)
                if r == "sex or gender" and normalize(t.obj) in ("male", "female"):
                    sex[normalize(t.subj)] = normalize(t.obj)
                elif r in FAMILY:
                    out.append((t.subj, r, t.obj))
            return out

        for person in (a, b):
            ring = family(person, None)
            edges += ring
            for x, r, y in ring:
                edges += family(y, f"{y} is the {r} of {x}")
        name = getattr(self.source, "name", "source")
        prov = getattr(self.world, "provenance", None)
        learned = []
        for x, r, y in edges:
            for f in family_facts(x, r, y, {x: sex.get(normalize(x)), y: sex.get(normalize(y))}):
                p = gate(f, self.world, name, x)
                if p.status != "accepted":
                    continue
                self.world.add(f)
                if hasattr(self.known, "add"):
                    self.known.add(f)
                if prov is not None:
                    prov.setdefault(f, name)
                learned.append(p)
        return learned, asked

    # -- the profile ask ---------------------------------------------------------------------
    def facts_about(self, entity: str) -> list:
        """The Triples the store holds with `entity` as subject (the index's subject lookup)."""
        from cubbyllm.reasoning.planner import normalize
        index = getattr(self.world, "index", None)
        by = getattr(index, "_by_subj", None)
        if by is None:
            return []
        return [t for _f, t in by.get(normalize(entity), []) if t is not None]


    def _fetch(self, qid: int, entity: str, hint: list[str], rec: dict, item: str | None = None, asker: str | None = None) -> dict:
        """The source asked about the entity (the ask kind's relations as the resolution hint), every
        fact through the gate, admitted ones into the store with provenance; the fetch and gate events."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import gate, _takes_item, _takes_relations
        from cubbyllm.reasoning.planner import Triple, normalize
        prov = getattr(self.world, "provenance", None)
        rec["entities"].append(entity)
        kw = {}
        if _takes_relations(self.source):
            kw["relations"] = hint
        if _takes_item(self.source):
            kw.update(qid=item, asker=asker)
        if "classes" in _inspect.signature(self.source.facts).parameters:
            kw["classes"] = True                         # a profile ask wants the KIND: see WikidataSource.facts
        items = list(self.source.facts(entity, **kw))
        last = getattr(self.source, "last", None) if isinstance(getattr(self.source, "last", None), dict) else {}
        fid = ev.emit("fetch", qid, source=getattr(self.source, "name", None), entity=entity, n=len(items), latent=False,
                      needs=hint, how=last.get("how"), item=last.get("qid"),
                      ambiguous=[list(a) for a in last.get("ambiguous", [])] or None)
        for item in items:
            fact = f"{item.obj} is the {item.rel} of {item.subj}" if isinstance(item, Triple) else item
            if isinstance(item, Triple):
                if hasattr(self.world, "index") and hasattr(self.world.index, "declare_relation"):
                    self.world.index.declare_relation(item.rel)
                if hasattr(self.known, "declare"):
                    self.known.declare(item.rel)
            p = gate(fact, self.world, self.source.name, entity)
            rec["learned"].append(_learned(p))
            if p.status == "accepted":
                self.world.add(fact)
                if hasattr(self.known, "add"):
                    self.known.add(fact)
                if prov is not None:
                    prov[" ".join(fact.split())] = self.source.name
            # the time is recorded for a DUPLICATE too. The preloaded world carries no qualifiers,
            # so every fact it already holds is undated -- and when the source restates one, the
            # gate calls it a duplicate and the old code dropped the date with it. That is why
            # "what was the population of Quebec City in 2011" stayed ambiguous through a refetch
            # that had the answer in hand (2026-09-14, live). Learning WHEN a fact was true adds
            # no claim: it is the same fact, said with the qualifier the source always had.
            if p.status in ("accepted", "duplicate"):
                times = getattr(self.world, "times", None)        # WHEN the source said the fact was true
                src_times = getattr(self.source, "times", None)
                if times is not None and src_times:
                    w = src_times.get(normalize(fact))
                    if w:
                        times[" ".join(fact.split())] = w
            ev.emit("gate", fid, fact=fact, status=p.status, clash=p.clash, lifted=False,
                    provenance=(prov or {}).get(" ".join(fact.split())))
        return last

    def introspect(self, question: str, kind: str, entity: str, t0: float, how: str) -> dict:
        """The loop asked about ITSELF. Same shape as a profile ask and a different store.

        The loop is the one subject it can witness directly -- it does not take anybody's word
        for its own history -- so these facts need no source and carry `self` as provenance.
        That privilege is fenced: a self fact never enters the world store, the answer says
        plainly that it is about the loop, and there is no relation here that carries a
        judgement. "refused 3 times" is a fact; "is unreliable" is an opinion, and this path
        cannot express one.

        The facts are DERIVED on every ask and never stored, so they cannot go stale and cannot
        be edited into something the loop did not do."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        qid = ev.emit("question", text=question, source="self")
        pid = ev.emit("plan", qid, seed=normalize(entity), relations=[], tail=entity, n_hop=0, ask=kind,
                      program=cot_self(normalize(entity), kind), how=how)
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": "self",
                     "plan": [], "seed": normalize(entity), "ask": kind, "world": "self",
                     "program": cot_self(normalize(entity), kind), "learned": [], "entities": [entity],
                     "aliased": [], "snapped": None, "trace": [], "profile": [], "prose": None,
                     "candidates": None, "clarify": None, "wall_s": 0.0}
        src = SelfSource(self)
        items = src.facts(entity)
        ev.emit("fetch", qid, source="self", entity=entity, n=len(items), latent=False,
                how=(src.last or {}).get("how"))
        if not items:
            rec["reason"] = "no_self_record"
            rec["needs"] = (src.last or {}).get("how") or "the loop has no record of that subject"
            ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"])
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            return rec
        lines = []
        for t in items:
            fact = f"{t.obj} is the {t.rel} of {t.subj}"
            lines.append({"relation": t.rel, "value": t.obj, "fact": fact, "provenance": "self",
                          "similarity": 1.0, "when": None})
        rec["profile"] = lines
        rec["trace"] = [l["fact"] for l in lines]
        by_rel: dict[str, list[str]] = {}
        for l in lines:
            by_rel.setdefault(say(l["relation"]), []).append(l["value"])
        rec["answer"] = entity + " \u2014 " + "; ".join(f"{r}: {', '.join(v)}" for r, v in by_rel.items())
        # NO PARAPHRASE ON THIS PATH, deliberately. 2026-09-14, live and wrong: given
        # `fact learned: 152; source: wikidata (152)` the talk adapter wrote "referenced in 152
        # sources" (it is 152 facts from ONE source), and given `last answer: 574482` it wrote
        # "the last answer was given 574482 units ago". Every name and number in both sentences
        # occurs in the facts, so the grounding guard passed them: the guard checks VOCABULARY,
        # not semantics, and it holds up on world facts only because the relations there are
        # ordinary English the adapter has seen a million times. "fact learned", "call",
        # "provenance" are not, and it confabulates what they mean.
        #
        # A loop reporting on itself has no business being fluent about it. The counts ARE the
        # answer, they are already readable, and prose can only add an interpretation of numbers
        # whose meaning the speaker does not know. Fluency here buys nothing and costs the kill
        # line.
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"],
                facts=rec["answer"], profile=rec["profile"], prose=rec["prose"], entities=[entity], fetched=0)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def _fetch_news(self, qid: int, entity: str, rec: dict) -> int:
        """A date's headlines into the store, through the SAME gate every other fact passes, each
        keeping its own publisher as provenance and its own day in `times`. Nothing downstream
        changes: `dated_facts` reads `times`, so the news arrives by the front door."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.learn import gate
        from cubbyllm.reasoning.planner import normalize
        if self.news is None:
            return 0
        try:
            items = list(self.news.facts(entity))
        except Exception as e:                               # noqa: BLE001 -- a feed that is down is not an answer
            ev.emit("fetch", qid, source="news-rss", entity=entity, n=0, latent=False,
                    how=f"news unavailable: {str(e)[:80]}")
            return 0
        prov = getattr(self.world, "provenance", None)
        times = getattr(self.world, "times", None)
        fid = ev.emit("fetch", qid, source="news-rss", entity=entity, n=len(items), latent=False,
                      how=(getattr(self.news, "last", None) or {}).get("how"),
                      feeds=(getattr(self.news, "last", None) or {}).get("feeds"))
        n = 0
        for t in items:
            fact = f"{t.obj} is the {t.rel} of {t.subj}"
            key = normalize(fact)
            publisher = self.news.provenance.get(key, "news-rss")
            if hasattr(self.world, "index") and hasattr(self.world.index, "declare_relation"):
                self.world.index.declare_relation(t.rel)
            if hasattr(self.known, "declare"):
                self.known.declare(t.rel)
            g = gate(fact, self.world, publisher, entity)
            rec["learned"].append(_learned(g))
            if g.status == "accepted":
                self.world.add(fact)
                if hasattr(self.known, "add"):
                    self.known.add(fact)
                if prov is not None:
                    prov[" ".join(fact.split())] = publisher       # the PUBLISHER, not "news-rss"
                w = self.news.times.get(key)
                if times is not None and w:
                    times[" ".join(fact.split())] = w
                n += 1
            ev.emit("gate", fid, fact=fact, status=g.status, clash=g.clash, lifted=False, provenance=publisher)
        return n

    def profile(self, question: str, kind: str, entity: str, t0: float, how: str, item: str | None = None, asker: str | None = None) -> dict:
        """'who / what / where is X' -- the retrieval program. The plan is SEED + ASK (what the emitter
        learns to write); the host fills the store from the source when it holds nothing about X; the
        facts under the ask kind's relations are bound into a program and the VM recovers each one --
        only what the VM returned at the 1-hop threshold is spoken. The reply is those facts; the talk
        adapter, when one is mounted, paraphrases them, and the paraphrase is kept only when every name
        and number in it is in the facts. `reason: profile` -- a retrieval, not a walked chain."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import normalize
        from cubbyllm.reasoning.programs import build_chain_program
        qid = ev.emit("question", text=question, source=getattr(self.source, "name", None))
        pid = ev.emit("plan", qid, seed=normalize(entity), relations=[], tail=entity, n_hop=0, ask=kind,
                      program=cot_profile(normalize(entity), kind), how=how)
        rec: dict = {"question": question, "answer": None, "verified": False, "reason": "profile", "plan": [], "seed": normalize(entity),
                     "ask": kind, "program": cot_profile(normalize(entity), kind), "learned": [], "entities": [], "aliased": [],
                     "snapped": None, "trace": [], "profile": [], "prose": None, "candidates": None, "clarify": None, "wall_s": 0.0}
        prov = getattr(self.world, "provenance", None)
        rels = PROFILE_KINDS[kind]
        if kind == "when":
            # the date index: the facts the store knows the DATE of. A `when` ask never FETCHES the date
            # itself -- a year's own item is trivia (2026-09-14, live: asking "what happened in 2026?"
            # gated `2026 is the point in time of 2026` into the store and then read it back as the
            # answer). Events come from facts about entities, dated by their source; nothing here
            # invents them, and an empty index is an honest refusal.
            dated = self.dated_facts(entity, limit=None)         # the TRUE count: see too_broad
            if not dated and self.news is not None:
                # ... and when the store knows nothing about that date, ask something that might.
                # This is the fix for the refusal Nick hit head-on: "if I ask what happened in
                # 2026? it refuses it" -- correctly, because the store held nothing dated. It
                # holds something now, and it came through the gate like everything else.
                self._fetch_news(qid, entity, rec)
                dated = self.dated_facts(entity, limit=None)
            if len(dated) > NEWS_CLARIFY_MIN:
                return self.too_broad(rec, qid, entity, dated, t0)
            if dated:
                return self.dated(rec, qid, pid, entity, dated[:PROFILE_MAX_LINES], t0)
            have = [t for t in self.facts_about(entity) if normalize(t.rel) in PROFILE_KINDS["when"]]
            if not have:
                return self.no_events(rec, qid, t0)
        have = self.facts_about(entity)
        last: dict = {}
        known_rels = set(rels) | (set(PROFILE) if kind in FALLBACK_KINDS else set())
        if not any(normalize(t.rel) in known_rels for t in have):
            last = self._fetch(qid, entity, list(RESOLVE_HINT[kind]), rec, item, asker)
            have = self.facts_about(entity)
        # the lines: the ask kind's relations first, then any other profile relation the store holds
        chosen: list[tuple[str, str, str]] = []                       # (rel, obj, the stored fact text)
        by_key = {(normalize(t.rel), t.obj): " ".join(f.split()) for f, t in self.world.index._by_subj.get(normalize(entity), []) if t is not None}
        order = tuple(rels) + (tuple(r for r in PROFILE if r not in rels) if kind in FALLBACK_KINDS else ())
        times = getattr(self.world, "times", None) or {}
        for rel in order:
            objs = list(dict.fromkeys(t.obj for t in have if normalize(t.rel) == rel))
            # a value whose truth depends on WHEN: Quebec City's population is 547 and 516,622 and
            # 546,958, each true at its own census (2026-09-14, Nick: `547 is the population of Quebec
            # City` gated in, clashing with 516622 -- the source dated every one of them and the store
            # dropped the date). A present-tense ask wants the LATEST, said with its date; the older
            # values stay stored, and a question that binds a year reads them through the date index.
            dated = [(o, (times.get(by_key.get((rel, o), "")) or {})) for o in objs]
            n_dated = sum(1 for _o, w in dated if w)
            if n_dated > 1:
                dated.sort(key=lambda q: (q[1].get("point") or q[1].get("start") or ""), reverse=True)
                objs = [o for o, _w in dated]
            for obj in objs[:1 if n_dated > 1 else PROFILE_MAX_PER_RELATION]:
                chosen.append((rel, obj, by_key.get((rel, obj), f"{obj} is the {rel} of {entity}")))
            if len(chosen) >= PROFILE_MAX_LINES:
                break
        chosen = chosen[:PROFILE_MAX_LINES]
        if not chosen:
            amb = last.get("ambiguous") or []
            if amb:
                # the asker decides, the host never picks (Nick, 2026-09-14: "when ambiguous it should
                # either select A based on context or B ask the user: do you want to know more about:
                # choices"). A is the resolution tiers above -- the hop the question needs, label over
                # alias, the meta-pages dropped; when they leave more than one, this is B: the choices go
                # back with the item ids, and the answer re-asks with the one the asker names.
                rec["reason"] = "ambiguous_entity"; rec["candidates"] = [[a[1], a[2], a[0]] for a in amb]
                rec["clarify"] = {"question": "Which one do you mean?",
                                  "choices": [{"label": a[1], "detail": a[2], "item": a[0]} for a in amb]}
            else:
                rec["reason"] = ("no_events_for_date" if kind == "when" else
                                 ("entity_unresolved" if rec["entities"] and not rec["learned"] else "profile_empty"))
                if kind == "when":
                    # the honest diagnosis (2026-09-14): nothing here indexes events BY DATE. The wiki world
                    # holds 76,897 `timeline event` facts, every one keyed by its ENTITY ('Girard Desargues'
                    # -> 'Girard Desargues Birth'), and 0 facts whose subject is a year; a year's item carries
                    # the year's own trivia, not its events. `unknown_relation` blamed the word 'happened'
                    # for a missing index. A dated-news world is what would answer this shape.
                    rec["needs"] = ("no source indexes events by date: the event facts are keyed by entity, not by "
                                    "year, and a year's own item carries only its trivia")
            ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], candidates=rec["candidates"], clarify=rec.get("clarify"), needs=rec.get("needs"),
                    entities=list(rec["entities"]), fetched=len(rec["learned"]))
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            return rec
        # the retrieval program: every chosen fact bound; the VM recovers each role -- what it returns is spoken
        from cubbyllm.reasoning.planner import Triple
        wid = ev.emit("walk", pid, verified=None, reason="profile", answer=None, refused=None)
        kept: list[dict] = []; n_calls = 0
        for rel, obj, fact in chosen:
            triple = Triple(obj=obj, rel=rel, subj=entity)
            source, fns = build_chain_program([triple], [rel])
            out = self._run_fn(source, fns[0]); n_calls += 1
            sim = out.get("similarity"); got = out.get("result")
            ok = sim is not None and sim >= self.tau_profile and normalize(got or "") == normalize(obj)
            src_name = (prov or {}).get(fact) or ("store" if fact in self.world else None)
            hid = ev.emit("hop", wid, hop=0, query=f"what is the {rel} of {entity}", fact=fact, how="profile · VM recover",
                          similarity=sim, provenance=src_name, verified=ok)
            ev.emit("fact", hid, text=fact, subj=entity, rel=rel, obj=obj, provenance=src_name)
            if ok:
                w = (getattr(self.world, "times", None) or {}).get(fact) or {}
                kept.append({"relation": rel, "value": obj, "fact": fact, "provenance": src_name, "similarity": sim,
                             "when": (w.get("point") or "–".join(x for x in (w.get("start"), w.get("end")) if x)) or None})
        ev.emit("vm", wid, verified=bool(kept) and len(kept) == len(chosen), program_lines=len(chosen) + 2,
                recovered=len(kept), bound=len(chosen))
        rec["profile"] = kept; rec["trace"] = [k["fact"] for k in kept]
        if kept:
            by_rel: dict[str, list[str]] = {}
            for l in kept:
                by_rel.setdefault(say(l["relation"]), []).append(l["value"] + (f" ({when_text(l['when'])})" if l.get("when") else ""))
            rec["answer"] = entity + " — " + "; ".join(f"{r}: {', '.join(v)}" for r, v in by_rel.items())
            rec["prose"] = self.paraphrase(pid, entity, kind, kept)
        else:
            rec["reason"] = "profile_unrecovered"
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"], facts=rec["answer"],
                profile=rec["profile"] or None, prose=rec["prose"], entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def paraphrase(self, pid: int, entity: str, kind: str, lines: list[dict]) -> str | None:
        """The talk adapter says the facts in a sentence or two; kept only when grounded -- every name and
        number in it occurs in the facts. No talk adapter mounted (a programs-only emitter): None, and
        the facts are the reply."""
        from cubbyllm.reasoning import events as ev
        adapters = getattr(self.emitter, "adapters", None)
        if not isinstance(adapters, dict) or "talk" not in adapters or not getattr(self.emitter, "is_split", False):
            return None
        facts = [l["fact"] for l in lines]
        # the adapter reads the SAID form (say() has already dropped the schema words) -- asking a 4B model
        # to translate "the administrative territorial entity" itself only spends its attention on the
        # database's vocabulary instead of on the sentence. 2026-09-14, Nick, live.
        said = [f"{say(l['relation'])}: {l['value']}" + (f" ({when_text(l['when'])})" if l.get("when") else "")
                for l in lines]
        ask = {"who": "who", "what": "what", "where": "where", "when": "what happened in"}.get(kind, "what")
        prompt = (f"Facts about {entity}:\n" + "\n".join(f"- {s}" for s in said) +
                  f"\n\nIn one or two plain sentences, say {ask} {entity} is, using ONLY these facts. "
                  "Every NAME, DATE and NUMBER must appear exactly as given, and you may add none that is "
                  "not in the facts.")
        from emitter import clean_reply
        try:
            raw = self.emitter.emit(prompt, max_new_tokens=160, context={"role": "talk"}, temperature=0.0,
                                    system=PARAPHRASE_SYSTEM)     # not the identity persona: under it the talk adapter says nothing here
            text = " ".join(clean_reply(raw).split())
        except Exception as e:                                   # noqa: BLE001 -- no paraphrase, the facts stand
            ev.emit("paraphrase", pid, ok=False, text=None, rejected=[f"talk adapter failed: {str(e)[:80]}"])
            return None
        ok, bad = grounded_prose(text, facts + said, entity)
        ev.emit("paraphrase", pid, ok=ok, text=text if ok else None, rejected=bad or None, draft=text, attempt=1)
        if ok:
            return text
        # A refusal is a result, but it is a poor one when the facts are right there and only one word
        # was invented (2026-09-14: the Bank of England lost its whole paraphrase over "institution").
        # The adapter is told which words it made up and gets ONE more turn. The guard is unchanged --
        # the second draft passes the same check or is refused the same way.
        try:
            retry = (prompt + "\n\nYour last attempt used " + ", ".join(f"'{b}'" for b in bad[:6]) +
                     " -- no fact above says that. Write it again without those words, and do not say "
                     "what KIND of thing it is unless a fact names the kind.")
            raw = self.emitter.emit(retry, max_new_tokens=160, context={"role": "talk"}, temperature=0.0,
                                    system=PARAPHRASE_SYSTEM)
            text2 = " ".join(clean_reply(raw).split())
        except Exception as e:                                   # noqa: BLE001
            ev.emit("paraphrase", pid, ok=False, text=None, rejected=[f"retry failed: {str(e)[:60]}"], attempt=2)
            return None
        ok2, bad2 = grounded_prose(text2, facts + said, entity)
        ev.emit("paraphrase", pid, ok=ok2, text=text2 if ok2 else None, rejected=bad2 or None, draft=text2, attempt=2)
        return text2 if ok2 else None

    # -- the date index: "what happened in <year>" over what the store knows the date of -------
    def dated_facts(self, when: str, limit: int | None = PROFILE_MAX_LINES) -> list[tuple[str, dict]]:
        """The stored facts whose recorded time falls in the asked year or date -- the `times` map the
        source fills (`start`/`end`/`point`), never the fact text. A span counts for every year it
        covers ('position held' 2011-2019 answers 'what happened in 2015'), a point for its own.

        `limit=None` returns them ALL. The caller that has to decide whether a date is too broad
        to answer needs the true count, and a truncated list always looked answerable -- the
        clarify threshold could never fire behind a cap of nine."""
        m = _re.search(r"\b(1[0-9]{3}|2[0-9]{3})\b", when)
        if not m:
            return []
        year = int(m.group(1))
        day = None
        d = _re.search(r"\b(1[0-9]{3}|2[0-9]{3})-(\d{2})-(\d{2})\b", when)
        if d:
            day = d.group(0)
        out = []
        for fact, t in (getattr(self.world, "times", None) or {}).items():
            pt, st, en = t.get("point"), t.get("start"), t.get("end")
            hit = None
            if pt and (pt == day or (not day and pt[:4] == str(year))):
                hit = "on"
            elif st or en:
                # a CLOSED span is known to have covered its years, so it counts for each of them; an OPEN
                # one (a start with no end) is a state whose end the source never stated, and counting it
                # forward claims something nobody said. 2026-09-14, live and WRONG: "what happened in 2026?"
                # answered "Quebec City was the capital of Quebec" -- capital SINCE 1867, open -- and listed
                # seven twin cities as 2026 events. An open span counts for its start year only: that is the
                # part that happened.
                y0 = int(st[:4]) if st else None
                y1 = int(en[:4]) if en else None
                if y0 is not None and y0 == year:
                    hit = "began"
                elif y1 is not None and y1 == year:
                    hit = "ended"
                elif y0 is not None and y1 is not None and y0 < year < y1:
                    hit = "ongoing"
            if hit:
                out.append((fact, dict(t, hit=hit)))
        out.sort(key=lambda p: (p[1].get("point") or p[1].get("start") or "", p[0]))
        return out if limit is None else out[:limit]

    def dated(self, rec: dict, qid: int, pid: int, entity: str, dated: list, t0: float) -> dict:
        """The dated facts, each recovered by the VM like any profile line, as the answer."""
        from cubbyllm.reasoning import events as ev
        from cubbyllm.reasoning.planner import Triple, normalize, parse_fact
        from cubbyllm.reasoning.programs import build_chain_program
        prov = getattr(self.world, "provenance", None)
        wid = ev.emit("walk", pid, verified=None, reason="profile", answer=None, refused=None)
        kept = []
        for fact, t in dated:
            tr = parse_fact(fact)
            if tr is None:
                continue
            source, fns = build_chain_program([Triple(obj=tr.obj, rel=tr.rel, subj=tr.subj)], [tr.rel])
            out = self._run_fn(source, fns[0])
            sim = out.get("similarity")
            ok = sim is not None and sim >= self.tau_profile and normalize(out.get("result") or "") == normalize(tr.obj)
            span = t.get("point") or "–".join(x for x in (t.get("start"), t.get("end")) if x)
            src_name = (prov or {}).get(fact) or ("store" if fact in self.world else None)
            hid = ev.emit("hop", wid, hop=0, query=f"what happened in {entity}", fact=fact, how="dated · VM recover",
                          similarity=sim, provenance=src_name, verified=ok, when=span)
            ev.emit("fact", hid, text=fact, subj=tr.subj, rel=tr.rel, obj=tr.obj, provenance=src_name, when=span)
            if ok:
                kept.append({"relation": tr.rel, "value": tr.obj, "subject": tr.subj, "fact": fact,
                             "provenance": src_name, "similarity": sim, "when": span, "hit": t.get("hit")})
        ev.emit("vm", wid, verified=bool(kept) and len(kept) == len(dated), program_lines=len(dated) + 2,
                recovered=len(kept), bound=len(dated))
        rec["profile"] = kept; rec["trace"] = [k["fact"] for k in kept]
        if kept:
            rec["answer"] = f"{entity} — " + "; ".join(f"{k['when']}: {k['value']} is the {say(k['relation'])} of {k['subject']}" for k in kept)
            rec["prose"] = self.paraphrase(pid, entity, "when", kept)
            rec["scope"] = "the facts the store knows a date for -- not a record of the year"
        else:
            rec["reason"] = "profile_unrecovered"
        ev.emit("answer", qid, verified=False, answer=rec["prose"] or rec["answer"], reason=rec["reason"], facts=rec["answer"],
                profile=rec["profile"] or None, prose=rec["prose"], scope=rec.get("scope"),
                entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def too_broad(self, rec: dict, qid: int, entity: str, dated: list, t0: float) -> dict:
        """Too much is dated in that range to answer in one breath, so the loop asks instead of
        picking. Nick, 2026-09-14: "when ambiguous it should either select A based on context or
        B ask the user: do you want to know more about: choices."

        The choices are a COUNT over the facts actually fetched -- the recurring names in them,
        commonest first. That is deliberately not a ranking of importance: this loop has no way
        to know what mattered on a day and no business pretending it does. It offers what was
        written about, says how much there is, and lets the asker narrow it."""
        from cubbyllm.reasoning import events as ev
        choices: list[str] = []
        if self.news is not None:
            try:
                choices = [w for w, _n in self.news.topics(entity)[:8]]
                if not choices:
                    # One day's headlines rarely repeat a name, so the recurrence bar that makes
                    # a YEAR's menu meaningful empties a DAY's (2026-09-14, live: 51 facts and
                    # nothing to offer). "51 things happened and I have no suggestions" is a
                    # worse answer than a list of single mentions, and both are honest.
                    choices = [w for w, _n in self.news.topics(entity, min_n=1)[:8]]
            except Exception:                                # noqa: BLE001
                choices = []
        if not choices:                                      # no news source: the relations themselves
            choices = list(dict.fromkeys(t.get("relation") or "" for _f, t in dated if isinstance(t, dict)))[:8]
        rec["reason"] = "date_too_broad"
        rec["clarify"] = {"entity": entity, "n": len(dated), "choices": [c for c in choices if c],
                          "how": "a count over the headlines fetched -- what was written about, not what mattered"}
        rec["needs"] = (f"{len(dated)} facts are dated in {entity}; ask about one of them, "
                        f"or a narrower date")
        ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"],
                clarify=rec["clarify"], entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def no_events(self, rec: dict, qid: int, t0: float) -> dict:
        """Nothing in the store is dated in that range, and a `when` ask does not fetch a date's own item."""
        from cubbyllm.reasoning import events as ev
        rec["reason"] = "no_events_for_date"
        rec["needs"] = ("no source indexes events by date: the event facts are keyed by entity, not by "
                        "year, and a date's own item carries only its trivia")
        ev.emit("answer", qid, verified=False, answer=None, reason=rec["reason"], needs=rec["needs"],
                entities=list(rec["entities"]), fetched=len(rec["learned"]))
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec
