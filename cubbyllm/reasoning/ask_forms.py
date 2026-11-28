"""cubbyllm/reasoning/ask_forms.py -- the question shapes the ask loop reads and the programs it writes for them: the
profile ask (who / what / where is X?), the temporal ask, the self ask, the relation ask; the VM thresholds by hop count
and the profile's relations by kind. Split out of ask.py (order 0031), verbatim; ask.py re-exports every name.

Wired: WIRED (the ask loop).
"""
from __future__ import annotations

import re as _re
from ..worlds.self_source import LOOP, SelfSource   # noqa: E402

__wiring__ = "WIRED"


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
_PROFILE_ASK = _re.compile(
    r"^\s*(?P<kind>who|what|where)(?:['’]s|\s+(?:is|was|are|were))\s+(?:the\s+)?(?P<ent>[^?]*?)\s*\??\s*$", _re.I)


# "what was X known/famous/notable for?" / "why is X famous?" (and French): a focused profile over
# notable work, claim to fame, field of work, awards and occupation -- never a bare entity named "X known for"
_KNOWN_FOR_EN = _re.compile(
    r"^\s*what(?:['’]s|\s+(?:is|was|are|were))\s+(?:the\s+)?(?P<ent>[^?]+?)\s+"
    r"(?:(?:best|most|mainly|chiefly|widely|principally|primarily)\s+)?"
    r"(?:known|famous|notable|noted|remembered|celebrated)\s+for\s*\??\s*$", _re.I)
_WHY_FAMOUS_EN = _re.compile(
    r"^\s*why(?:['’]s|\s+(?:is|was|are|were))\s+(?:the\s+)?(?P<ent>[^?]+?)\s+"
    r"(?:(?:so|well|best|most|widely)\s+)?(?:famous|known|notable)\s*\??\s*$", _re.I)
_KNOWN_FOR_FR = _re.compile(
    r"^\s*(?:(?:pour\s*quoi|pourquoi)\s+(?P<ent1>[^?]+?)\s+"
    r"(?:est|était|sont|étaient)(?:-(?:il|elle|ils|elles))?\s+"
    r"(?:(?:si|très|bien|le\s+plus|la\s+plus)\s+)?(?:connue?s?|célèbres?)"
    r"|qu['’]est-ce\s+qui\s+(?:a\s+rendu|rend)\s+(?P<ent2>[^?]+?)\s+(?:célèbre|connue?))\s*\??\s*$", _re.I)
_FR_PROFILE_ASK = _re.compile(
    r"^\s*(?:(?P<who>qui)\s+(?:est|était|sont|étaient)\s+|"
    r"(?P<what>qu['’]est-ce\s+qu['’]\s*|qu['’]est-ce\s+que\s+|c['’]est\s+quoi\s+)|"
    r"(?P<where>où)\s+(?:est|était|sont|étaient|se\s+trouve(?:nt)?|se\s+situe(?:nt)?)\s+)(?P<ent>[^?]*?)\s*\??\s*$", _re.I)


# the temporal ask: the entity is a DATE and the relation is 'what happened' -- the same retrieval
# program with ASK 'when' (2026-09-14, Nick: "if I ask what happened in 2026? it refuses it"; it was
# refused as `unknown_relation`, blaming the word 'happened', where the truth is that no source the
# host has indexes events BY DATE -- the wiki world holds 0 facts whose subject is a year)
_WHEN_ASK = _re.compile(r"^\s*(?:what|which)\s+(?:\w+\s+)?(?:happened|occurred|occured|took\s+place|went\s+on)"
                        r"(?:\s+(?:in|on|during|around))?\s+(?:the\s+)?(?P<ent>[^?]*?)\s*\??\s*$", _re.I)


# a trailing past participle or dangling preposition is the QUESTION's predicate, not the end of a name
# (2026-10-16: "Where was Marie Curie born?"; 2026-11-27: "What was marie curie known for?" read as a
# profile of 'marie curie known for', refused entity_unresolved). A name's own capitalised word
# (Manchester United) is left alone.
_PARTICIPLES = {"born", "built", "made", "held", "written", "known", "sold", "shot", "led", "won", "begun",
                "found", "spoken", "grown", "drawn", "sung", "shown", "taught", "fought", "taken", "given",
                "chosen", "broken", "driven", "frozen", "thrown", "stolen", "hidden", "worn", "caught",
                "bought", "brought", "heard", "kept", "spent", "sent", "done", "seen"}
_TRAIL_PREPS = {"for", "by", "to", "in", "on", "at", "from", "with", "about", "after", "into", "onto",
                "through", "under", "over", "against", "between", "as"}
_FR_PREDICATE = {"né", "née", "nés", "nées", "mort", "morte", "morts", "mortes", "situé", "située", "situés",
                 "situées", "construit", "construite", "fondé", "fondée", "écrit", "écrite", "créé", "créée",
                 "connu", "connue", "fait", "faite", "enterré", "enterrée", "découvert", "découverte"}
_FR_CLAUSE = {"a", "ont", "est", "sont", "était", "étaient", "fait", "pour", "par", "dans", "sur", "avec"}
_PRONOUN_ENTS = {"it", "this", "that", "these", "those", "he", "she", "they", "you", "i", "we", "me", "him",
                 "her", "them", "cela", "ça", "c'est", "c’est", "il", "elle", "ils", "elles", "tu", "vous", "je", "nous"}


_ASK_ROLE = _re.compile(r'bind\s+frame\s*,\s*ASK\s*,\s*"(?P<kind>who|what|where|when|known_for)"\s*;', _re.I)


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
    "known_for": ("notable work", "claim to fame", "field of work", "award received", "occupation", "position held",
                  "has use", "use", "significant event", "genre", "instance of"),
}


PROFILE = tuple(dict.fromkeys(r for k in ("who", "what", "where") for r in PROFILE_KINDS[k]))


FALLBACK_KINDS = ("who", "what", "where", "known_for")   # kinds that may also show any other profile relation


# the resolution hint: the kind's relations, specific first, with the classifiers dropped -- `instance of`
# and `part of` are carried by almost everything and decide nothing
RESOLVE_HINT = {k: tuple(r for r in v if r not in ("instance of", "subclass of", "part of", "sex or gender"))[:8] for k, v in PROFILE_KINDS.items()}


PROFILE_MAX_PER_RELATION = 3


PROFILE_MAX_LINES = 9


# Past this many dated facts, "what happened in 2026?" is not a question with an answer, it is a
# question with a library. Nick, 2026-09-14: "when ambiguous it should either select A based on
# context or B ask the user: do you want to know more about: choices."
NEWS_CLARIFY_MIN = 12


PARAPHRASE_SYSTEM = ("You are a careful writer. You restate given facts in plain sentences and never add anything. "
                     "The facts are written in a database's own vocabulary; say them in ordinary English.")


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


def _verb_tail(ent: str) -> bool:
    """Does the entity end in a past participle or dangling preposition (not a capitalised word of a name)?"""
    words = ent.split()
    if len(words) < 2 or words[-1][:1].isupper():
        return False
    tail = words[-1].lower().strip(".,")
    return tail in _PARTICIPLES or tail in _TRAIL_PREPS or (len(tail) >= 5 and tail.endswith("ed"))


def _of_relation(ent: str, is_relation) -> bool:
    if " of " not in f" {ent} ":
        return False
    heads = [ent[:m.start()].strip() for m in _re.finditer(" of ", ent)]
    heads = [h[4:].strip() if h.lower().startswith("the ") else h for h in heads]
    return is_relation is None or not heads or not heads[0] or any(is_relation(h) for h in heads if h)


def _strip_article(ent: str, fr: bool = False) -> str:
    if fr:
        for art in ("le ", "la ", "les ", "un ", "une ", "des "):
            if ent.startswith(art):
                return ent[len(art):].strip()
        if ent.startswith(("l'", "l’")):
            return ent[2:].strip()
        return ent
    return ent.split(" ", 1)[1].strip() if ent.startswith(("a ", "an ")) else ent


def profile_ask(question: str, is_relation=None) -> tuple[str, str] | None:
    """(kind, entity) of a 'who / what / where is X', 'what is X known for', or 'what happened in <date>'
    question (EN/FR), or None when the question asks a relation plan for the emitter."""
    m = _WHEN_ASK.match(question)
    if m:
        ent = m.group("ent").strip().strip(",")
        return ("when", ent) if ent and len(ent.split()) <= 5 else None
    for rx, fr in ((_KNOWN_FOR_EN, False), (_WHY_FAMOUS_EN, False), (_KNOWN_FOR_FR, True)):
        m = rx.match(question)
        if m:
            raw = (m.group("ent1") or m.group("ent2")) if fr else m.group("ent")
            ent = _strip_article(raw.strip().strip(".,"), fr=fr)
            if (not ent or "'s " in ent or ent.endswith("'s") or len(ent.split()) > 6
                    or ent.lower() in _PRONOUN_ENTS or _of_relation(ent, is_relation)):
                return None
            return ("known_for", ent)
    m = _PROFILE_ASK.match(question)
    if m:
        ent = _strip_article(m.group("ent").strip())
        if not ent or "'s " in ent or ent.endswith("'s") or len(ent.split()) > 6:
            return None
        if _verb_tail(ent) or _of_relation(ent, is_relation):
            return None
        return m.group("kind").lower(), ent
    m = _FR_PROFILE_ASK.match(question)
    if m:
        ent = _strip_article(m.group("ent").strip(), fr=True)
        low = [w.lower().strip(".,") for w in ent.split()]
        if (not ent or len(low) > 6 or ent.lower() in _PRONOUN_ENTS or low[0] in _FR_PREDICATE
                or low[-1] in _FR_PREDICATE or any(w in _FR_CLAUSE for w in low)
                or any(p in f" {ent.lower()} " for p in (" de ", " du ", " des "))
                or _re.search(r"\bd['’]", ent.lower())):
            return None
        kind = "who" if m.group("who") else "where" if m.group("where") else "what"
        return kind, ent
    return None


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
