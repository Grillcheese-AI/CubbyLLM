"""saying — scoring a candidate UTTERANCE into the five terms `choosing` bends.

Wired: WIRED (`CubbyChat.candidates`). Knows what a sentence is; knows nothing
about a maze, an encyclopedia or a ghost — the same rung of the SDK as
`choosing` is one below it.

WHY THIS EXISTS AS ITS OWN MODULE. `choosing` is deliberately ignorant: it
takes five numbers and the body's knobs and returns a ranking. Something has
to turn a REPLY into those five numbers, and that job is neither the filter's
(it would stop being world-agnostic) nor one world's (every world with a mouth
needs it). So it sits between them, and any agent that speaks can use it.

THE TERM THAT MATTERS IS `risk`, and it is the reason to do this at all. The
kill line for this project is zero wrong answers spoken. Today that is a guard
cascade in `CubbyChat.candidates`: a reply either passes every host check and
is spoken, or fails one and is dropped. A cascade has no notion of HOW FAR a
claim reaches past what can be backed, so it cannot be more careful when the
body has reason to be careful — it is the same cascade in a quiet room and in
a fire.

`risk` here is claim surface: the specifics a reply commits to that the host
cannot check — figures, years, quantities, named things. Hedging lowers it,
because a hedged claim commits to less. Anything the caller CAN back (pass it
in `grounded`) does not count against the reply, which is the hook the
verified path in `ask.py` plugs into.

That number then meets `caution` in `choosing.gains()`, and the result is the
behaviour the cascade could not express: an alarmed Cubby needs a better
reply before he will speak at all, and falls back to the sanctioned line when
he does not have one. Same words available, same guards, different threshold —
set by the body, per turn, with no retraining and no second model.

TWO CONSTANTS ARE POLICY, NOT MEASUREMENT, and they are named rather than
buried so they can be argued with:

  SAFE_VALUE   what the don't-know line is worth. It is a real answer to
               "I have nothing verified to say", so it is not zero; it says
               nothing the asker wanted, so it is not one.
  SAFE_NOVELTY zero, always. The fallback is by definition the line he has
               said before. This is what lets low `creativity` — a
               frightened body — actively PREFER it, via the signed novelty
               term, instead of merely stopping to prefer anything else.

Every number that comes out of this is [stand-in].
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import choosing  # noqa: E402

__wiring__ = "WIRED"

SAFE_VALUE = 0.55        # the don't-know line answers the wrong question, honestly
SAFE_NOVELTY = 0.0       # it is the familiar line, by construction
ANSWER_VALUE = 1.0       # a reply that survived the host guards

COST_CAP = 60.0          # words at which a reply costs everything it can
RISK_CAP = 10.0          # unbacked specifics at which risk saturates
HEDGE_RELIEF = 0.45      # how much of the claim surface a hedge takes back

# Claim surface: what a sentence commits to that a host cannot check in free
# chat. Deliberately shallow and deliberately listed — a shallow measure whose
# failure modes are visible beats a deep one whose failures are not.
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", re.M)
_UNIT = re.compile(r"\b\d+\s*(?:%|km|kg|m|cm|mm|s|ms|h|hrs?|years?|ans|°[CF])\b", re.I)

_HEDGE = re.compile(
    r"\b(i think|i believe|maybe|perhaps|probably|might|may be|not sure|i'm not certain|"
    r"je pense|je crois|peut-être|sans doute|il se peut|pas s[ûu]r|je ne suis pas certain)\b", re.I)

_SECOND_PERSON = re.compile(
    r"\b(you|your|yours|you're|tu|toi|ton|ta|tes|vous|votre|vos|t'as|t'es)\b", re.I)

_WORD = re.compile(r"[\w']+", re.UNICODE)

# Function words carry no content, so they must not make two different replies
# look similar to the novelty term.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "that", "this", "these", "those", "there", "here", "so", "as", "not", "no", "yes",
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais", "si", "dans", "sur",
    "est", "sont", "suis", "es", "être", "avoir", "ai", "as", "a", "ont", "je", "tu", "il",
    "elle", "nous", "vous", "ils", "elles", "ce", "cette", "ces", "que", "qui", "pas", "ne",
}


def _content(text: str) -> set[str]:
    return {w for w in (m.group(0).lower() for m in _WORD.finditer(text or "")) if w not in _STOP}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def specifics(text: str) -> set[str]:
    """Every specific `claim_surface` would count, lowercased.

    THE UNIT OF GROUNDING, and it is deliberately the same function on both
    sides. `claim_surface(reply, grounded=specifics(source))` then reads as
    exactly what it means: the specifics the reply commits to that the source
    does not contain. Handing `grounded` a bag of every word in the source
    instead — which is what the first coach wiring did — makes risk zero by
    construction and quietly retires the term.

    ONE COMMITMENT COUNTS ONCE. The three patterns overlap on purpose
    ("12000 years" is a number AND a quantity), so they are merged by span,
    longest first. Counting it twice was the first version, and it made an
    ordinary sentence with one date and three names read as maximally
    reckless.
    """
    t = text or ""
    if not t.strip():
        return set()
    spans: list[tuple[int, int, str]] = []
    for rx in (_UNIT, _NUMBER, _PROPER):
        for m in rx.finditer(t):
            a, b = m.span()
            if any(a < eb and ea < b for ea, eb, _ in spans):
                continue
            spans.append((a, b, m.group(0)))
    return {s[2].lower() for s in spans}


def claim_surface(text: str, grounded: set[str] | None = None) -> float:
    """0..1 — how far the sentence commits past what a host can check.

    Counts the specifics a reader would take as facts: figures, quantities,
    years, named things. A specific the caller says it can back does not
    count; that is the whole interface to a grounded path. A hedge takes back
    part of the surface rather than all of it, because "I think Rome fell in
    1650" is still a claim, only a softer one.
    """
    t = text or ""
    if not t.strip():
        return 0.0
    backed = {g.lower() for g in (grounded or ())}
    mine = specifics(t)
    unbacked = [s for s in mine
                if s not in backed
                and not all(w in backed for w in _WORD.findall(s))]
    surface = _clip(len(unbacked) / RISK_CAP)
    if _HEDGE.search(t):
        surface *= (1.0 - HEDGE_RELIEF)
    return surface


def novelty_against(text: str, history) -> float:
    """1 − the closest overlap with anything already said.

    NOTHING SAID YET MEANS NOVELTY IS ZERO, not one. For speech this term is
    a repetition signal, and with an empty history there is nothing to be
    unlike — the quantity is not maximal, it is undefined. `choosing` states
    the convention for exactly this case: a world that cannot score a
    dimension leaves it at 0, and the filter then has nothing to bend there.
    Returning 1.0 instead let a frightened body's negative novelty gain punish
    the first reply of every conversation for being new, which is not
    neophobia, it is an artefact of an empty list.
    """
    words = _content(text)
    if not words or not history:
        return 0.0
    best = 0.0
    for prior in history or ():
        p = _content(prior)
        if not p:
            continue
        inter = len(words & p)
        if inter:
            best = max(best, inter / len(words | p))
    return _clip(1.0 - best)


def social_reach(text: str) -> float:
    """How much the utterance is ADDRESSED to someone rather than narrated.
    Second person, and a question, which hands the turn back."""
    t = text or ""
    if not t.strip():
        return 0.0
    reach = _clip(len(_SECOND_PERSON.findall(t)) / 3.0)
    if "?" in t:
        reach = _clip(reach + 0.25)
    return reach


def effort(text: str) -> float:
    """What saying it takes out of you, as length. Crude on purpose: the point
    is that a depleted body gets terser without anyone writing a rule that
    says so."""
    return _clip(len(_WORD.findall(text or "")) / COST_CAP)


def score(text: str, *, safe: bool = False, history=(), grounded: set[str] | None = None,
          key: str | None = None) -> choosing.Option:
    """One candidate reply as a `choosing.Option`.

    `safe=True` marks the sanctioned fallback (the don't-know line): its value
    is policy, its risk is zero because it is verbatim from the identity
    facts, and its novelty is zero because it is the line he always has.
    """
    if safe:
        return choosing.Option(
            key=key if key is not None else text,
            value=SAFE_VALUE, risk=0.0, novelty=SAFE_NOVELTY,
            cost=effort(text), social=social_reach(text),
            meta={"safe": True})
    risk = claim_surface(text, grounded)
    return choosing.Option(
        key=key if key is not None else text,
        value=ANSWER_VALUE, risk=risk, novelty=novelty_against(text, history),
        cost=effort(text), social=social_reach(text),
        meta={"safe": False})


def rank(candidates, mod: dict, *, safe: str | None = None, history=(),
         grounded: set[str] | None = None):
    """-> (ordered_texts, detail). The whole thing, for a caller holding a list
    of survivors and a body.

    EVERY CANDIDATE COMES BACK, re-ordered. The caller speaks the first one,
    but the rest stay in the list because downstream the VM will only accept a
    selection that was offered, and dropping the sanctioned fallback here to
    save a line would quietly remove the thing the whole guard rests on.

    `choosing.weigh` rather than `choosing.choose`, for two stated reasons:

      stickiness is not passed. It is right for a body moving through a world —
      a settled agent keeps walking the way it was walking — and wrong for a
      mouth, where it would mean a calm Cubby repeats himself. Persistence of
      ACTION is a trait; persistence of PHRASE is a defect.

      `breadth` is not applied HERE. With no stickiness and no sampling, the
      best of a narrowed field is the best of the whole field, so narrowing at
      this point would change nothing while looking like it did. Urgency
      narrows speech one step earlier instead — at how many candidates get
      generated at all (`CubbyChat.candidates`), which is a real effect and
      also the cheaper one.
    """
    texts = list(candidates)
    if safe is not None and safe not in texts:
        texts.append(safe)
    if not texts:
        return [], {}
    opts = [score(t, safe=(safe is not None and t == safe), history=history,
                  grounded=grounded, key=f"c{i}") for i, t in enumerate(texts)]
    ranked = choosing.weigh(opts, mod)
    by_key = {o.key: t for o, t in zip(opts, texts)}
    ordered = [by_key[o.key] for o, _ in ranked]
    detail = dict(choosing.explain(mod))
    detail["narrowed_to"] = len(ordered)
    detail["scored"] = [{"text": by_key[o.key], "score": round(s, 3), "safe": bool(o.meta.get("safe")),
                         "value": round(o.value, 3), "risk": round(o.risk, 3),
                         "novelty": round(o.novelty, 3), "cost": round(o.cost, 3),
                         "social": round(o.social, 3)}
                        for o, s in ranked]
    return ordered, detail
