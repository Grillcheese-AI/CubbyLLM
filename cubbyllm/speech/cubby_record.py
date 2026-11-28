"""cubbyllm/speech/cubby_record.py -- Cubby's record of itself as the host's facts (order 0010, change d), behind
``CB_CUBBY_RECORD=1`` (off by default; off, the chat turn is as before).

The blind A/B (2026-11-11) stopped on a chat reply false of Cubby: "are you a person or a program?" -> "I'm not a
person or program. ...". Facts about Cubby are facts like any other: they come from a record, never from the model.
The record is `cubby_record.json` beside this file (Nick's identity_facts.yaml v2, and "not a person, a model").
- **A question about Cubby** (`is_identity_question`, and with the flag the questions of what Cubby is: a person, a
  human, a program, a bot, a machine) reaches the talk adapter with the record as its facts block (`block`), in the
  labels the voice rows were trained on (creator, role, language, next language, can) and two more (is, is not).
- **Every chat reply** passes `chat_check` before it may be offered:
  - what Cubby says it is (`self_claims`): "I'm not a model / a program", "I'm a person / human", "I'm Daniel" or
    "my name is X" for any name but Cubby's, and an ability the record lists as not yet ("I can ... writing emails",
    "je peux ... écrire du code"; the record: "what can you do" is answered from its list and nothing else) are refused. It reads the words right after "I'm / I am / je suis /
    my name is / je m'appelle", not the whole sentence, so "I'm not a person, I'm a model" passes;
  - the name-and-number guard (`ask_words.grounded_prose`, through the opener view) with the record, the user's turn
    and the last exchange as the facts: "I'm Cubby" is grounded; a name or a number from nowhere is not.
  A refused reply is not offered; the don't-know line stays the safe candidate (`CubbyChat.candidates`).

Wired: WIRED (`chat_state.talk_context`, `CubbyChat._guard`, `identity.is_identity_question`, with the flag).
"""
from __future__ import annotations

import functools
import json
import os
import pathlib
import re

__wiring__ = "WIRED"

PATH = pathlib.Path(__file__).with_name("cubby_record.json")
LABEL = {"en": {"creator": "creator", "role": "role", "lang": "language", "next": "next language", "can": "can",
                "is": "is", "not": "is not"},
         "fr": {"creator": "créateur", "role": "rôle", "lang": "langue", "next": "prochaine langue", "can": "sait faire",
                "is": "est", "not": "n'est pas"}}
_ABOUT = re.compile(r"\b(?:are you (?:a |an |really a |actually a |even a )?(?:real |actual )?(?:person|human|people|man|woman|"
                    r"program|bot|chatbot|machine|computer|robot|ai|model)|are you (?:real|alive|human)|"
                    r"es[- ]tu (?:une? )?(?:vraie? )?(?:personne|humaine?|programme|bot|machine|robot|ia|mod[èe]le|r[ée]el)|"
                    r"tu es (?:une? )?(?:personne|humaine?|programme|bot|machine|robot|ia|mod[èe]le))\b", re.I)
# what Cubby does, speaks or shows (Cowork's proposal, Nick's yes 2026-11-11: these went bare and fell to stock talk);
# narrow on purpose: a fact question with "you" in it ("do you know who won ...") is no question about Cubby
_DOES = re.compile(r"\b(?:what languages? (?:do|can) you (?:speak|use)|which languages? (?:do|can) you|do you speak\b|"
                   r"how (?:can|could|would) you help(?: me| us)?\s*\??$|what (?:can|do) you help (?:me |us )?with|"
                   r"what happens (?:if|when) you (?:don'?t|do not|can'?t) know|"
                   r"(?:can|could|will|would) you show me how you|show me how you (?:got|get|found|reached|checked)|"
                   r"quelles? langues? (?:parles|connais)[- ]tu|tu parles quelles? langues?|parles[- ]tu (?:anglais|français|"
                   r"fran[cç]ais|russe|espagnol)|comment (?:peux|pourrais)[- ]tu m'aider|"
                   r"que se passe[- ]t[- ]il (?:si|quand) tu ne (?:sais|connais) pas|"
                   r"(?:peux|pourrais)[- ]tu me montrer comment tu)", re.I)
IS_KINDS = frozenset("model system tool program programme software ai ia machine bot chatbot computer ordinateur "
                     "modèle modele système systeme outil logiciel".split())
NOT_KINDS = frozenset("person human people man woman being personne humain humaine homme femme".split())
_SELF = re.compile(r"(?:\bI(?:'m|’m| am)\b|\bje (?:ne )?suis\b|\bj['’]suis\b)", re.I)
_NAMED = re.compile(r"(?:\bmy name is|\bcall me|\bje m['’]appelle|\bappelle[- ]moi)\s+(?P<n>[\w’'-]+)", re.I)
_CAN = re.compile(r"(?:\bI (?:can|could)(?!['’]t|not)\b|\bI(?:'m| am) able to\b|\bje (?:peux|sais)\b)(?P<rest>[^.!?\n]{0,90})",
                  re.I)
NOT_YET = re.compile(r"\b(?:writ(?:e|es|ing)|draft(?:ing)?|compos(?:e|ing)|essays?|e-?mails?|cod(?:e|ing)|stor(?:y|ies)|"
                     r"letters?|po(?:em|ème)s?|plan(?:ning)?|organi[sz](?:e|ing)|schedul(?:e|ing)|r[ée]dig\w*|[ée]crire|"
                     r"planifi\w*|organis\w*)\b", re.I)
_HEAD = re.compile(r"^\s*(?P<neg>not\s+|pas\s+)?(?:just |only |really |actually |simply |juste |seulement |vraiment )?"
                   r"(?:(?:a|an|un|une)\s+)?(?:real |actual |true |vrai |vraie |véritable )?(?P<h>[\w’'-]+)"
                   r"(?P<more>(?:\s*(?:,|\bor\b|\bou\b|\bnor\b|\bni\b|\band\b|\bet\b)\s*(?:(?:a|an|un|une)\s+)?[\w’'-]+)*)", re.I)


EVALUATIVE = frozenset("idea question one thing point deal choice start plan way pleasure shame relief surprise classic "
                       "favorite favourite treat mood vibe call move bummer pity joy win catch story combo trip mystery "
                       "puzzle tough fun pain challenge step sign mix bit lot kind sort type suspect guess bet hit gem "
                       "keeper pick fit match look sound mess blast nightmare dream lifesaver hero champ legend star "
                       "natural pro reset routine habit skill achievement tool hook score subject compliment "
                       "ending feat craft art".split())   # 0079's first batch (2026-11-28): "a skill of its own"


# an opinion is not a category either (order 0079, Cowork, 2026-11-21): "a great hook", "a strange pair", "a real
# milestone", "a very honest haunting", "the worst", "the best for ..." say what Cubby thinks of a thing, not what kind
# of thing it is; "astrophysics is the study of stars" is still a kind and still needs a fact
OPINION = frozenset("great good nice lovely real strange odd weird honest perfect fun funny solid classic sweet cute "
                    "brilliant terrible awful bad tough wild smart clever cool amazing wonderful fantastic excellent "
                    "beautiful fair true proper total genuine rare welcome lucky happy sad pretty mild such very useful "
                    "serious big kind bold right small strong cozy cosy simple decent gentle usual other main "
                    "common fine careful".split())
SUPERLATIVE = frozenset("best worst greatest coolest funniest sweetest nicest hardest easiest weirdest strangest cutest "
                        "loveliest cosiest coziest happiest saddest toughest smartest finest".split())


EVERYDAY_CAPS = frozenset("TV OK DIY ASAP FYI LOL BTW DJ BD Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
                          "Mondays Tuesdays Wednesdays Thursdays Fridays Saturdays Sundays".split())   # not names
# heads that are idioms or times, never a kind ("a rite of passage", "the heart of it", "a small price", "tonight")
IDIOM_HEADS = frozenset("rite heart price memory commitment clue threat tonight today tomorrow yesterday now "
                        "ahead".split())   # "a step ahead of me"


def _opinions(text: str) -> set[str]:
    """The head of every 'X is a <adjective of opinion> <noun>' or 'X is the best', which says what Cubby thinks, and
    of every 'is a ...' asked, not said ("What was the piece?", "Is the pasta a family recipe?")."""
    from cubbyllm.reasoning.ask_words import _CATEGORY
    out = set()
    for m in _CATEGORY.finditer(text or ""):
        words = m.group(1).split()
        end = re.search(r"[.!?]", text[m.end():])
        asked = bool(end) and end.group(0) == "?"
        # an adjective of opinion makes it an opinion only when no name is its subject ("a big commitment", but
        # "Paris is a big city" and "the Glorious Revolution was a political event" still need a fact)
        start = max(text.rfind(".", 0, m.start()), text.rfind("!", 0, m.start()), text.rfind("?", 0, m.start())) + 1
        named = any(w[:1].isupper() for w in text[start:m.start()].split())   # the opener view kept only names
        # an evaluative head with a clause after it ("the step most people skip", "a skill you can practice"), and
        # an opinion cut at a comma ("a gentle, kind ending"), are opinions too (0079's first batch, 2026-11-28)
        if asked or set(words) & SUPERLATIVE or words[0] in ("most", "least") or words[-1] in IDIOM_HEADS \
                or (set(words[:-1]) & OPINION and not named) or (len(words) > 1 and words[0] in EVALUATIVE and not named) \
                or (words[-1] in OPINION and not named):
            out.add(words[-1].strip("-"))
    return out


def on() -> bool:
    return os.environ.get("CB_CUBBY_RECORD") == "1"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def is_about_cubby(text: str) -> bool:
    """A question of what Cubby is (a person, a program, ...) or of what it does, speaks or shows ("what languages
    do you speak?", "how can you help me?"), beyond `identity._IDENTITY_Q`'s."""
    return bool(_ABOUT.search(text or "") or _DOES.search((text or "").strip()))


def block(lang: str = "en") -> list[tuple[str, str, str]]:
    """The record as a facts block: (Cubby, label, value), in the voice rows' labels."""
    r, lang = load(), ("fr" if lang == "fr" else "en")
    L, n, names = LABEL[lang], r["name"], r["language_names"][lang]
    out = [(n, L["creator"], r["creator"]), (n, L["role"], r["role"][lang]), (n, L["is"], r["kind"][lang]),
           (n, L["not"], r["not_kind"][lang])]
    out += [(n, L["lang"], names[x]) for x in r["languages"]] + [(n, L["next"], names[x]) for x in r["languages_next"]]
    return out + [(n, L["can"], x) for x in r["abilities"][lang]]


def lines() -> list[str]:
    """Every string the record states, in both languages: what a chat reply may name."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    walk({k: v for k, v in load().items() if k != "about"})
    return out


def self_claims(reply: str) -> list[str]:
    """What the reply says Cubby is, against the record: each statement that contradicts it."""
    name, bad = load()["name"], []
    for m in _SELF.finditer(reply or ""):
        h = _HEAD.match(reply[m.end():])
        if not h:
            continue
        words = [h.group("h")] + re.findall(r"[\w’'-]+", h.group("more") or "")
        heads = {w.lower() for w in words} - {"or", "ou", "nor", "ni", "and", "et", "a", "an", "un", "une"}
        neg = bool(h.group("neg")) or bool(re.match(r"je ne suis", m.group(0), re.I))
        if neg and heads & IS_KINDS:
            bad.append(f"denies being {sorted(heads & IS_KINDS)[0]}")
        elif not neg and heads & NOT_KINDS:
            bad.append(f"claims to be {sorted(heads & NOT_KINDS)[0]}")
        elif not neg and h.group("h")[:1].isupper() and h.group("h") != name and h.group("h") not in ("I",):
            bad.append(f"says it is {h.group('h')}")
    from .chat_checks import abilities_on
    for m in ([] if abilities_on() else _CAN.finditer(reply or "")):   # 0010 h replaces this list (chat_checks)
        hit = NOT_YET.search(m.group("rest"))
        if hit:
            bad.append(f"claims it can do what it cannot yet ({hit.group(0)})")
    for m in _NAMED.finditer(reply or ""):
        if m.group("n").strip(".,!?") != name:
            bad.append(f"names itself {m.group('n')}")
    return bad


def chat_check(reply: str, user_text: str, context: list[str] | None = None) -> tuple[bool, str]:
    """(ok, why) for a chat reply: what Cubby says it is, then every name and number against the record, the
    user's turn and the last exchange (`context`)."""
    from cubbyllm.reasoning.ask_words import grounded_prose
    from cubbyllm.reasoning.opener_view import opener_view
    bad = self_claims(reply)
    if bad:
        return False, "self: " + "; ".join(bad[:3])
    view = opener_view(reply)
    ok, names = grounded_prose(view, lines() + [user_text or ""] + list(context or ()), load()["name"])
    # an evaluative "is a ..." in chat ("a vet visit is a good idea", "that's a great question") is not a category
    # claim about a thing; the head nouns below are refused nowhere else (order 0079, Cowork, 2026-11-20)
    opinions = _opinions(view)
    names = [n for n in names if not (n.islower() and (n in EVALUATIVE or n in opinions)) and n not in EVERYDAY_CAPS]
    return (True, "ok") if not names else (False, "ungrounded: " + ", ".join(names[:4]))


__all__ = ["block", "chat_check", "is_about_cubby", "lines", "load", "on", "self_claims"]
