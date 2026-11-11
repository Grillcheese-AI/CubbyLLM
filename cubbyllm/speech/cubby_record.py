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
    ok, names = grounded_prose(opener_view(reply), lines() + [user_text or ""] + list(context or ()), load()["name"])
    return (True, "ok") if ok else (False, "ungrounded: " + ", ".join(names[:4]))


__all__ = ["block", "chat_check", "is_about_cubby", "lines", "load", "on", "self_claims"]
