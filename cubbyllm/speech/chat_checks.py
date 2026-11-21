"""cubbyllm/speech/chat_checks.py -- two checks on a chat reply (order 0010, changes g and h; approved by Nick 2026-11-11),
each behind its own flag, off by default; read in `CubbyChat._guard` after the checks there. A refused reply is not
offered, and the don't-know line stays.

- **g, a reply that loops is not spoken** (``CB_CHAT_NO_LOOP=1``, reason ``loop``): the generation stopped at its token
  limit; or a sentence or a line of three words or more is said twice; or the same run of one to six words is said
  three times in a row. Compared after folding case, spaces, punctuation and emoji.
- **h, what Cubby says it can do comes from its record** (``CB_CHAT_ABILITIES=1``, reason ``ability: <the words>``):
  - "I can ...", "I could ...", "I'm able to ...", "je peux ...", "je sais ..." is spoken when it says one of the
    record's can-do lines (its words are among that line's), or names no ability ("here's what I can do", "I can help
    you with that");
  - "I can't ...", "I cannot ...", "I'm not able to ...", "I don't have ...", "je ne peux pas ...", "je ne sais pas
    ...", "je n'ai pas ..." is spoken when the record says so: what Cubby does not do yet, or the host's don't-know
    lines;
  - with h on, it replaces change d's list of forbidden abilities for chat replies (`cubby_record.self_claims`).

Wired: WIRED (`CubbyChat._guard`).
"""
from __future__ import annotations

import functools
import os
import re

__wiring__ = "WIRED"

_WORD = re.compile(r"[a-zà-ÿœæ0-9]+(?:['’][a-zà-ÿ]+)?", re.I)
_UNIT = re.compile(r"[^.!?\n]+")
_CAN = re.compile(r"(?:\bI (?:can|could)(?!['’]t|not)\b|\bI(?:'m|’m| am) able to\b|\bje (?:peux|sais)\b)(?P<rest>[^.!?:\n]*)", re.I)   # a colon ends it (a list follows)
_CANT = re.compile(r"(?:\bI (?:can['’]t|cannot|can not|could ?n['’]?t)\b|\bI(?:'m|’m| am) not able to\b|"
                   r"\bI (?:don['’]t|do not) have\b|\bje ne (?:peux|sais) pas\b|\bje n['’]ai pas\b)(?P<rest>[^.!?:\n]*)", re.I)
_OFFER = re.compile(r"^\s*(?:(?:help|assist)(?:\s+(?:you|vous|toi))?(?:\s+(?:with|avec)\s+(?:that|this|it|anything|ça|"
                    r"cela|tout|whatever you need))?|do(?:\s+for\s+you)?(?:\s+today)?|faire(?:\s+pour\s+(?:toi|vous))?|"
                    r"t['’]aider|vous aider|te l['’]expliquer)\s*[:,]?\s*$", re.I)
STOP = frozenset("a an the to of for and or but i you me my your it this that be is are with on in at from by as so "
                 "just also really still yet now today here there what how which some any le la les un une de des du "
                 "et ou je tu te toi vous me mon ma mes ton ta tes ce cet cette que qui à a en pour avec sur dans pas "
                 "ne n encore them those one ça cela".split())
# held rows of the voice set (2026-11-11) showed h refusing true lines; these name no ability, or the record's own:
OFFER = frozenset("help assi offe offr give do serv aide t'ai fair donn prop inst aujo list voic star get sure need "
                  "anyt ever whic what ques ask shar part conf veri chec t'of tout".split())   # stems of an offer that names nothing
# ("what I can offer"), or that offers the checked facts themselves ("I can share this checked information": the
# record's "I answer questions from facts I have on record")
_TALK = re.compile(r"^\s*(?:(?:te|vous|you)\s+)?(?:talk|chat|parler|discuter|see why|understand|imagine|comprendre|voir pourquoi|"
                   r"(?:tell|dire)\s+(?:you\s+|te\s+|vous\s+)?(?:who|what|where|which|qui|ce qui|ce que|où|lequel))\b", re.I)
_CHECKED = frozenset("conf vouc veri sign back sett guar assu cert vali gara affi".split())   # "I can't confirm that
_SURE = re.compile(r"^\s*(?:tell|say|dire)\s+(?:you\s+|te\s+|vous\s+)?(?:if|whether|si|for sure|pour sûr)\b", re.I)
# claim": the record's "if I have no checked fact, I tell you so"
_CLAUSE = re.compile(r"[;,]|\s(?:but|mais|and I|et je)\s")      # an "I can" says its ability before the next clause
_IDIOM = re.compile(r"^\s*(?:agree more|wait|believe|thank you enough|stress enough|emphasi[sz]e enough|m['’]empêcher)\b", re.I)
# order 0079 (Cowork, 2026-11-20; the pilot's first 20 conversations): h refused the very honesty the record asks for.
# "I don't have anything on OAR", "I can't say much more", "je ne sais pas grand-chose de plus" are the record's
# "If I have no checked fact, I tell you so, and what is missing"; "I don't have hands for a teacup" is true of a model
# ("is: a model"). They pass. "I can't tell jokes", "I can't send emails" still do not.
_NOT_KNOWING = re.compile(
    r"^\s*(?:(?:anything|much|more|that|this|it|any\s+(?:info|information|facts?|details?|news|notes|"
    r"records?|data)|the\s+(?:detail|details|info|answer|facts?)|a\s+clear\s+picture|an\s+answer|enough|"
    r"hands|a\s+body|eyes|ears|a\s+mouth|taste\s+buds|a\s+stomach|a\s+home|pets?|a\s+family|kids|a\s+past|a\s+childhood)\b|"
    r"(?:say|tell\s+(?:you|for\s+sure|what|whether|if|which|how|much)|speak\s+to|confirm|be\s+sure|answer\s+that|"
    r"check\s+that|know\s+for\s+sure)\b|"
    r"(?:grand[- ]chose|beaucoup|tout|si\b|ce\s+qu|quoi|d['’]info|d['’]informations|de\s+détails|de\s+faits|ça\b|"
    r"de\s+mains|de\s+corps|(?:te\s+)?dire|confirmer|savoir))", re.I)
# order 0079 (Cowork, 2026-11-21; the pilot's next 50 conversations): h still refused true lines. What a model has not
# got ("I don't have friends the way you do", "I don't have a car", "je n'ai pas de famille", "I don't have one to root
# for"), what it cannot do without a body ("I can't watch anything myself", "I can't judge their music from here") and
# more ways of not knowing ("I don't have any book titles", "je n'ai pas plus de détails", "I can't match that horse to
# a name") are the record's "is: a model" and its don't-know line. They pass; "I can't tell jokes" still does not.
_MODEL_LACKS = re.compile(
    r"^\s*(?:(?:any|a|an|my\s+own|one)\s+)?(?:really\s+)?(?:favou?rite\s+)?(?:one\b|friends?|cars?|days?|favou?rites?|albums?|"
    r"teams?|phones?|jobs?|weekends?|commutes?|houses?|apartments?|kitchens?|gardens?|money|birthdays?|beds?|siblings|"
    r"parents|a\s+life|(?:\w+\s+)?(?:titles?|names?|suggestions?|recommendations?|specifics|examples?))\b|"
    r"^\s*(?:(?:really|actually|even|exactly|quite)\s+)?(?:judge|match|place|identify|pin\s+(?:\w+\s+)?down|watch|see|"
    r"hear|listen|taste|smell|touch|eat|drink|cook|drive|travel|go|visit|sleep|play|dance|hug|wear|check|look\s+(?:\w+\s+)?up|"
    r"fill\s+in|add\s+(?:much|more|anything)|weigh\s+in|get\s+(?:cold|hot|tired|sick|hungry|bored)|regarder|écouter|voir|"
    r"goûter|manger|boire|cuisiner|conduire|voyager|dormir|jouer|danser|vérifier)\b|"
    r"^\s*(?:plus\s+de|de\s+(?:titres?|noms?|suggestions?|famille|voiture|journée|équipe|maison|préférée?s?|favori(?:te)?s?|"
    r"canapé|cuisine|télé|livres?|chat|chien)|d['’](?:amis|exemples?|albums?|équipe|appart))|"
    # not knowing, said with what is missing: "a year for the school", "facts about sharks", "le détail de son rôle"
    r"^\s*(?:(?:a|an|the|any)\s+)?(?:\w+\s+){0,2}?(?:facts?|details?|years?|dates?|recommendations?|information|info|"
    r"notes?|data|answers?)\b|"
    r"^\s*(?:le|les|un|une|de|d['’])\s*(?:\w+\s+)?(?:détails?|infos?|informations?|recommandations?|dates?|années?|faits?)\b|"
    # what a model owns: "a lap", "a place to put it", "a TV memory of my own", "de livres à moi"
    r"^\s*(?:(?:a|an|any)\s+)?(?:laps?|couch|sofa|place\s+to|room\s+for|laundry|bags?|feet)\b|"
    r"^[^.!?]*\b(?:of\s+my\s+own|à\s+moi)\b", re.I)
# order 0079 (Cowork, 2026-11-21; the pilot's last batches): "I don't have ..." was still refused for one more thing a
# model has not got after another ("legs", "a tongue", "a hobby the way you do", "a game to win", "a TV or a couch",
# "anyone to tell", "a book in mind", "a solution on hand", "de coin précis à te proposer"). Saying what Cubby has not
# got is refused only when it denies what the record says Cubby has: its languages, its mood, access to its facts.
_HAS = frozenset("lang engl fren angl fran mood hume acce".split())   # stems: language, English, French, mood, access
# "I can't give you a real take on it", "I can't explain its meaning", "I can't really pick one": not knowing
_CANT_SAY = re.compile(r"^\s*(?:(?:really|quite|honestly)\s+)?(?:pick|choose|explain|recommend|be\s+(?:a|an)\s+\w+|"
                       r"give\s+(?:you\s+)?(?:a|an|any|much\s+of\s+an?)\s+(?:\w+\s+)?(?:take|answer|opinion|view|verdict|rule|"
                       r"recommendation|suggestion|tip|guess))\b", re.I)
# "je n'ai pas suivi le match": a past tense, not a thing Cubby has not got
_FR_PAST = re.compile(r"^\s*(?:encore\s+|jamais\s+|vraiment\s+|bien\s+)?(?!plus\b)[a-zà-ÿ]{2,}(?:é|ée|és|ées|i|is|it|u|us|ue|ert|eint)\b",
                      re.I)
_RELATIVE = re.compile(r"(?:\b(?:anything|nothing|much|more|whatever|all|any)\b[^.!?,;:\n]{0,40}|"   # "nothing on it I can share"
                       r"\b(?:glad|happy|pleased|nice)\s+|\bce\s+que\s+|"                    # "glad I could help", "ce que je sais"
                       r"\b(?:a|an|the|one|some|every)\s+(?:\w+\s+){0,2}?\w+\s+)$", re.I)        # "a plan I can get behind"
_CAN_HEAR = re.compile(r"^\s*(?:(?:only|still|just)\s+)?(?:hear|picture|imagine|guess|enjoy|sense|relate|feel\s+how|"
                       r"tell\s+you\s+what\s+I\s+have|"                    # "I can only tell you what I have": the record
                       r"see\s+(?:why|how|that|what|it|clearly)|get\s+behind|tell\s+(?:how|that|it['’]s|you['’]re))\b", re.I)
_KNOW_THAT = re.compile(r"^\s*(?:que|qu['’])", re.I)        # "je sais que X": knowing a fact, not an ability


def no_loop_on() -> bool:
    return os.environ.get("CB_CHAT_NO_LOOP") == "1"


def abilities_on() -> bool:
    return os.environ.get("CB_CHAT_ABILITIES") == "1"


def _fold(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(text or "")]


def loops(reply: str, hit_limit: bool = False) -> str | None:
    """Change g: why the reply loops, or None."""
    if hit_limit:
        return "loop: stopped at the token limit"
    seen = set()
    for u in _UNIT.findall(reply or ""):
        words = tuple(_fold(u))
        if len(words) >= 3:
            if words in seen:
                return "loop: said twice: " + " ".join(words)[:60]
            seen.add(words)
    w = _fold(reply)
    for n in range(1, 7):
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return "loop: three times in a row: " + " ".join(w[i:i + n])
    return None


def _stem(w: str) -> str:
    return w[:4]


def _content(text: str) -> set[str]:
    return {_stem(w) for w in _fold(text) if w not in STOP and not w.isdigit()}


@functools.lru_cache(maxsize=1)
def _lines() -> tuple[tuple, tuple]:
    """(the can-do lines, the can't lines) of the record and the host's own don't-know lines, as stemmed word sets."""
    from .cubby_record import load
    from .identity import load_facts
    r, f = load(), load_facts()
    can = [x for lang in ("en", "fr") for key in ("abilities", "ability_labels") for x in r.get(key, {}).get(lang, [])]
    cant = list(r["not_yet"]["en"]) + [f[k][lang] for k in ("dont_know_line", "when_unsure") if k in f for lang in ("en", "fr")]
    cant += list(f.get("refuses_to_claim", []))                    # feelings, consciousness: what Cubby has not
    words = lambda xs: tuple(frozenset(_stem(w) for w in _fold(x)) for x in xs)
    return words(can), words(cant)


def abilities(reply: str) -> list[str]:
    """Change h: each "I can / I can't" in the reply that the record does not hold."""
    from .cubby_record import NOT_YET
    can, cant = _lines()
    bad = []
    for m in _CAN.finditer(reply or ""):
        rest = _CLAUSE.split(m.group("rest"))[0]
        c = _content(rest)
        if _RELATIVE.search(reply[max(0, m.start() - 40):m.start()]) and not NOT_YET.search(rest):
            continue                                              # a relative clause, unless it says what is not yet
        if _OFFER.match(rest) or _TALK.match(rest) or _CAN_HEAR.match(rest) or _KNOW_THAT.match(rest) \
                or c <= OFFER or any(c <= line for line in can):
            continue
        bad.append(f"ability: {m.group(0).strip()[:60]}")
    for m in _CANT.finditer(reply or ""):
        rest = _CLAUSE.split(m.group("rest"))[0]
        c, head = _content(rest), [_stem(w) for w in _fold(rest) if w not in STOP][:1]
        said = m.group(0).lower().replace("’", "'")
        past = said.startswith("je n'ai") and _FR_PAST.match(rest)
        bare = {_stem(re.sub(r"^(?:qu|[dljnsmtc])'", "", w)) for w in _fold(rest)}   # "d'humeur" is "humeur"
        lacks = said.startswith(("i don't have", "i do not have", "je n'ai")) and not bare & _HAS
        if not c or past or lacks or _IDIOM.match(rest) or _SURE.match(rest) or _NOT_KNOWING.match(rest) \
                or _MODEL_LACKS.match(rest) or _CANT_SAY.match(rest) or set(head) & _CHECKED \
                or any(c <= line for line in cant):
            continue                                              # "je ne sais pas" alone; the record says so
        bad.append(f"ability: {m.group(0).strip()[:60]}")
    return bad


def check(reply: str, hit_limit: bool = False) -> str | None:
    """The reason the reply may not be spoken under g and h (each with its flag), or None."""
    if no_loop_on():
        why = loops(reply, hit_limit)
        if why:
            return why
    if abilities_on():
        bad = abilities(reply)
        if bad:
            return bad[0]
    return None


__all__ = ["abilities", "abilities_on", "check", "loops", "no_loop_on"]
