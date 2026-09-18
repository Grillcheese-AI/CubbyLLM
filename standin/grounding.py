"""grounding - the speech guards: he may only say what his percepts support.

Wired: WIRED (the pac agent's speech path). STANDALONE of any world.

Split out of `pacman.py` because none of it is about a maze. `grounded_ok` asks
whether a sentence is supported by a percept record, `rephrase_ok` whether a
rewrite still says the same thing, `longest_run` whether it is copying rather
than speaking, `split_mood` separates a mood tag from the words. Give it a
different world's percept records and it works unchanged, which is the test of
whether a thing belonged in a file named after pac-man.
"""
from __future__ import annotations

import re
__wiring__ = "WIRED"


# the live game's hidden word per level and what each word NAMES (ostensive
# definition): collect its letters and the word is grounded -> cubby SAYS it
# when the percept recurs. Speech goes through CubbyTalk's ASK like all speech.
_WORDS = ["HELLO", "CUBBY", "MAZE", "LEARN", "GHOST", "POWER", "SMART"]


_WORD_CONCEPT = {"HELLO": "player_present", "CUBBY": "proud", "MAZE": "new_maze",
                 "LEARN": "discovery", "GHOST": "ghost_near", "POWER": "power_up",
                 "SMART": "solved"}


_CONCEPT_WORD = {v: k for k, v in _WORD_CONCEPT.items()}


# The THINGS this world contains. He may name one only when the step's record
# or the facts he learned in it mention it — exp_r37 caught him saying "without
# triggering the ghost" on a ghost-free level, which passed every other check
# because "ghost" is neither a figure nor a cell name. Invented entities are
# the same defect as invented numbers; this is the clause that says so.
_ENTITIES = {"ghost": ("ghost", "ghosts"), "pellet": ("pellet", "pellets"),
             "star": ("star", "stars"), "wall": ("wall", "walls"),
             "hazard": ("hazard", "hazards"), "trap": ("trap", "traps"),
             "edge": ("edge",), "level": ("level",)}



class _Safe(dict):
    """format_map helper: a missing field renders as ? instead of raising."""

    def __missing__(self, k):
        return "?"



_MOOD_TAG = re.compile(r"^\(([^)]{1,40})\)\s*")


_ECHO = re.compile(r"own words|first person|short sentence|keeping every|propres mots|premi\u00e8re personne|phrase courte|"
                   r"\breword|\brephras|here'?s the sentence|\bsure\b|\bcertainly\b|\bi understand you\b|\bi see, so\b|"
                   r"\byou'?re\b|\byou are\b|\byour\b|\byou\b|\bvous\b|\btu\b|\bton\b|\bta\b|\btes\b", re.I)



# SECOND PERSON IS A DEFECT IN A THOUGHT AND THE POINT OF A REPLY, so the two
# guards cannot share one pattern. `_ECHO` keeps both halves, for `grounded_ok`;
# `_INSTRUCTION` is the half that is wrong in anybody's mouth.
#
# THE SECOND LINE IS THE REPLY PROMPT'S OWN FRAMING, in both languages, and it
# is here because the first live run of the coach box spoke one: *"Je ne vois
# pas quelqu'un qui vient de te dire ça."* — the model answering the
# instruction instead of the player, and passing every other check because it
# invents no figure and names no cell. English got caught by the entries above
# and French did not, which is the whole argument for writing the framing of
# each prompt into the guard rather than trusting that one language's leak
# looks like another's.
_INSTRUCTION = re.compile(r"own words|first person|short sentence|keeping every|propres mots|"
                          r"première personne|phrase courte|\breword|\brephras|"
                          r"here'?s the sentence|\bsure\b|\bcertainly\b|\bi understand you\b|"
                          r"\bi see, so\b|"
                          r"somebody just said|answer them|do not invent|very short sentence|"
                          r"vient de te dire|r[eé]ponds-lui|n'invente rien|phrase tr[eè]s courte",
                          re.I)


def split_mood(line: str) -> tuple[str, str]:
    """'(at ease) Pellet 6/12, nice.' -> ('(at ease) ', 'Pellet 6/12, nice.'); no tag -> ('', line)."""
    m = _MOOD_TAG.match(line)
    return (line[:m.end()], line[m.end():]) if m else ("", line)



def _content_words(text: str) -> set[str]:
    """Stems (first five letters) of the words that carry content (>= 5 letters)."""
    return {w[:5] for w in re.findall(r"[a-z\u00e0-\u00ff]{5,}", text.lower())}



_NAME_RE = re.compile(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b")


# the shapes of a COPIED prompt rather than a spoken sentence: the record's
# own header, its field markers, a bullet, a markdown heading
_COPY = re.compile(r"here is what i just perceived|voici ce que je viens de perceptoir|"
                   r"voici ce que je viens de percevoir|^\s*#|\n\s*[-*]\s|"
                   r"\b(about|i am at|how i feel|pellets i can see|goal|tried|hit)\s*:", re.I | re.M)



# "level-1 cell 0-2-0" is 4 tokens and he must be able to say it; beyond that
# he is reciting. Measured on exp_r37's first run, where the copies ran 20+.
MAX_RUN = 6



def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9à-ÿ\-]+", s.lower())



def longest_run(a: str, b: str) -> int:
    """Longest run of consecutive tokens `a` shares with `b`, in order. A
    sentence in his own words reuses the names; it does not reproduce the
    record's word ORDER for long stretches."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0
    best = 0
    prev = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best



def grounded_ok(record_line: str, text: str, facts, learned: str = "", slack: int = 5) -> bool:
    """Is `text` something he could honestly say about this step?

    The referent is the PERCEPT RECORD, not a host-written sentence, so the
    test is factual rather than lexical (2026-09-15):

      * every number in the sentence is in the record, AND no number in the
        sentence is absent from it — the old check was one-directional and let
        an invented figure through as long as the host's own numbers survived;
      * same both ways for cell and move NAMES;
      * content words come from the record or from the facts he learned this
        step, plus room for the words a sentence needs to BE a sentence. The
        exact figures and names carry the anti-invention work; this clause
        only stops a reply that is mostly words from nowhere, so the budget
        is proportional (a fluent short sentence really does add about five);
      * it is a SENTENCE, not the record read back. exp_r37's first run
        scored 100% "spoke" while the model was reproducing the prompt
        verbatim — a copy passes every grounding test there is, because
        everything in it came from the record. So: no header, no `key:`
        field markers, no bullets, and no run of `MAX_RUN` consecutive
        tokens shared with the record in order.
      * the voice rules, no echo of the instruction, no second person, no
        question back, no base-model guard, not a bio.

    A sentence that fails is refused, not repaired: the host then states the
    record flatly. A refusal is a result; a made-up thought is a defect."""
    from identity import has_non_latin, is_identity_reply, is_model_guard, voice_ok
    from forge import numbers
    if not text or "?" in text or _ECHO.search(text) or has_non_latin(text):
        return False
    if _COPY.search(text):
        return False                                     # the record read back, not a thought
    n_words = len(text.split())
    if n_words > 40 or n_words < 2:
        return False
    if longest_run(text, record_line) > MAX_RUN:
        return False                                     # his own words, not the record's word order
    if set(numbers(text)) != set(numbers(text)) & set(numbers(record_line)):
        return False                                     # a figure the step did not contain
    if set(_NAME_RE.findall(text)) - set(_NAME_RE.findall(record_line)):
        return False                                     # a cell or move he did not meet
    ground = f"{record_line} {learned}".lower()          # everything he perceived this step, verbatim
    mine = _content_words(text)
    if len(mine - _content_words(ground)) > max(slack, int(0.7 * len(mine))):
        return False                                     # mostly words for things he has not perceived
    said = text.lower()
    for thing, forms in _ENTITIES.items():
        if any(re.search(rf"\b{f}\b", said) for f in forms) and thing not in ground:
            return False                                 # a thing this step did not contain
    return voice_ok(text, facts) and not is_model_guard(text) and not is_identity_reply(text, facts)



def _contiguous_in(needle: list[str], hay: list[str]) -> bool:
    """Is `needle` a run of consecutive tokens inside `hay`?"""
    if not needle or len(needle) > len(hay):
        return False
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


def parroting(heard: str, text: str) -> bool:
    """Is `text` just the player's own sentence handed back?

    THE COST OF LETTING THE REFERENT INCLUDE WHAT THEY SAID, and it showed up
    in the coach box's second live run: *"you can do it!!"* came back as
    *"You can do it!!"*. Every word of it is grounded — they are the player's
    words — so no invention check can see it, and it is not an answer.

    Narrow on purpose, because quoting IS most of answering. Refused only when
    the reply is a run of their sentence, or contains their whole sentence and
    adds no content word of its own. *"il n'y a rien à cacher, alors je ne
    lache pas la patate"* keeps their phrase and adds its own, and passes.
    """
    r, h = _tokens(text), _tokens(heard)
    if not r or not h:
        return False
    if _contiguous_in(r, h):
        return True
    return _contiguous_in(h, r) and not (_content_words(text) - _content_words(heard))


def reply_ok(record_line: str, heard: str, text: str, facts, learned: str = "", slack: int = 5) -> bool:
    """Is `text` something he could honestly say BACK TO A PERSON this step?

    `grounded_ok`'s sibling. Every difference follows from one thing: a
    thought is about the world, a reply is about the world AND the person who
    just spoke. So

      * the referent gains what they SAID. He may use their words back at
        them without that counting as invention, which is most of what an
        answer to a sentence consists of.
      * SECOND PERSON IS ALLOWED. In a thought it means the model started
        addressing the user instead of thinking, which is why `_ECHO` catches
        it; in a reply it is the grammatical point, and a guard that forbids
        it can only pass sentences that ignore the person standing there.
      * a question back is allowed. Handing the turn over is what somebody
        being talked TO does.
      * one word is enough. "noted." is a complete reply; `grounded_ok`'s
        two-word floor is there to catch a truncated thought.

    WHAT DOES NOT RELAX IS INVENTION, and that is the half that matters.
    Every figure and every cell or move name still has to appear in the
    referent, both directions, so he cannot tell the player there are three
    ghosts when there are two. A made-up reassurance is worse than a made-up
    thought, because somebody acts on it."""
    from identity import has_non_latin, is_identity_reply, is_model_guard, voice_ok
    from forge import numbers
    if not text or _INSTRUCTION.search(text) or has_non_latin(text):
        return False
    if _COPY.search(text):
        return False                                     # the record read back, not an answer
    if parroting(heard, text):
        return False                                     # their sentence read back, not an answer
    n_words = len(text.split())
    if n_words > 30 or n_words < 1:
        return False
    if longest_run(text, record_line) > MAX_RUN:
        return False
    # the referent: what he perceived, what he learned here, and what they
    # just said to him. Quoting the player is not inventing.
    ref = f"{record_line} {learned} {heard}"
    if set(numbers(text)) - set(numbers(ref)):
        return False                                     # a figure neither the step nor the player contained
    if set(_NAME_RE.findall(text)) - set(_NAME_RE.findall(ref)):
        return False                                     # a cell or move nobody mentioned
    mine = _content_words(text)
    if len(mine - _content_words(ref.lower())) > max(slack, int(0.7 * len(mine))):
        return False
    said = text.lower()
    ground = ref.lower()
    for thing, forms in _ENTITIES.items():
        if any(re.search(rf"\b{f}\b", said) for f in forms) and thing not in ground:
            return False                                 # a thing neither of them has brought up
    return voice_ok(text, facts) and not is_model_guard(text) and not is_identity_reply(text, facts)



def rephrase_ok(line: str, text: str, facts) -> bool:
    """Is `text` an acceptable rephrasing of the host's `line` (mood tag already
    stripped)? Every number and name kept, the voice rules, no base-model guard,
    not a bio, at most 40 words and under 2x the host line, no echo of the
    instruction, no second person, no question back, and at most ONE content
    word the host line did not have — the live trace's failures were invented
    meaning around kept numbers ('6 grains weighing 12 grams', '7 fingers')."""
    from forge import numbers
    from identity import has_non_latin, is_identity_reply, is_model_guard, voice_ok
    if not text or "?" in text or _ECHO.search(text) or has_non_latin(text):   # "建筑师: 8/15, nice." (Qwen arm, 2026-09-04)
        return False
    n_words = len(text.split())
    if n_words > 40 or n_words > max(12, 2 * len(line.split()) + 4):
        return False
    names = re.findall(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b", line)   # cells and move NAMES, not sentence-initial words
    if not (set(numbers(line)) <= set(numbers(text)) and all(n in text for n in names)):
        return False
    if len(_content_words(text) - _content_words(line)) > 1:
        return False
    return voice_ok(text, facts) and not is_model_guard(text) and not is_identity_reply(text, facts)
