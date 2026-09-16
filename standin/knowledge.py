"""knowledge — the worlds outside his own that can answer a question.

Wired: WIRED (mounted on `CubbyMan.other_worlds`; asked through `hypothesis.ask_verifier`).

WO-2.13. Nick's example, verbatim:

    "the science world is having gravity inside, cubby dont know it tries stuff
     then all of a sudden oh... let me ask the world: 'how can I know when
     something is about to fall on me?' then the science world sends the gravity
     + attraction laws so it understands... same for the coding world, if it
     needs to code something, it will ask from the coding world: 'I need to do
     write an hello world' then the coding world will reply with the right code
     for it (in the chosen language), then cubby stores it in long term memory
     so it knows that part and dont have to ask already."

Two worlds, because one would prove nothing about the shape:

  PHYSICS   answers with LAWS. A law generalises — it is worth holding because
            it applies to the next falling thing, not only to this one.
  CODING    answers with an ARTEFACT, which is the interesting case: what it
            hands back is a claim, and the `runner` verifier can check it by
            running it before he holds it. A world that knows is still not an
            authority.

What these worlds may NOT contain, and the reason the distinction is the whole
design: no world here knows anything about the board. `PhysicsWorld` has never
heard of cubby-man, cannot name a place in it, and does not know a thing is
falling right now — it knows how falling WORKS. Handing him "a rock is two
above you" would be the oracle back; handing him "a thing above you that is
nearer than it was is falling toward you" tells him what to look at and leaves
the looking to him.

Both are in-system: a dict of laws and a dict of snippets. No external calls —
*"the llm is only to build the dataset"*.
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


def _score(question: str, words: tuple[str, ...]) -> float:
    """How much of this question is in my vocabulary. Deliberately blunt: the
    routing only has to be good enough to pick between a handful of domains,
    and a too-clever matcher would claim questions it cannot answer."""
    q = set(re.findall(r"[a-z']+", question.lower()))
    if not q:
        return 0.0
    hit = sum(1 for w in words if w in q)
    # A domain word is specific — nobody says "gravity" or "falling" by
    # accident — so one is evidence and two is enough. Counting the fraction of
    # the QUESTION that is mine would punish a well-phrased question for having
    # ordinary words in it, which is backwards.
    return min(1.0, hit / 2.0)


class PhysicsWorld:
    """How things move when nothing is pushing them: falling, landing, weight.

    The law set is SMALL on purpose, and everything in it is true of any world
    where things fall at a steady rate. Acceleration is deliberately absent: it
    is true of the real world and false of a toy that drops one step per step,
    and a world that hands out a law its asker's world will refute is doing the
    asker no favours. If cubby-man ever accelerates its fallers, the law gets
    added here — earned, not assumed."""

    name = "physics"
    domain = "how things move on their own — falling, landing, weight, what pulls what"

    VOCAB = ("fall", "falls", "falling", "fell", "drop", "drops", "dropping", "above",
             "overhead", "land", "lands", "landing", "down", "downward", "gravity",
             "weight", "heavy", "light", "pull", "pulled", "sky", "ceiling")

    # each family: the words that ask for it -> the laws it answers with
    LAWS: list[tuple[tuple[str, ...], list[str]]] = [
        (("fall", "falls", "falling", "fell", "drop", "drops", "dropping", "above",
          "overhead", "gravity", "pull", "pulled", "down"),
         [
             "a thing with nothing under it falls",
             "a falling thing goes down and never back up",
             "a thing above me that is nearer than it was is falling toward me",
             "a thing above me that is one step up lands on me next",
             "stepping out from under a falling thing is enough, it does not follow me",
         ]),
        (("land", "lands", "landing", "where"),
         [
             "a falling thing lands straight below where it let go",
             "a falling thing stops when it lands",
         ]),
        (("weight", "heavy", "light", "heavier", "lighter"),
         [
             "a heavy thing and a light thing fall the same",
         ]),
    ]

    def covers(self, question: str) -> float:
        return _score(question, self.VOCAB)

    def answer(self, question: str) -> list[str]:
        q = set(re.findall(r"[a-z']+", question.lower()))
        out: list[str] = []
        for words, laws in self.LAWS:
            if q & set(words):
                out.extend(f for f in laws if f not in out)
        return out


class CodingWorld:
    """How to say a thing in a language. Answers with an artefact, not a law.

    What comes back is a CLAIM — `answer_with_test` hands over the code and the
    output it says the code produces, so the asker can settle it with the
    `runner` verifier instead of taking a world's word for it. A world that
    knows is not an authority; it is a better place to start than guessing."""

    name = "coding"
    domain = "how to write a thing in a programming language"

    VOCAB = ("code", "write", "program", "script", "print", "python", "hello",
             "world", "function", "language", "syntax", "how")

    SNIPPETS: dict[str, dict] = {
        "hello world": {
            "python": {"code": 'print("hello world")\n', "prints": "hello world"},
        },
        "count to three": {
            "python": {"code": "for i in range(1, 4):\n    print(i)\n", "prints": "1\n2\n3"},
        },
    }

    def covers(self, question: str) -> float:
        q = question.lower()
        if not any(t in q for t in self.SNIPPETS):
            return 0.0                              # I only claim what I actually have
        return _score(question, self.VOCAB)

    def _find(self, question: str) -> tuple[str, str, dict] | None:
        q = question.lower()
        lang = next((l for l in ("python",) if l in q), "python")
        for task, langs in self.SNIPPETS.items():
            if task in q and lang in langs:
                return task, lang, langs[lang]
        return None

    def answer(self, question: str) -> list[str]:
        found = self._find(question)
        if found is None:
            return []
        task, lang, s = found
        return [f"in {lang}, {task} is: {s['code'].strip()}"]

    def answer_with_test(self, question: str) -> dict | None:
        """The same answer, plus the test that would settle it: run it and see
        what it prints."""
        found = self._find(question)
        if found is None:
            return None
        task, lang, s = found
        return {"facts": self.answer(question), "code": s["code"], "expected": s["prints"],
                "task": task, "language": lang}
