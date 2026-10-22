"""cubbyllm/speech/chat_state.py -- the chat turn's state: the hormones and their modulation, a nudge, the user's
state and emotion, the talk context and the history it carries. Split out of chat.py (order 0031) as a mixin of
CubbyChat, verbatim.

Wired: WIRED (CubbyChat).
"""
from __future__ import annotations

import re
from .identity import (T, affect_block, derived, identity_system, load_facts,  # noqa: E402
                      guess_lang, is_identity_question, is_identity_reply, is_model_guard, voice_ok)
from ..affect.neurochem import Neurochemistry, appraise  # noqa: E402

__wiring__ = "WIRED"


class ChatState:
    """CubbyChat's state (a mixin; order 0031)."""

    # ── hormones: the cubemind neurochemistry ODE is the state source ───────
    def _hormones(self) -> dict:
        d = self.chem.to_dict()
        return {h: round(d[h], 3) for h in ("dopamine", "serotonin", "cortisol",
                                            "oxytocin", "noradrenaline")}

    def set_state(self, state: dict) -> None:
        """Pin an explicit state (tests / replaying a recorded trajectory);
        the next nudge() resumes from the ODE, not from this override."""
        self.state = dict(state)

    def _modulation(self) -> dict:
        """The seven knobs the filter runs on.

        From the ODE, EXCEPT when a state has been pinned that the ODE is not
        currently in (`set_state`, a replayed trajectory, a state handed to
        `__init__`). Then the pinned hormones win, because that state is what
        went into the model's system prompt and the body the filter uses has
        to be the body the words came from. Two sources for one turn is how
        `wariness` went wrong — a second opinion about what alarm means.

        `pain` and `craving` are not in the five-hormone serving slice, so a
        replayed state carries neither. That is the honest reading: a recorded
        trajectory records hormones, not injury.
        """
        live = self._hormones()
        if all(abs(live[h] - float(self.state.get(h, live[h]))) < 1e-9 for h in live):
            return self.chem.modulation()
        probe = Neurochemistry()
        for h, v in self.state.items():
            if hasattr(probe, h):
                setattr(probe, h, float(v))
        return probe.modulation()

    def nudge(self, user_text: str, *, gap_s: float | None = None, burst: int = 0,
              hour: int | None = None) -> dict:
        """One message through appraisal -> the ODE (a couple of perception
        frames), then the 5-hormone slice becomes the serving state.

        THE READ AND THE TRANSFER ARE TWO STEPS, and keeping them apart is the
        point. The appraiser reports how the USER is; `afferent` decides what
        that may do to CUBBY. They used to be one step, which is how a sad user
        made Cubby sad and an angry one made him frightened.

        The split of the five drives is by what they are about:

          novelty, threat   the TASK. A message can be urgent with nobody upset,
                            and `appraise`'s threat lexicon reads exactly that.
          focus             both: the task asks something, and concern attends.
                            Whichever is larger.
          valence, social   the PERSON, so they go through the transfer and
                            come out as concern and warmth rather than as a copy.
          surge             arousal contagion, which only exists on this path.

        `gap_s`/`burst`/`hour` are the behavioural channel and do nothing at
        all until the user grants it (`self.afferent.grant()`).
        """
        if self.appraiser is not None:                   # v5: the trunk reads the emotion, its petal drives
            self.signals = self.appraiser.signals(user_text, self._seen_vocab, guess_lang(user_text))
        else:
            self.signals = appraise(user_text, self._seen_vocab)
        self._seen_vocab.update(re.findall(r"[\w']+", user_text.lower()))

        self.afferent.read(user_text, lexical_valence=self.signals.get("valence", 0.0),
                           warmth=self.signals.get("social", 0.0), gap_s=gap_s, burst=burst,
                           hour=hour, words=len(user_text.split()))
        tr = self.afferent.drives()
        self.chem.step_message({
            "novelty": self.signals.get("novelty", 0.0),
            "threat": self.signals.get("threat", 0.0),          # the task's danger, never the user's mood
            "focus": max(self.signals.get("focus", 0.0), tr["focus"]),
            "valence": tr["valence"],                           # 0.0 when they are having a bad time
            "social": tr["social"],                             # ... which is where it went instead
            "surge": tr["surge"]})
        self.state = self._hormones()
        return self.state

    def user_state(self) -> dict:
        """What Cubby holds about the person, in full. Small enough to show
        them, which is the only honest way to keep it."""
        return self.afferent.state()

    @property
    def emotion(self) -> str:
        return self.chem.dominant_emotion

    def talk_context(self, user_text: str, lang: str) -> dict:
        """The structured side of the turn (2026-10-17, for the 450M talk adapter, which renders its own
        formats): an identity question comes with who Cubby is as a facts block; any other turn as the
        conversation's turns, the last spoken exchange included when history is on. The stand-in ignores
        both keys and reads the prompt."""
        if is_identity_question(user_text):
            name = T(self.facts, "name", lang)
            facts = [(name, k.replace("_", " "), T(self.facts, k, lang)) for k in ("name", "builder", "tagline", "how_it_answers", "when_unsure")
                     if k in self.facts and isinstance(T(self.facts, k, lang), str)]
            return {"facts": facts, "question": user_text}
        turns = []
        if self.history_turns and self.history:
            last = self.history[-1]
            u1, a1 = str(last.get("user") or "").strip(), str(last.get("reply") or "").strip()
            if u1 and a1:
                turns += [("user", u1), ("assistant", a1)]
        return {"turns": turns + [("user", user_text)]}

    def with_history(self, user_text: str) -> str:
        """The prompt for this turn: the previous spoken exchange inline (v9 multi-turn shape), or the bare
        turn when there is none / history is off. An identity question stands alone: it is answered from
        who Cubby is, not from the last exchange."""
        if not self.history_turns or not self.history or is_identity_question(user_text):
            return user_text
        last = self.history[-1]
        u1, a1 = str(last.get("user") or "").strip(), str(last.get("reply") or "").strip()
        if not u1 or not a1:
            return user_text
        lang = "fr" if guess_lang(user_text) == "fr" else "en"
        return self.MULTI_PROMPT[lang].format(u1=u1, a1=a1, u2=user_text)
