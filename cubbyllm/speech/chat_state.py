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


def regulated_on() -> bool:
    """Order 0079: novelty habituates within a conversation and a friendly turn raises serotonin. Off by default."""
    import os
    return os.environ.get("CB_MOOD_REGULATION") == "1"


# 0079's second round (Nick: "one more", then "Balanced"): a calm, unthreatening exchange counts as slightly safe with
# nothing warm in it (SAFETY_BASE), and safety quiets attention's push on noradrenaline (FOCUS_SAFE). Measured on about
# 2,900 Topical-Chat conversations against 0 and 0 (the first round): mild anger 94% -> 52% of turns, strong anger 1%
# -> 0%, joy above 0 30% -> 63%; the upsetting messages still move the mood (12 -> 13, 8 -> 9 on their first turn).
SAFETY_BASE = 0.2
FOCUS_SAFE = 1.0
NOVELTY_PRIOR = 0.33
# CARING FOR SOMEONE IN TROUBLE IS NOT DANGER (0079, Cowork, 2026-11-21): concern (the transfer's warmth for a user who is
# having a bad time) raised attention with no serotonin behind it, which the cube reads as anger: with the serving read
# replayed over the pilot, anger +1 or more on 36 of the 44 turns where the user seemed nervous or afraid. Half the
# concern now counts as safety: with `mood.SIDE_MARGIN`, 8 of 44, and no joy shade on the 40 sad turns
# (exp_0079_mood_read).
CONCERN_SAFE = 0.5


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
        novelty, extra, focus_scale = self.signals.get("novelty", 0.0), {}, 1.0
        if regulated_on():                                       # order 0079, behind CB_MOOD_REGULATION=1
            # NOVELTY HABITUATES: what counts is novelty above what this conversation has been bringing (an EMA of
            # it), so a conversation that keeps bringing new words stops lifting dopamine and noradrenaline turn
            # after turn, and a real surprise still does.
            seen = getattr(self, "_novelty_seen", None)
            # a new conversation is EXPECTED to bring new words: the first turn is measured against the novelty an
            # ordinary turn brings (0.33, the mean over 97,204 Topical-Chat turns), not against none, which read
            # every opening message as a jolt (anger +2 on 83 of 99 openers; with this, 0 of 99)
            expected = NOVELTY_PRIOR if seen is None else seen
            self._novelty_seen = novelty if seen is None else 0.7 * seen + 0.3 * novelty
            novelty = max(0.0, novelty - expected)
            # SAFETY: the user's own warmth and good mood, with no threat in the turn, is contentment (serotonin)
            threat = self.signals.get("threat", 0.0)
            extra["safety"] = min(1.0, (SAFETY_BASE + max(0.0, self.signals.get("valence", 0.0))
                                        + max(self.signals.get("social", 0.0), CONCERN_SAFE * tr["social"]))
                                       * (1.0 - threat))
            focus_scale = 1.0 - FOCUS_SAFE * extra["safety"]    # attention in a safe exchange is calm attention
        self.chem.step_message({
            "novelty": novelty,
            "threat": self.signals.get("threat", 0.0),          # the task's danger, never the user's mood
            "focus": max(self.signals.get("focus", 0.0), tr["focus"]) * focus_scale,
            "valence": tr["valence"],                           # 0.0 when they are having a bad time
            "social": tr["social"],                             # ... which is where it went instead
            "surge": tr["surge"], **extra})
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
        from .cubby_record import block, on as record_on            # 0010 change d: Cubby's record as the block of a
        from .identity import asks_about_cubby
        from . import sheet_checks                                   # 0079 item 5 (CB_TALK_CHECKED=1): "how are you?" is a
        mood_only = sheet_checks.on() and sheet_checks.HOW_ARE_YOU.search(user_text or "")   # chat turn, not the record
        if record_on() and asks_about_cubby(user_text) and not mood_only:   # turn that asks about Cubby; a hello, a thanks or a
            return {"facts": block(lang), "question": user_text}     # goodbye alone goes as a chat turn (Cowork, 2026-11-11)
        elif asks_about_cubby(user_text) and not mood_only:
            name = T(self.facts, "name", lang)
            facts = [(name, k.replace("_", " "), T(self.facts, k, lang)) for k in ("name", "builder", "tagline", "how_it_answers", "when_unsure")
                     if k in self.facts and isinstance(T(self.facts, k, lang), str)]
            return {"facts": facts, "question": user_text}
        turns = []
        if self.history_turns and self.history:
            for prev in self.history[-self.history_turns:]:
                u1, a1 = str(prev.get("user") or "").strip(), str(prev.get("reply") or "").strip()
                if u1 and a1:
                    turns += [("user", u1), ("assistant", a1)]
        return {"turns": turns + [("user", user_text)]}

    def with_history(self, user_text: str) -> str:
        """The prompt for this turn: the previous spoken exchange(s) inline (v9 multi-turn shape), or the bare
        turn when there is none / history is off. A question about Cubby stands alone: it is answered from
        who Cubby is, not from the last exchange."""
        from .identity import asks_about_cubby
        if not self.history_turns or not self.history or asks_about_cubby(user_text):
            return user_text
        pairs = []
        for prev in self.history[-self.history_turns:]:
            u1, a1 = str(prev.get("user") or "").strip(), str(prev.get("reply") or "").strip()
            if u1 and a1:
                pairs.append((u1, a1))
        if not pairs:
            return user_text
        lang = "fr" if guess_lang(user_text) == "fr" else "en"
        if len(pairs) == 1:
            return self.MULTI_PROMPT[lang].format(u1=pairs[0][0], a1=pairs[0][1], u2=user_text)
        head = "Conversation jusqu'ici :" if lang == "fr" else "Conversation so far:"
        u_lbl, a_lbl = ("Utilisateur", "Cubby") if lang == "fr" else ("User", "Cubby")
        lines = [head] + [f"{u_lbl}: {u}\n{a_lbl}: {a}" for u, a in pairs] + [f"{u_lbl}: {user_text}"]
        return "\n".join(lines)
