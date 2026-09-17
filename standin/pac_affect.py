"""pac_affect - how the maze reaches his body, and how a person does.

Wired: WIRED (mixed into `pacman.CubbyGhost`).

Two channels, and keeping them apart is the design. The WORLD may hurt him,
reward him, end his level or run out his clock - those raise `world.EVENTS` and
they are the only things allowed to reach `pain` and `reward`. A PERSON goes
through `coach`, which turns their words into concern and warmth rather than a
copy of their mood, and may reach `threat` only with a claim about the maze
that the maze will then score.

The naming is `feeling`'s and the chemistry is `neurochem`'s; what is here is
the wiring between them and this particular world - what a catch costs, what
being talked to is worth, and how far dopamine moved when it happened.
"""
from __future__ import annotations

from feeling import _PETAL, _SOCIAL_CORNERS, _plutchik  # noqa: E402
from pacworld import _manh  # noqa: E402

__wiring__ = "WIRED"


class AffectMixin:

    # What he is here for, in the words he was given it in.
    MISSION = "survive this maze at all cost"

    MISSION_PRESSURE = 0.22                              # the extra wanting it takes to risk it anyway

    MAX_EXPERIENCES = 400                                # the spoken-turn ring; nothing reads it yet


    CAUGHT_FEAR = 0.7                                    # the live game's ghost_penalty increment

    CAUGHT_SHOCK_FRAMES = 6                              # frames of full threat: NE surges, cortisol integrates

    FAIL_FRAMES = 5                                      # frames of deflation when the clock beats him

    CLEAR_FRAMES = 4                                     # frames of relief when he finally gets out


    def hurt_of(self, seen: int | None) -> float:
        """How much this one hurt, 0..1 — from how little warning he had.

        Being blindsided is worse than being run down after watching it come,
        and the last life is worse than the first two. Both are true of the
        SITUATION, not of his mood, which is what makes this an input to the
        chemistry rather than a reading of it."""
        warned = 0.0 if seen is None else min(1.0, seen / max(1, self.danger_radius))
        hurt = 0.55 + 0.35 * (1.0 - warned)
        if self.env.lives <= 1:                          # about to be, or just was, the last one
            hurt = max(hurt, 0.95)
        return round(min(1.0, hurt), 2)


    # ── somebody is watching, and talking ───────────────────────────────────
    HEARD_FRAMES = 3            # a sentence is not a shock; three frames, not six


    def _need(self) -> float:
        """How badly it is going, 0..1 — what decides whether a kind word is
        worth anything.

        Not a mood reading. `pain` and `craving` are the somatic signals the
        cube has no axis for, and `attempt` is the one thing neither of them
        knows: the third go at the same maze is worse than the first even when
        nothing has hurt him yet. Whichever is worst is the answer."""
        if self.chem is None:
            return 0.0
        pressure = min(1.0, max(0, self.env.attempt - 1) / 3.0)
        lives_gone = min(1.0, max(0, 3 - self.env.lives) / 3.0)
        return max(self.chem.pain, self.chem.craving, pressure, lives_gone * 0.8)


    def hear(self, text: str) -> dict:
        """The player says something while he plays.

        The drives come from `coach`, which decides what another person's words
        may do to a body — concern rather than contagion, `threat` only for an
        actual claim about the maze, never `reward`. This method's only job is
        to run them through the chemistry and let the corner label follow."""
        got = self.coach.hear(text, step=self.env.steps, need=self._need())
        d = {k: v for k, v in got["drives"].items() if v}
        if self.chem is not None and d:
            for _ in range(self.HEARD_FRAMES):
                self.chem.update(**d)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.0)
        self._t("heard", says=got["kind"], why=got.get("why"),      # `kind` is _t's own first arg
                credibility=round(self.coach.credibility, 2))
        return got


    def say_to_coach(self, got: dict, lang: str = "en") -> str:
        """What he says back. HOST-WRITTEN, and deliberately so for now.

        The model path (`_think` -> the VM's ASK -> the voice rules) is how he
        speaks about what he DID, and routing every "you can do it" through a
        generation is a lot of machinery for an acknowledgement. What matters
        is that the reply is a function of state rather than a canned string:
        a warning from somebody who has been right reads differently from the
        same words out of a voice that has cried ghost twelve times, and that
        difference is `credibility`, which he earned.

        No emotion word appears here, per the standing rule — `manner()` gives
        DELIVERY (clipped, slow, open) and never a feeling."""
        cred = self.coach.credibility
        if got["kind"] == "warn":
            if cred >= 0.65:
                line = "ok — looking." if lang != "fr" else "ok — je regarde."
            elif cred >= 0.35:
                line = "noted." if lang != "fr" else "noté."
            else:
                line = "you said that before." if lang != "fr" else "tu m'as déjà dit ça."
        elif got["drives"].get("social", 0) > 0.25:
            line = "heard." if lang != "fr" else "je t'entends."
        else:
            line = "mm." if lang != "fr" else "mm."
        m = self.chem.manner() if self.chem is not None else ""
        return f"{line} ({m})" if m else line


    DA_FULL_SWING = 0.35        # a dopamine move this size is "as much as it gets"


    def _da_moved(self, before: float | None) -> float:
        """How far dopamine actually travelled, 0..1 — the neuromodulator term.

        The third factor of a three-factor rule has to be MEASURED or it is not
        a third factor, it is a constant with a biological name on it. A catch
        that grazed him and a catch on his last life both call the same method
        and should not teach the same amount, and the difference between them
        is already sitting in the chemistry by the time this is read."""
        if before is None or self.chem is None:
            return 1.0                            # no body to measure: fall back to full credit
        return min(1.0, abs(self.chem.dopamine - before) / self.DA_FULL_SWING)


    def _coach_tick(self) -> None:
        """Settle any outstanding warning against WHAT WAS ACTUALLY THERE.

        `env.ghosts`, deliberately, and not `believed_ghosts()`. Everything he
        decides reads his beliefs, because a ghost he cannot sense is one he
        does not know about — but a warning is a claim that something is out
        there whether or not he can see it, and scoring it against his own
        beliefs would mean the player only ever gets credit for telling him
        what he already knew. The whole value of a warning is the case where
        the player can see further than he can."""
        env = self.env
        here = env.coords(self.place)
        near = min((_manh(here, g) for g in env.ghosts), default=99)
        self.coach.tick(env.steps, ghost_near=near <= self.danger_radius + 1)


    def _on_caught(self) -> None:
        """Being caught raises fear TWICE: the learned scalar (+0.7) and the
        neurochemistry — not one frame of threat but a shock of several, so
        noradrenaline surges and the slow cortisol actually moves; the
        compass reads it as fear, and the mood word follows for a while.

        It also records THE LESSON, which is what his danger radius is made
        of (2026-09-15, replacing a radius fitted to `fear` with thresholds
        nobody measured):

          * caught by a ghost he was tracking -> the berth must cover the
            distance he last saw it at;
          * caught by one he never sensed -> his senses failed him, so
            whatever berth he was keeping was too small: widen it by one.

        Being caught by something already inside his berth teaches nothing
        new about distance, and leaves the radius alone.

        The distance that matters is the one at DECISION time, not at
        capture: a ghost that has just caught him is by definition on top of
        him, so "it was 0 away" is a lesson worth nothing. `_threat_seen_at`
        is how far off the nearest ghost he believed in was when he last
        chose a move — the last moment he could have done something about
        it. (exp_r36, 200 steps: measuring at capture left the berth at 1
        through 8 catches in a row.)"""
        seen = getattr(self, "_threat_seen_at", None)
        hurt = self.hurt_of(seen)
        # An ambush is the only case with no measurement in it — his senses
        # failed and the distance he "last saw it at" does not exist. How much
        # it hurt is then the only signal available about how badly they
        # failed, so it is the only place pain sets the width. Where he DID
        # see it coming, the measurement stands and pain does not get to
        # inflate it: a berth fitted to how bad it felt is the `fear`-fitted
        # radius this replaced.
        # An ambush widens by one. It widens by TWO only for the worst kind
        # there is — blindsided on the last life — because that is the one
        # where the evidence says the width he just adopted was not enough
        # either. Letting a plain ambush widen by two put the berth at 3 on the
        # very first catch, which is not learning, it is flinching.
        widen = 2 if hurt >= 0.95 else 1
        lesson = self.danger_radius + widen if seen is None else max(seen, self.danger_radius)
        self.caught_at.append(max(1, min(self.MAX_DANGER_RADIUS, lesson)))
        self.fear = min(4.0, self.fear + self.CAUGHT_FEAR)
        self._last_hurt = hurt
        da_before = self.chem.dopamine if self.chem is not None else None
        if self.chem is not None:
            for _ in range(self.CAUGHT_SHOCK_FRAMES):
                self.chem.update(threat=1.0, valence=-0.8, pain=hurt)
            # the aversive outcome: a dopamine DIP (reward-prediction error), which the
            # port has no input for -- without it the NE->DA coupling lifts dopamine and
            # Lövheim's low-5HT/high-DA/high-NE corner reads as RAGE instead of fear
            self.chem.dopamine = max(0.15, self.chem.dopamine - 0.30)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.0)   # the corner label is set inside update()
        # and the world just labelled whatever was said in the run-up to it.
        # The magnitude is the MEASURED dopamine move, not a constant: this is
        # the third factor of the plasticity rule, and a catch that grazed him
        # should teach less than one that took the floor out.
        self.coach.outcome("caught", step=self.env.steps,
                           magnitude=self._da_moved(da_before))


    # ── the live frontend's schema ──────────────────────────────────────────
    def emotion(self) -> dict:
        """The compass reading. The Lövheim corner label when the state sits
        near a corner; in the mid-range (where the corner readout says
        'neutral' — e.g. sustained threat lifts NE AND, via the NE→DA
        coupling, dopamine) the petal comes from the ODE's own last appraisal
        (affect_arousal, valence) and the hormone deviations from resting."""
        chem = self.chem
        calm = {"name": "calm", "intensity": 0.0, "angle": 0, "color": "#dfe6ff", "msg": None}
        if chem is None:
            return calm
        name = chem.dominant_emotion
        if name in _SOCIAL_CORNERS:
            return calm
        if name not in _PETAL:
            ar, val = chem.affect_arousal, chem.valence
            # against where the hormones ACTUALLY settle, not the declared
            # resting levels — those differ, and with the old constants
            # oxytocin idled 0.115 above its supposed rest, so the `ot > 0.10`
            # clause fired on a body doing nothing and a quiet stretch came out
            # as "at ease" forever
            base = chem.quiescent() if hasattr(chem, "quiescent") else {"NE": 0.15, "DA": 0.30, "OT": 0.20}
            ne = chem.noradrenaline - base["NE"]
            da = chem.dopamine - base["DA"]
            ot = chem.oxytocin - base["OT"]
            # PAIN OUTRANKS EVERYTHING HERE (owner, 2026-09-15: *"he shouldn't
            # feel safe or right after being eaten by a ghost"*). It used to
            # sit below the oxytocin clause, so a step or two after a catch —
            # once the arousal spike had passed but the ache had not — a body
            # still hurting read as WARM, and the mood word came out "at
            # ease". Nothing about a high-arousal branch ordering was wrong;
            # what was wrong is that hurting was not a branch at all.
            # Corner names re-pointed at the corrected `_CORNERS` (2026-09-17).
            # `anxious` -> `fear`, `curious` -> `interest`, `sad` -> `distress`
            # when aroused and `spent` when flat: the cube now separates anguish
            # from depletion, so this fallback can too. `warm` had no corner
            # left — (1,1,0) is enjoyment/joy, not trust — and an oxytocin rise
            # with nothing else moving is the quiet end of joy, so it folds in
            # there rather than inventing a corner for itself.
            if getattr(chem, "pain", 0.0) > 0.25:
                name = "fear" if ne > 0.0 else "distress"
            elif ar >= 0.5 and val <= 0 and ne > 0.05:
                name = "fear"
            elif ar >= 0.5 and val > 0:
                name = "interest"
            elif ot > 0.10:
                name = "joy"
            elif da > 0.10 and val >= 0:
                name = "joy"
            elif da < -0.05 or val < -0.3:
                name = "distress" if ne > 0.0 else "spent"
            else:
                return calm
        petal, angle, pcolor, tiers = _PETAL[name]
        intensity = round(min(1.2, 0.6 * chem.arousal + 0.6 * chem.affect_arousal), 2)
        tier = tiers[0 if intensity < 0.4 else 1 if intensity < 0.8 else 2]
        meta = _plutchik().get(tier, {})
        return {"name": tier, "intensity": intensity, "angle": angle,
                "color": meta.get("color", pcolor), "msg": meta.get("message")}


    def affect(self) -> dict:
        env, chem, ev = self.env, self.chem, self._last
        believed = self.believed_ghosts()                # what he thinks is out there, not what is
        near = min((_manh(env.coords(self.place), g) for g in believed), default=99)
        scared = bool(believed) and env.frightened == 0 and near <= self.danger_radius
        da = chem.dopamine if chem is not None else 0.30
        c = chem.cortisol if chem is not None else 0.15
        mood = max(-1.0, min(1.0, (da - c) * 2.0))
        emo = self.emotion()
        return {"mood": round(mood, 2), "dopa": round(da / 0.5, 2), "cort": round(c / 0.4, 2),
                "satisfaction": round(chem.weight if chem is not None else 0.67, 2),
                "depressed": bool(da < 0.2 and mood < -0.3), "scared": scared,
                # relabeled in the lifted page: world model = facts / derived / walls
                "w_dopa": len(self.world), "w_cort": len(self.derived), "w_treat": len(self.walls),
                "memories": len(self.world),
                "emotion": emo["name"], "emotion_intensity": emo["intensity"],
                "emotion_angle": emo["angle"], "emotion_color": emo["color"], "emotion_msg": emo["msg"],
                "word": None, "collected": "",           # the letter mechanic is gone; the page's guard stays false
                "says": ev.get("says"), "thought": self._thought, "vocab": [], "talk": None,
                "lay_low": len(self.walls),              # relabeled: refused moves (walls learned)
                "pursuit": sum(1 for g in believed
                               if _manh(env.coords(self.place), g) <= self.danger_radius),
                # what he holds about whoever is watching. Shown, not hidden:
                # a number kept about a person belongs on their screen.
                "coach": self.coach.state()}
