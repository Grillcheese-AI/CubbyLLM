"""pac_speech - what he says about what he just did, and the gate it passes.

Wired: WIRED (mixed into `pacman.CubbyGhost`).

The division of labour this project keeps: the MODEL supplies the words, the
HOST checks them against the percept record, and a sentence that is not
supported by something he actually perceived is refused rather than repaired. A
refusal beats a wrong answer, so `say_flatly` exists and is not a failure mode.

The guards themselves live in `grounding` because none of them are about a
maze. What is here is the part that IS about this world: which percepts are
worth speaking about, how a thought is prioritised, and the prompt that carries
his state without carrying a feeling word.
"""
from __future__ import annotations

import random

from feeling import felt_names  # noqa: E402
from grounding import _Safe, grounded_ok, rephrase_ok, split_mood  # noqa: E402

__wiring__ = "WIRED"


class SpeechMixin:

    # ── thinking out loud ──────────────────────────────────────────────────
    # The MODEL says it; the host decides what is true. Each step hands the
    # model a PERCEPT RECORD — what he sensed, what he did, what the world did
    # back — and it answers in one sentence. The host then checks that every
    # number and every name in that sentence is in the record, and refuses
    # anything else. Voice rules apply. One thought per step wins by priority;
    # the salient ones also reach the page's bubble.
    #
    # Until 2026-09-15 the host wrote the sentence from THOUGHTS (below) and
    # the model only rephrased it, so what he "said" was a hand-written line
    # with a synonym swapped. Nick: *let it talk to see what it will do as it
    # receives information.* THOUGHTS is kept because `data/gap_families.py`
    # builds the verbalize SFT family out of it — templates are fine for
    # making a corpus, which is the one place an authored phrasing belongs.
    _PRIO = {"caught": 6, "level_up": 6, "hit_by_falling": 6, "out_of_time": 5, "program": 5,
             "modify": 5, "power": 4, "forge": 4, "flee": 4, "trapped": 5, "mine": 4, "dodge": 4,
             "wonder": 4, "chasing": 4, "superpower_move": 3, "eat": 3, "rest": 3, "predict": 3,
             "bump": 2, "probe": 2, "derive": 2, "plan": 1, "idle": 0}


    # what each event kind is ABOUT, in his own vocabulary. Not a phrasing:
    # a noun for the thing that happened, so the record reads as percepts
    # rather than as an opcode. The model does the language.
    _ABOUT = {"plan": "heading somewhere", "flee": "running from a ghost", "caught": "caught by a ghost",
              "bump": "walked into something solid", "probe": "tried a way that was not offered",
              "eat": "ate a pellet", "power": "ate a power star", "program": "invented a move",
              "modify": "changed one of my moves", "superpower_move": "used one of my own moves",
              "out_of_time": "ran out of time", "level_up": "cleared the level", "derive": "worked something out",
              "forge": "checked something with my own program", "idle": "moving on", "rest": "resting",
              "mine": "dropped a trap", "trapped": "a ghost hit my trap",
              "hit_by_falling": "something came down on me", "wonder": "something I cannot work out",
              "predict": "something above me is coming down", "dodge": "stepping out of the way",
              "chasing": "going after a ghost"}


    # several phrasings per event and language — NOT used at runtime any more.
    # `data/gap_families.py` builds the verbalize SFT family from this table.
    THOUGHTS = {
        "plan_pellet": {"en": ["I saw a pellet at {to} — heading there.", "Pellet spotted at {to}. Going for it.",
                               "There's one at {to}; that's my next stop."],
                        "fr": ["Je vois une pastille en {to} — j'y vais.", "Pastille repérée en {to}. J'y file.",
                               "Il y en a une en {to} ; prochaine étape."]},
        "plan_frontier": {"en": ["Nothing in sight; let me explore {to}.", "No pellets around — off to {to} to look.",
                                 "Unknown ground at {to}. Let's see what's there."],
                          "fr": ["Rien en vue, je vais explorer {to}.", "Pas de pastille par ici — direction {to}.",
                                 "Terrain inconnu en {to}. Voyons ça."]},
        "flee": {"en": ["A ghost {ghost_distance} cells away — too close for my nerves ({radius}), I'm running.",
                        "Ghost at {ghost_distance}. My comfort zone is {radius}; out of here.",
                        "That ghost is {ghost_distance} away. Not today — moving off."],
                 "fr": ["Un fantôme à {ghost_distance} — trop près pour mes nerfs ({radius}), je file.",
                        "Fantôme à {ghost_distance}. Ma zone de confort est {radius} ; je décampe.",
                        "Ce fantôme est à {ghost_distance}. Pas aujourd'hui — je m'écarte."]},
        "caught": {"en": ["Ouch, caught at {place}. I'll remember that spot is dangerous.",
                          "Got me at {place}. Lesson taken: that corner bites.",
                          "Caught at {place}… I'll keep a wider berth from now on."],
                   "fr": ["Aïe, attrapé en {place}. Je retiens que cet endroit est dangereux.",
                          "Il m'a eu en {place}. Leçon retenue : ce coin mord.",
                          "Attrapé en {place}… je garderai mes distances désormais."]},
        "probe": {"en": ["Tried {tried} — a wall. Noted.", "No way {tried}: a wall. Filed.",
                         "{tried}? Blocked. Good to know."],
                  "fr": ["J'ai essayé {tried} — un mur. Noté.", "Pas de passage vers {tried} : un mur. Classé.",
                         "{tried} ? Bloqué. Bon à savoir."]},
        "eat": {"en": ["Got a pellet ({score}/{total}).", "One more: {score} of {total}.", "Pellet {score}/{total}, nice."],
                "fr": ["Une pastille ({score}/{total}).", "Une de plus : {score} sur {total}.", "Pastille {score}/{total}, bien."]},
        "power": {"en": ["A star! The ghosts are mine for {steps} moves.", "Star! {steps} moves of payback.",
                         "Power up — for {steps} moves, they run from me."],
                  "fr": ["Une étoile ! Les fantômes sont à moi pendant {steps} coups.", "Étoile ! {steps} coups de revanche.",
                         "Boost — pendant {steps} coups, c'est eux qui fuient."]},
        "program": {"en": ["Made a new move, {name} — the VM checked my recipe {pattern}.",
                           "New move: {name}. Recipe {pattern}, and the VM says it holds.",
                           "I put together {name} ({pattern}); the VM signed off."],
                    "fr": ["Un nouveau coup, {name} — la VM a vérifié ma recette {pattern}.",
                           "Nouveau coup : {name}. Recette {pattern}, validée par la VM.",
                           "J'ai assemblé {name} ({pattern}) ; la VM a dit oui."]},
        "modify": {"en": ["Tweaked {parent} into {child} ({edit}).", "{parent} → {child}: {edit}.",
                          "Took {parent} and {edit}: meet {child}."],
                   "fr": ["J'ai retouché {parent} en {child} ({edit}).", "{parent} → {child} : {edit}.",
                          "J'ai pris {parent} et {edit} : voici {child}."]},
        "superpower_move": {"en": ["Using {name}: {saved} steps saved.", "{name}! {saved} steps for the price of one.",
                                   "Shortcut with {name} — {saved} saved."],
                            "fr": ["Coup spécial {name} : {saved} pas gagnés.", "{name} ! {saved} pas pour le prix d'un.",
                                   "Raccourci avec {name} — {saved} de gagnés."]},
        "out_of_time": {"en": ["Out of time on level {level} — attempt {attempt}. I need to be faster{learned_en}",
                               "Level {level}, attempt {attempt}: the clock beat me. Faster next time{learned_en}"],
                        "fr": ["Plus de temps au niveau {level} — essai {attempt}. Il me faut être plus rapide{learned_fr}",
                               "Niveau {level}, essai {attempt} : le temps m'a eu. Plus vite la prochaine fois{learned_fr}"]},
        "level_up": {"en": ["Level {cleared} cleared! On to {next}.", "That's level {cleared} done. Level {next}, here I come.",
                            "Cleared {cleared}. Next up: {next}."],
                     "fr": ["Niveau {cleared} terminé ! Au suivant : {next}.", "Et voilà le niveau {cleared}. Au {next} !",
                            "Niveau {cleared} bouclé. Prochain : {next}."]},
        "derive": {"en": ["Worked out: {fact}.", "So it follows: {fact}."],
                   "fr": ["J'en déduis : {fact}.", "Donc : {fact}."]},
        "forge_ok": {"en": ["Let me think this through… my check says: {answer}.", "Thinking… the check comes back: {answer}."],
                     "fr": ["Réfléchissons… mon calcul dit : {answer}.", "Voyons… le calcul répond : {answer}."]},
        "forge_bad": {"en": ["Let me think this through… my check does not hold, I'll go with my rule.",
                             "Thinking… that check fell apart; back to my rule."],
                      "fr": ["Réfléchissons… mon calcul ne tient pas, je m'en tiens à ma règle.",
                             "Voyons… ce calcul ne tient pas ; je reste sur ma règle."]},
        "idle": {"en": ["Carrying on toward {to}.", "Onward to {to}.", "Still heading for {to}."],
                 "fr": ["Je continue vers {to}.", "En route vers {to}.", "Toujours cap sur {to}."]},
        "rest": {"en": ["Tired ({energy}) — this spot looks safe, resting a moment.",
                        "Energy at {energy}. Nobody around; a short breather.",
                        "Running low ({energy}). Safe corner — I'll rest a bit."],
                 "fr": ["Fatigué ({energy}) — l'endroit est sûr, je me repose un instant.",
                        "Énergie à {energy}. Personne dans le coin ; petite pause.",
                        "À plat ({energy}). Coin tranquille — je souffle un peu."]},
        "mine": {"en": ["Dropping a trap here — whoever chases me will regret it ({left} left).",
                        "A trap on my way out. Follow me and see ({left} left).",
                        "Setting a trap right here. {left} left after this."],
                 "fr": ["Je pose un piège ici — celui qui me suit va le regretter (il m'en reste {left}).",
                        "Un piège en partant. Suis-moi pour voir (il m'en reste {left}).",
                        "Je place un piège ici même. Il m'en restera {left}."]},
        "trapped": {"en": ["Got one! A ghost walked into my trap.", "Trap sprung — one ghost sent home.",
                           "Told you. A ghost just met my trap."],
                    "fr": ["Et d'un ! Un fantôme est tombé dans mon piège.", "Piège déclenché — un fantôme renvoyé chez lui.",
                           "Je l'avais dit. Un fantôme vient de goûter à mon piège."]},
    }

    _thought_rng = random.Random(20260902)


    @classmethod
    def think(cls, kind: str, lang: str = "en", mood: str = "", pick: int | None = None, **d) -> str:
        """The line for one event kind (EN/FR): one of several phrasings,
        chosen at random (or `pick`, for tests)."""
        key = kind
        if kind == "plan":
            key = "plan_pellet" if d.get("goal") == "pellet" else "plan_frontier"
        elif kind == "forge":
            key = "forge_ok" if d.get("ok") else "forge_bad"
        variants = cls.THOUGHTS.get(key, {}).get("fr" if lang == "fr" else "en")
        if not variants:
            return ""
        learned = d.get("learned")
        d = dict(d, learned_en=(f": I made {learned}." if learned else "."),
                 learned_fr=(f" : j'ai inventé {learned}." if learned else "."))
        i = cls._thought_rng.randrange(len(variants)) if pick is None else pick % len(variants)
        line = variants[i].format_map(_Safe(d))
        return (mood + line) if line else ""


    # the compass tiers -> a mood word (EN, FR); calm / faint readings get none
    _MOOD = {"serenity": ("content", "serein"), "joy": ("glad", "content"), "ecstasy": ("thrilled", "ravi"),
             "acceptance": ("at ease", "à l'aise"), "trust": ("trusting", "confiant"), "admiration": ("grateful", "reconnaissant"),
             "apprehension": ("uneasy", "inquiet"), "fear": ("scared", "effrayé"), "terror": ("terrified", "terrifié"),
             "distraction": ("puzzled", "perplexe"), "surprise": ("surprised", "surpris"), "amazement": ("stunned", "sidéré"),
             "pensiveness": ("wistful", "songeur"), "sadness": ("down", "triste"), "grief": ("crushed", "abattu"),
             "boredom": ("bored", "blasé"), "disgust": ("fed up", "dégoûté"), "loathing": ("sick of it", "écœuré"),
             "annoyance": ("annoyed", "agacé"), "anger": ("angry", "en colère"), "rage": ("furious", "furieux"),
             "interest": ("curious", "curieux"), "anticipation": ("eager", "impatient"), "vigilance": ("alert", "aux aguets")}


    def _mood(self) -> str:
        """The mood prefix from the compass he already computes: the Plutchik
        tier (with its intensity) rather than a dopamine switch — so the same
        state that colors the compass colors the thought. Slow stress (high
        cortisol) shows as 'on edge' when the compass is not already afraid."""
        if self.chem is None:
            return ""
        fr = self.lang == "fr"
        emo = self.emotion()
        name, intensity = emo["name"], emo["intensity"]
        if self.chem.cortisol >= 0.35 and name not in ("apprehension", "fear", "terror"):
            return "(sur les nerfs) " if fr else "(on edge) "
        if name == "calm" or intensity < 0.25 or name not in self._MOOD:
            return ""
        en, fr_word = self._MOOD[name]
        return f"({fr_word}) " if fr else f"({en}) "


    # ── the percept record: what he hands the model to speak from ──────────
    SPEAK_FROM_PERCEPTS = True                           # False -> the old THOUGHTS templates (for A/B)

    # the lowest `_PRIO` that is worth an emit. Everything below states its
    # record flatly: one emit per step is the slow part of the live loop, and
    # a `plan`/`idle` record is the most repetitive thing he has — exp_r37
    # measured the keep rate falling from 45% to 25% when those were let
    # through, because a repetitive scene is the one a model recites.
    SPEAK_ABOVE = 2

    _SAY_FIELDS = ("to", "goal", "place", "tried", "hit", "score", "total", "steps", "energy",
                   "ghost_distance", "radius", "name", "pattern", "parent", "child", "edit",
                   "saved", "level", "next", "cleared", "attempt", "learned", "fact", "answer",
                   "left", "n", "seen_at", "in_steps", "times", "move", "claim", "why")


    def percepts(self, kind: str, **d) -> dict:
        """What just happened to him, as data — no phrasing, no template.
        This is the whole of what the model is given to speak from, so
        anything it says that is not in here is something it made up."""
        rec = {"about": self._ABOUT.get(kind, kind), "i am at": self.place}
        for k in self._SAY_FIELDS:
            v = d.get(k)
            if v not in (None, "", [], {}):
                rec[k.replace("_", " ")] = v
        seen = len(self.sighted - self._eaten_run)
        if seen:
            rec["pellets i can see"] = seen
        if self.mission:
            rec["what I am here for"] = self.mission
        if self.chem is not None:
            # THE BODY, NOT THE VERDICT (owner, 2026-09-15: *"we need to not
            # tell it how to interpret the hormonal changes. However we can
            # name emotions that suit with hormone A or B or C mixed with D to
            # guide the model in its speech"*).
            #
            # What used to be here was `how i feel: <one word>`, computed by
            # the host from the Lövheim corner and handed over finished. The
            # model then said the word back, which is why it read as a readout
            # with a feeling printed on it rather than as somebody feeling
            # something. Now the record carries what the state is LIKE, and a
            # short list of names that fit it — the opposed readings of ONE
            # region, because low dopamine under high cortisol is despair or it
            # is stubbornness and the chemistry does not decide which. Whatever
            # comes out is his reading, and the guard still holds him to the
            # record: he may pick a name that is offered, and he may not invent
            # a fact.
            sensations = self.chem.body(limit=2)
            if sensations:
                rec["my body"] = ", ".join(sensations)
            # Layer 3 names it: one corner, its intensity tier, and the dyad
            # when he is sitting on a boundary. An empty list is a real answer
            # and the common one — at the centre of the cube there is nothing
            # to name, and a body doing nothing should not be announcing a
            # feeling.
            names = felt_names(self.chem)
            if names:
                rec["could be"] = " or ".join(names)
        return rec


    # fields that exist to PROMPT the model, not to be said. They belong in the
    # guard's referent — he may use those words — and not in his mouth.
    _NOT_SPEECH = ("could be", "what I am here for")


    @staticmethod
    def render_percepts(rec: dict) -> str:
        """The record as a flat line: THE GUARD'S REFERENT. Everything the
        model is allowed to have drawn on, so `grounded_ok` can check a
        sentence against it. Generated from the data every time, so unlike a
        template it cannot contain anything the step did not."""
        return "; ".join(f"{k}: {v}" for k, v in rec.items() if k != "about") or str(rec.get("about", ""))


    @classmethod
    def say_flatly(cls, rec: dict) -> str:
        """What the HOST says when the model's sentence was refused.

        Not the same string as the guard's referent, which is the conflation
        this fixes. The referent has to carry every word he is licensed to use,
        including the candidate names — but those are a menu written FOR the
        model to choose from, and reading the menu out loud is not speech.
        Owner, live: *"(furious) i am at: level-3 cell 0-0-1; ... could be:
        elation or recklessness or being on a roll"* — the most redundant
        sentence in the run, and it was the fallback, not the model."""
        parts = [f"{k}: {v}" for k, v in rec.items()
                 if k != "about" and k not in cls._NOT_SPEECH]
        return "; ".join(parts) or str(rec.get("about", ""))


    def _speak_prompt(self, rec: dict) -> str:
        """Ask for a sentence ABOUT the record, not a rephrasing OF a line.

        Scene first, question second, on one line — the shape exp_r37 measured
        best (45% of attempts kept, against 25% for a bulleted `key: value`
        block and 0% for question-first). The failure modes are specific and
        worth keeping written down:

          * BULLETS come back as more bullets — the model continues the list.
          * QUESTION FIRST gets the instruction narrated back ("The user is
            asking me to respond in first person…").
          * SCENE FIRST gets an answer.

        The guard catches all three either way; this is about how often there
        is something worth keeping."""
        situation = self.render_percepts(rec)
        # HOW, not WHAT. The manner comes from the hormone dials and talks
        # about delivery — clipped, measured, flat — never about what he is
        # supposed to be feeling. A speaker told "you are anxious" says the
        # word anxious; a speaker whose sentence comes out short and wary
        # sounds anxious with the word nowhere in it.
        manner = self.chem.manner() if self.chem is not None else ""
        if self.lang == "fr":
            how = f" Ton ton : {manner}." if manner else ""
            return (f"Situation : {situation}.\n"
                    "Que penses-tu ? Réponds en une phrase courte, à la première personne, "
                    f"sans répéter la liste. N'invente rien que tu ne perçois pas.{how}")
        how = f" Your delivery: {manner}." if manner else ""
        return (f"Situation: {situation}.\n"
                "What are you thinking? Answer in one short sentence, first person, "
                f"without repeating the list back. Do not invent anything you do not perceive.{how}")


    verbalize = True                                     # let the LLM phrase the thought (content stays the host's)


    def _verbalize(self, line: str) -> tuple[str, bool]:
        """The model says the host's thought in its own words, under the
        hormonal state. The mood tag is the host's: stripped before the
        prompt, put back after. Accepted only by `rephrase_ok` (numbers,
        names, voice, no echo of the instruction, no second person, no
        invented content) — else the host line stands. -> (text, verbalized)."""
        if self.brain is None or not self.verbalize:
            return line, False
        from identity import identity_system
        mood, core = split_mood(line)
        fr = self.lang == "fr"
        prompt = ((f"Dis ceci avec tes propres mots, une phrase courte, à la première personne, en gardant "
                   f"chaque nombre et chaque nom : {core}") if fr else
                  (f"Say this in your own words, one short sentence, first person, keeping every number "
                   f"and every name: {core}"))
        try:
            raw = self.brain.emitter.emit(prompt, context="talk", max_new_tokens=48,   # words, not programs: the talk adapter
                                          system=identity_system(self.brain.facts, self.brain.chat.state),
                                          temperature=0.85, seed=self.env.steps * 7919 + self.env.level)   # words: sample, never repeat verbatim
        except Exception:
            return line, False
        from emitter import clean_reply
        text = " ".join(clean_reply(raw).split())
        return (mood + text, True) if rephrase_ok(core, text, self.brain.facts) else (line, False)


    def _speak(self, rec: dict) -> tuple[str, bool]:
        """Hand the model the percept record and let it say something. What
        comes back is checked against the record by `grounded_ok`; a sentence
        that fails is refused and the host states the record flatly instead.
        -> (text, spoke)."""
        flat = self.render_percepts(rec)                 # what the guard checks against
        spoken = self.say_flatly(rec)                    # what the host says if the guard refuses
        if self.brain is None or not self.verbalize:
            return spoken, False
        from identity import identity_system
        # what he may speak about: the record, plus the facts he learned this
        # step. Passed verbatim so the guard can check whole words ("ghost")
        # as well as stems.
        learned = " ".join(self._learned_here)
        # THE DIALS REACH THE DECODE, not only the prompt. A prompt asking for
        # a clipped sentence and a sampler set to wander produce a sentence
        # that describes being clipped; a shorter budget and a tighter
        # temperature produce a clipped sentence. Only one of those is the
        # state actually showing up in the words.
        temp, budget = 0.85, 48
        if self.chem is not None:
            m = self.chem.modulation()
            temp = round(min(1.05, max(0.45, 0.55 + 0.45 * m["creativity"]
                                       - 0.30 * m["caution"] + 0.15 * m["energy"])), 2)
            budget = int(max(18, min(56, 48 - 28 * m["urgency"] + 10 * m["warmth"])))
        try:
            raw = self.brain.emitter.emit(
                self._speak_prompt(rec), context="talk", max_new_tokens=budget,
                system=identity_system(self.brain.facts, self.brain.chat.state),
                temperature=temp, seed=self.env.steps * 7919 + self.env.level)
        except Exception as e:
            # traced, not swallowed: a broken emitter used to look exactly
            # like a refused sentence, which cost an afternoon
            self._t("thought_error", error=f"{type(e).__name__}: {e}"[:160])
            return spoken, False
        from emitter import clean_reply
        text = " ".join(clean_reply(raw).split())
        kept = grounded_ok(flat, text, self.brain.facts, learned)
        # THE EXPERIENCE, GrillCheese-style (backend/brain_state/experiences.json:
        # text, emotional_state, the modulation it was said under, and whether
        # the stance worked). Here "worked" is not a rating anybody guessed —
        # it is whether the guard kept the sentence, which is already measured
        # every turn. Kept as a ring so a long run cannot fill memory; nothing
        # reads it yet, and that is the honest state of it: the data a later
        # pass needs to learn WHICH manner survives in WHICH state, collected
        # now because it cannot be collected retroactively.
        if self.chem is not None:
            self.experiences.append({
                "step": self.env.steps, "level": self.env.level, "about": rec.get("about"),
                "text": text[:200], "kept": bool(kept),
                "state": {k: round(v, 3) for k, v in self.chem.to_dict().items()
                          if isinstance(v, (int, float))},
                "modulation": {k: round(v, 3) for k, v in self.chem.modulation().items()},
                "manner": self.chem.manner(), "temperature": temp, "budget": budget,
            })
            del self.experiences[:-self.MAX_EXPERIENCES]
        if kept:
            return text, True
        self._t("thought_refused", said=text[:160], record=flat[:200])
        return spoken, False


    def _think(self, kind: str, **d) -> None:
        """Record a thought for this step if it outranks the current one; the
        line is also a `thought` event, so the console shows it.

        Since 2026-09-15 the MODEL writes the sentence, from the step's
        percept record — the host no longer writes a line for it to rephrase.
        Below the `probe` priority he does not speak at all: the record is
        stated flatly, which keeps the cheap steps cheap (one emit per step is
        already the slow part of the loop)."""
        prio = self._PRIO.get(kind, 0)
        if prio < self._thought_prio:
            return
        if not self.SPEAK_FROM_PERCEPTS:                 # the old path, kept for A/B
            line = self.think(kind, self.lang, self._mood(), **d)
            if not line:
                return
            text, spoke = self._verbalize(line) if prio >= 2 else (line, False)
            self._thought, self._thought_prio = text, prio
            if prio >= 4:
                self._say_aloud = text
            self._t("thought", text=text, about=kind, verbalized=spoke, **({"raw": line} if spoke else {}))
            return
        rec = self.percepts(kind, **d)
        flat = self.render_percepts(rec)                 # the guard referent, kept on the trace
        text, spoke = (self._speak(rec) if prio >= self.SPEAK_ABOVE
                       else (self.say_flatly(rec), False))
        # THE MOOD TAG ONLY GOES ON THE FLAT FALLBACK. When the model wrote the
        # sentence it already chose what this feels like, from the body and the
        # names on offer — stamping the host's own label on the front is the
        # second voice saying the same thing, and worse, saying something else:
        # *"(furious) ... chasing pellets and it's so fun"* (owner, live).
        # A record with no interpretation in it still needs the colour, so the
        # fallback keeps it.
        if not spoke:
            text = self._mood() + text
        self._thought, self._thought_prio = text, prio
        if prio >= 4:
            self._say_aloud = text
        # `raw` carries the record on EVERY thought, spoken or not: it is what
        # the sentence is answerable to, so an audit can check a kept sentence
        # against it without re-deriving anything
        self._t("thought", text=text, about=kind, verbalized=spoke, raw=flat)
