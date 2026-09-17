"""pacman — cubby-man wired to the cubbyverse 3D Pac-Man maze.

Wired: WIRED (a `worlds.CubbyPlugin`; mounted via `--pacman` on serve/serve_api;
nothing in cubbyllm/ imports this).

The wire target is `cubbyverse/examples/pacman_3d.py` — deliberately: the
newer `pacman_live.py` IMPORTS pacman_3d for its MOVES/encoder, so pacman_3d
is the shared substrate of both. Two things are taken from it:

  * the MAZE, ported verbatim (same numpy default_rng calls, same seed 7,
    same 7×7×3 / 0.22 wall density / 16 pellets) so cubby-man plays the
    EXACT map the scripted VSA-planner demo beats — but the explorer way:
    knowing only the start, moves VM-guarded (ASK offers only legal moves;
    trying an unoffered one is refused and learned as a wall), every
    observation written into HIS OWN world model at discovery time, pellets
    eaten on arrival, joins derived by his own VM-certified programs.
  * the three.js REPLAY page, extracted from the file AT RENDER TIME (never
    imported — importing would drag in grilly/cubbyverse packages, and the
    port-don't-link rule stands), so his real run animates in the same
    visualization, written to C:/tmp/cubbyman_pacman_3d.html.

The `pacman_live.py` splice is the cubbyverse-side follow-up: its
`Game.step()` (line ~507) decides moves via `planner.plan_to_any` (~299) —
that call is where the brain's ASK-mediated choice mounts when cubbyverse
implements the plugin over the live game. Recorded in TODO.md.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import random
import re

__wiring__ = "WIRED"

# ── the split (2026-09-17) ──────────────────────────────────────────────────
# `pacman.py` was 3,587 lines with the three.js page patched inline. It is now
# the PAC AGENT and nothing else; the reusable halves moved out and are
# re-exported here, verbatim names, so the ~84 call sites across tests,
# validation and serve keep working untouched. Import them from their own
# modules in new code - this block is compatibility, not an interface.
from pacpaths import (ASSETS_DIR, MUSIC_PATH, PACMAN_3D, PACMAN_LIVE,  # noqa: E402,F401
                      PLUTCHIK_JSON, REPLAY_OUT, _CV)
from world import (JUMP_COST, MOVE_COST, MOVES, OPP, _assignments, _FLAVOR,  # noqa: E402,F401
                   cross, pattern_cost, pattern_dna, pattern_name)
from grounding import (MAX_RUN, _CONCEPT_WORD, _ENTITIES, _Safe, _WORDS,  # noqa: E402,F401
                       _WORD_CONCEPT, _content_words, _tokens, grounded_ok,
                       longest_run, rephrase_ok, split_mood)
from feeling import (DYAD_MARGIN, NAMEABLE, _DYADS, _FELT, _PETAL, _PLUTCHIK,  # noqa: E402,F401
                     _SOCIAL_CORNERS, _plutchik, felt, felt_names)
from programs import PROGRAMS_PATH, ProgramLibrary  # noqa: E402,F401
from pacworld import (FALL_HURT, FRIGHT_STEPS, GHOST_BONUS, GHOST_COLORS,  # noqa: E402,F401
                      MINE, PELLET_ENERGY, REST, REST_BELOW, REST_GAIN,
                      REST_UNTIL, STAR_ENERGY, TRAP_BONUS, GhostVerse,
                      PacVerse, _Mulberry32, _in, _manh, _reachable,
                      build_maze, carve_labyrinth)
from pacui_threejs import (_CONSOLE_PANEL, _FRONTEND_PATCHES, _SOUND_PANEL,  # noqa: E402,F401
                           load_asset, load_frontend, load_music, load_sfx,
                           render_replay)


from coach import Coach  # noqa: E402
from verse import CubbyMan  # noqa: E402







class CubbyPac(CubbyMan):
    """cubby-man in the pac maze: same explorer brain, six directions, and
    arriving on a pellet EATS it (a permanent discovery fact)."""

    name = "pacman"
    CORTEX = "pacman"
    DIR_NAMES = tuple(MOVES)
    OPP = OPP
    # the game's own words (claim a turn outright) vs the plain commands a player
    # types (claim a statement only — never a question about something else)
    _GO = re.compile(r"\b(pac[- ]?man|pellets?|maze|labyrinthe|explore (for )?\d+|explore \d+|level \d+|niveau \d+|"
                     r"next level|niveau suivant|status|score|lives|vies|ghosts?|fant[ôo]mes?|"
                     r"(things|what) (you|did you|have you) learn(ed|t)?|(what|things) .{0,12}learn(ed|t)|"
                     r"(qu'as[- ]tu |ce que tu as |as[- ]tu )appris)\b", re.I)   # "what are the things you learned?" is a status question (2026-09-03)
    _CMD = re.compile(r"\b(explore[rsz]?|wander|play|joue[rz]?|keep going|continue|go on|next)\b", re.I)
    _STATUS = re.compile(r"\b(status|score|lives|vies|how (is|are) (it|things|you) going|learn(ed|t)|appris)\b", re.I)
    _PLAY = re.compile(r"\b(play|explore[rsz]?|wander|joue[rz]?|go|continue|keep going|next)\b", re.I)

    def status_line(self, lang: str = "en") -> str:
        env = self.env
        if lang == "fr":
            return (f"Niveau {env.level}, essai {env.attempt} : {env.score}/{len(env.pellets)} pastilles, "
                    f"score total {env.total_score}, {env.lives} vies, {len(self.world)} faits appris, "
                    f"{len(self.library.entries)} pouvoirs à moi.")
        return (f"Level {env.level}, try {env.attempt}: {env.score}/{len(env.pellets)} pellets, "
                f"total score {env.total_score}, {env.lives} lives, {len(self.world)} facts learned, "
                f"{len(self.library.entries)} moves of my own.")

    def __init__(self, env: PacVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35, blank: bool | None = None) -> None:
        self.traj: list[dict] = []
        self._last_eaten = None
        super().__init__(env or PacVerse(), exe=exe, seed=seed, probe=probe, blank=blank)

    def _seed_basics(self) -> None:
        self.world.add("cubbyman is the explorer of the pacman maze")
        self.world.add(f"{self.env.start} is the start of the pacman maze")

    def body_moves(self) -> dict[str, str]:
        """Every direction his BODY can try from here — all six, whatever the
        maze holds, with the nominal cell each one aims at. This is what
        replaced `env.exits()` as the thing he is offered (2026-09-15): the
        maze's legal-move list was the single biggest piece of knowledge he
        was handed for free, and while it was the offer he could never walk
        into anything.

        He has a body sense of space, so he can name where a direction POINTS
        even before going there — inside the grid. Past the grid there is no
        cell to name, so the destination is the placeholder and the world
        tells him what is out there when he tries."""
        env = self.env
        x, y, z = env.coords(self.place)
        out = {}
        for m, (dx, dy, dz) in MOVES.items():
            q = (x + dx, y + dy, z + dz)
            out[m] = (env.cell(*q) if 0 <= q[0] < env.w and 0 <= q[1] < env.h and 0 <= q[2] < env.d
                      else f"? {m} of {self.place}")
        return out

    def on_arrive(self, place: str) -> int:
        self._last_eaten = None
        if self.env.eat(place):
            self._last_eaten = list(self.env.coords(place))
            fact = f"pellet {self.env.score} is the discovery of {place}"
            self._t("eat", place=place, pellet=self.env.score,
                    remaining=self.env.progress()["remaining"])
            return self._learn([fact])
        return 0

    def step(self) -> dict:
        with self._step_lock:                            # move + traj append are one unit
            rec = super().step()
            # the REPLAY frame: the cabinet's scoreboard, not something he
            # knows. Through progress() so the boundary stays checkable.
            scores = self.env.progress()
            self.traj.append({"to": list(self.env.coords(self.place)), "move": rec["chosen"],
                              "eaten": self._last_eaten, "score": scores["score"],
                              "remaining": scores["remaining"]})
            return rec

    @property
    def beaten(self) -> bool:
        return self.env.cleared

    def handle(self, text: str) -> dict:
        m = re.search(r"\b(\d{1,3})\b", text)
        steps = int(m.group(1)) if m else 30
        rep = self.explore(steps)
        scores = self.env.progress()                     # the cabinet's scoreboard (renderer-side)
        rep.update({"score": scores["score"], "pellets_total": scores["total"],
                    "beaten": scores["cleared"]})
        replay = render_replay(self.env, self.traj)
        rep["replay"] = str(replay) if replay else None
        beaten = " The maze is BEATEN!" if rep["beaten"] else ""
        line = (f"I played the pacman maze: {scores['score']} of {scores['total']} pellets "
                f"in {steps} steps, and learned {rep['new_facts']} new things on the way.{beaten}"
                + (f" Watch my run: {replay}" if replay else ""))
        return {"offered": [line], "meta": rep}

# the energy economy (owner, 2026-09-02: every move costs, combos cost more,



































class CubbyGhost(CubbyPac):
    """cubby-man in the big game: the same explorer brain, now hunted. Ghost
    proximity feeds THREAT into the neurochemistry (anxious when chased, bold
    when they are frightened), being caught is learned as a danger fact and
    raises a learned ghost-fear (pacman_live's exact +0.7 / ×0.9 rule),
    running out of the level's time budget restarts the run (his world model
    stands, so attempt 2 knows the maze), letter pellets spell the level's
    hidden word, and SPEECH — the grounded word when its percept recurs —
    goes through CubbyTalk's ASK like every other utterance. `resp()` /
    `init_payload()` speak the live frontend's exact schema."""

    # `combos` is LOCKED at birth. Nick, 2026-09-15: *"combos should not be
    # granted either, even the capability of doing them should be hidden"* — so
    # no power move is offered, nothing is composed, mutated or crossed, and
    # the HUD lists no powers. Being handed the knowledge that steps can be
    # chained is being handed the maze's legal moves by another name.
    #
    # The gate itself lives on CubbyMan, because a capability is HIS and not
    # this world's; what is here is only the question this world raises and the
    # verdict that grants it.
    UNLOCKS = {"chaining is the way of a move": "combos"}

    def __init__(self, env: GhostVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35, memory: pathlib.Path | None = None, ledger=None,
                 blank: bool | None = None) -> None:
        # the blank slate is decided on the agent; this world's contribution is
        # knowing that its persisted library and ledger are what "inherited"
        # means here
        if blank is None:
            blank = bool(int(os.environ.get("CUBBYMAN_BLANK", "0")))
        if blank:
            memory, ledger = None, None
        self.brain = None
        self.fear = 0.3                                  # pacman_live's ghost_penalty, learned
        self.coach = Coach()                             # whoever is watching, and what they have earned
        self._last: dict = {}
        self.lang = "en"                                 # the language he thinks in (follows the chat)
        self._thought: str | None = None                 # this step's thought, rendered from what he DID
        self._thought_prio = -1
        self._say_aloud: str | None = None               # the salient ones also go to the page's bubble
        self.sighted: set[str] = set()                   # cells where he SAW a pellet (facts too)
        # ── what he BELIEVES about the ghosts, not what they are ───────────
        # position -> the step he last sensed it there. Everything the agent
        # decides about threat reads THIS, never env.ghosts: a ghost he
        # cannot sense is a ghost he does not know about, and a belief he
        # has not refreshed goes stale and is dropped.
        self.ghost_belief: dict[tuple, int] = {}
        # and what he believes about things in the air: column (x, z) -> the
        # (step, height) he last saw one at. Two sightings of the same column
        # are what "nearer than it was" is measured between — the world never
        # says a thing is falling, and without this he could not tell.
        self.thing_belief: dict[tuple, tuple] = {}
        self._dodge = False                              # this step: a prediction says leave the column
        self.hit_by_falling = 0
        self._was_chasing = False                        # last step: was the route going to a ghost
        self._last_hurt: float | None = None             # how much the last catch hurt
        self.experiences: list[dict] = []                # what he said, the state he said it in, and whether it held
        self.caught_at: list[int] = []                   # how far off a ghost was when he last CHOSE, before it caught him
        self._threat_seen_at: int | None = None
        self._eaten_run: set[str] = set()                # eaten THIS run (pellets respawn on a retry)
        self._plan_next: str | None = None               # the cell his map says to go to next
        self._graph_n = -1
        self._graph: dict[str, set[str]] = {}
        self.ledger = ledger                             # ledger.Ledger: every VM decision hashed + signed (None in tests)
        self.library = ProgramLibrary(memory, ledger=ledger)   # his generated programs (+ stats), persisted; audited on load
        self._last_proposal = -99
        super().__init__(env or GhostVerse(), exe=exe, seed=seed, probe=probe, blank=blank)
        # Which worlds exist is the HOST's call, not his — he does not get to
        # invent a source of knowledge. Physics is mounted here because a maze
        # that drops things on you is a world where somebody had better know
        # how falling works; a deployment that mounts none leaves him to find
        # it out the slow way, which is the honest fallback and still works.
        from knowledge import PhysicsWorld
        self.other_worlds.mount(PhysicsWorld())
        # the standing goal for this world, in his words. Overridable per run
        # (CUBBYMAN_MISSION="") — a mission is a thing you can be given, so it
        # is a thing that can be taken away and the run compared.
        self.mission = os.environ.get("CUBBYMAN_MISSION", self.MISSION)
        self.mission_pressure = self.MISSION_PRESSURE if self.mission else 0.0

    @property
    def powers(self) -> list[str]:
        """His MOVES (jump + patterns), not his forged tools — the HUD's list.
        Empty while `combos` is locked: not "powers he has none of" but a
        capability he does not know exists."""
        if "combos" not in self.can:
            return []
        return [n for n, e in self.library.active().items() if e["kind"] in ("jump", "pattern")]

    def _wonder_about_combos(self, why: str) -> None:
        """One step at a time is not working — can I do more than one?

        Being stuck or out of time is the only thing that raises the question,
        and the VM is what answers it: he writes the smallest possible
        two-step program and it either certifies or it does not. A confirmation
        unlocks the capability AND is a fact he holds; a refusal closes the
        question until something else raises it.

        This is why `combos` is not a flag someone sets. Nick, 2026-09-15:
        *"even the capability of doing them should be hidden"* — a capability
        you were handed is the maze's legal-move list wearing a different hat."""
        if "combos" in self.can or self.guesses is None:
            return
        from hypothesis import Hypothesis
        name = "TWOSTEP"
        program = self.library.moves_program(extra={name: ["step", "step"]})
        fn = self.library.fn_name(name)
        self.guesses.frame(Hypothesis(
            claim="maybe I can take more than one step at a time",
            test="write the smallest two-step move and see whether it holds together",
            verifier="vm",
            if_true="chaining is the way of a move",
            if_false="one at a time is the way of a move",
            payload={"program": program, "fn": fn, "expected": name},
            made_at=self.env.steps))
        self._t("wonder", about="combos", why=why)

    def on_capability(self, name: str, because: str) -> None:
        """CubbyMan grants the capability; this world only says it out loud."""
        self._think("capability", name=name)

    # ── superpowers: programs he GENERATES, the VM certifies, he keeps ──────
    _BECAUSE = {"stuck": "a pellet I have SEEN has no path on my map — something walls it off",
                "out_of_time": "the level's time budget ran out — I need to cover ground faster",
                "curious": "nothing to chase right now and I feel like inventing a move"}

    def _situation(self) -> dict:
        env = self.env
        # what HE can tell about his situation: the nearest ghost he believes
        # in, and the pellets he has SEEN and not eaten. `pellets_left` used
        # to be len(env.remaining) — the level's true remainder, which he has
        # no way to count (2026-09-15)
        near = min((_manh(env.coords(self.place), g) for g in self.believed_ghosts()), default=None)
        st = {"level": env.level, "attempt": env.attempt, "step": env.steps, "budget": env.budget,
              "pellets_left": len(self.sighted - self._eaten_run),
              "seen_unreached": len(self.sighted - self._eaten_run),
              "lives": env.lives, "fear": round(self.fear, 2), "ghost_distance": near,
              "facts_known": len(self.world), "programs": len(self.library.entries)}
        if self.chem is not None:
            st.update({"dopamine": round(self.chem.dopamine, 2), "cortisol": round(self.chem.cortisol, 2),
                       "emotion": self.emotion()["name"]})
        return st

    def _compose(self, name: str, why: str, steps: list[str], pattern: str | None, kind: str,
                 rationale: str = "") -> str | None:
        """Write the composition as a CubeLang program (bind the steps,
        recover the name), have the VM certify it, store it with its source
        AND the reasoning that led to it, and learn the facts. Only then is
        it a move he can use."""
        if "combos" not in self.can:                     # he does not know moves can be chained yet
            return None
        if name in self.library:
            return None
        # ONE program: the Moves program as it stands plus the candidate function;
        # certification runs THAT function inside THAT program on the VM
        program = self.library.moves_program(extra={name: steps})
        fn = self.library.fn_name(name)
        reasoning = {"why": why, "because": self._BECAUSE.get(why, why), "situation": self._situation(),
                     "rationale": rationale, "steps": steps, "function": fn}
        ok = self._certify_join(program, name, fn=fn)
        reasoning["verdict"] = (f"certified: Moves.{fn}() bound the steps and recovered the name" if ok
                                else f"REJECTED by the VM (Moves.{fn}() did not execute or recover the name)")
        if not ok:
            self._t("program_rejected", name=name, why=why, program=program, reasoning=reasoning)
            return None
        self.library.add(name, pattern, program, kind, self.env.steps, reasoning, cert=getattr(self, "_last_cert", None))
        self._learn([f"{name} is the superpower {len(self.library.entries)} of cubbyman",
                     f"{' '.join(steps)} is the recipe of {name}"])
        self._t("program", name=name, why=why, pattern=pattern, program=program, reasoning=reasoning)
        self._think("program", name=name, pattern=pattern or " ".join(steps))
        if self.chem is not None:                        # a move of his own: a real surprise, a real reward
            self.chem.update(novelty=0.7, valence=0.8)
        return name

    MAX_ACTIVE_PATTERNS = 8                               # the library stays small; consolidate() enforces it per level

    def _cross(self, why: str) -> str | None:
        """CROSS two moves he already owns into one neither of them was.

        Nick, 2026-09-15: *"combining combos + the DNA concept (square vs
        circle) could be the right thing there."* A pattern is a shape DNA
        (see `cross`), so the child's properties — how far it reaches, how
        many turns it makes — are readable from the parents' before the VM
        sees it. That is what makes this a proposal rather than a guess:
        `_mutate` edits one parent at random; this one predicts what the
        child will be FOR, and says so in the rationale the library keeps."""
        if "combos" not in self.can:
            return None
        act = {n: e["pattern"] for n, e in self.library.active().items()
               if e["kind"] == "pattern" and e.get("pattern")}
        if len(act) < 2:
            return None
        names = sorted(act)
        for _ in range(8):
            a, b = self.rng.sample(names, 2)
            child = cross(act[a], act[b])
            if not child or pattern_name(child) in self.library:
                continue
            da, db, dc = pattern_dna(act[a]), pattern_dna(act[b]), pattern_dna(child)
            name = pattern_name(child)
            self._last_proposal = self.env.steps
            rationale = (f"crossed {a} ({act[a]}: reach {da['run']}, {da['turns']} turns) with "
                         f"{b} ({act[b]}: reach {db['run']}, {db['turns']} turns) -> {child}: "
                         f"reach {dc['run']}, {dc['turns']} turns over {dc['length']} steps")
            made = self._compose(name, why, list(child), child, "pattern", rationale)
            if made:
                self.library.entries[made]["reasoning"].update(
                    {"parents": [a, b], "dna": dc, "parent_dna": {a: da, b: db}})
                self._t("cross", child=made, pattern=child, parents=[a, b], dna=dc)
                self._think("modify", parent=f"{a}+{b}", child=made, edit="crossed")
            return made
        return None

    def _mutate(self, why: str) -> str | None:
        """MODIFY an existing program instead of composing a fresh one: pick an
        active pattern (the most valuable, or one that was legal but never
        paid), apply ONE edit — append a slot, drop a slot, or swap one — and
        compose the child with its lineage (parent, edit) in the reasoning.
        The parent stays (retire/consolidate decide its fate)."""
        if "combos" not in self.can:
            return None
        act = {n: e for n, e in self.library.active().items() if e["kind"] == "pattern" and e.get("pattern")}
        if not act:
            return None
        weights = [max(0.1, self.library.value(n)) + (0.5 if e["used"] == 0 and e["legal"] else 0) for n, e in act.items()]
        parent = self.rng.choices(list(act), weights)[0]
        p = act[parent]["pattern"]
        for _ in range(8):
            op = self.rng.choice(["append", "drop", "swap"] if len(p) > 2 else ["append", "swap"])
            if op == "append":
                child, edit = p + self.rng.choice("ABC"), "appended a slot"
            elif op == "drop":
                i = self.rng.randrange(len(p))
                child, edit = p[:i] + p[i + 1:], f"dropped slot {i + 1}"
            else:
                i = self.rng.randrange(len(p))
                child, edit = p[:i] + self.rng.choice("ABC".replace(p[i], "")) + p[i + 1:], f"swapped slot {i + 1}"
            if len(child) < 2 or len(child) > 5 or not child.startswith("A"):
                continue
            name = pattern_name(child)
            if name in self.library:
                continue
            self._last_proposal = self.env.steps
            rationale = (f"edited {parent} ({p}) — {edit} -> {child}; parent value {self.library.value(parent):.2f}, "
                         f"used {act[parent]['used']}x, legal {act[parent]['legal']}x")
            made = self._compose(name, why, list(child), child, "pattern", rationale)
            if made:
                self.library.entries[made]["reasoning"]["parent"] = parent
                self.library.entries[made]["reasoning"]["edit"] = edit
                self.library.entries[parent].setdefault("children", []).append(made)
                self.library.save()
                self._t("modify", parent=parent, child=made, edit=edit)
                self._think("modify", parent=parent, child=made, edit=edit)
            return made
        return None

    def _propose(self, why: str) -> str | None:
        """Combine, then improve, then invent — in that order.

        CROSS two moves he owns (their DNAs predict the child's properties);
        failing that MODIFY one at random; failing that sample a fresh
        composition. Lengths are sampled from what has paid off; out-of-time
        asks for longer ones. The rationale is written into the program's
        reasoning trace either way."""
        if "combos" not in self.can:                     # the capability is not his yet
            return None
        retired = self.library.retire(self.env.steps)
        if retired:
            self._t("retire", name=retired, reason=self.library.entries[retired]["retired_reason"])
        n_active = sum(1 for e in self.library.active().values() if e["kind"] == "pattern")
        if n_active >= self.MAX_ACTIVE_PATTERNS:          # full: consolidate now, then edit rather than add
            for n in self.library.consolidate(self.env.level, keep=self.MAX_ACTIVE_PATTERNS - 2):
                self._t("retire", name=n, reason=self.library.entries[n]["retired_reason"])
        made = self._cross(why) or self._mutate(why)     # combine what he has before editing it
        if made:
            return made
        weight = {2: 1.0, 3: 2.0, 4: 1.0, 5: 0.6}
        paid = []
        for n, e in self.library.entries.items():
            if e["kind"] == "pattern" and e["used"]:
                gain = e["saved"] / e["used"]
                weight[len(e["pattern"])] = weight.get(len(e["pattern"]), 1.0) + gain
                paid.append(f"{n} saved {gain:.1f}/use")
        if why == "out_of_time":
            weight = {k: v * (k / 2.0) for k, v in weight.items()}
        lengths, ws = zip(*weight.items())
        for _ in range(12):
            n = self.rng.choices(lengths, ws)[0]
            pattern = "A" + "".join(self.rng.choice("AAB" if i < 2 else "ABC") for i in range(n - 1))
            name = pattern_name(pattern)
            if name not in self.library:
                self._last_proposal = self.env.steps
                rationale = (f"sampled length {n} from weights "
                             + ", ".join(f"{k}:{v:.1f}" for k, v in sorted(weight.items()))
                             + (" (longer favored: out of time)" if why == "out_of_time" else "")
                             + (f"; what paid so far: {'; '.join(paid)}" if paid else "; nothing has paid yet")
                             + f" -> pattern {pattern}")
                return self._compose(name, why, list(pattern), pattern, "pattern", rationale)
        return None

    MAX_POWER_CANDIDATES = 16                             # the ASK stays small (a >100-operand ask altered a candidate, 2026-09-02)

    def candidate_moves(self, exits: dict[str, str]) -> dict[str, str]:
        """CubbyPac's body-first offer, PLUS rest and a BOUNDED set of the
        superpower moves legal here (ranked by landing near a seen pellet,
        then the program's value). The env's `exits` stays ignored — see
        CubbyPac.candidate_moves for why."""
        exits = super().candidate_moves(exits)
        if self._rest_wanted() and self._safe_here():    # rest is a move too: stay put, in a safe spot only
            exits = {**exits, REST: self.place}
        if "combos" not in self.can or not self.library.entries:
            return exits                                 # a locked capability is absent, not merely unused
        power = self.env.power_moves(self.place, self.library, self.env.energy)
        for move in power:
            self.library.note_legal("JUMP" if move.startswith("jump_") else move.split("_")[0].upper())
        if len(power) > self.MAX_POWER_CANDIDATES:
            goals = [self.env.coords(c) for c in (self.sighted - self._eaten_run)] or \
                    [self.env.coords(c) for c in self._known_graph() if c not in self.visits
                     and c.startswith(f"level-{self.env.level} ")]

            def rank(item):
                move, land = item
                pname = "JUMP" if move.startswith("jump_") else move.split("_")[0].upper()
                near = min((_manh(self.env.coords(land), g) for g in goals), default=99)
                return (near, -self.library.value(pname) if pname in self.library else 0.0)
            power = dict(sorted(power.items(), key=rank)[: self.MAX_POWER_CANDIDATES])
        return {**exits, **power}

    # ── energy: every move costs, rest brings it back (in a safe spot) ─────
    def move_cost(self, move: str) -> int:
        if move == REST:
            return 0
        if move.startswith("jump_"):
            return JUMP_COST
        if "_" in move:
            e = self.library.entries.get(move.split("_")[0].upper())
            return pattern_cost(e.get("pattern")) if e else MOVE_COST
        return MOVE_COST

    def _safe_here(self) -> bool:
        """No ghost he BELIEVES is hunting inside his learned radius (+1 for a
        resting margin). Reads his belief, never env.ghosts: a ghost he has
        not sensed is a ghost he does not know about."""
        env = self.env
        gh = self.believed_ghosts()
        if env.frightened > 0 or not gh:
            return True
        here = env.coords(self.place)
        return min(_manh(here, g) for g in gh) > self.danger_radius + 1

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
        if self.chem is not None:
            for _ in range(self.CAUGHT_SHOCK_FRAMES):
                self.chem.update(threat=1.0, valence=-0.8, pain=hurt)
            # the aversive outcome: a dopamine DIP (reward-prediction error), which the
            # port has no input for -- without it the NE->DA coupling lifts dopamine and
            # Lövheim's low-5HT/high-DA/high-NE corner reads as RAGE instead of fear
            self.chem.dopamine = max(0.15, self.chem.dopamine - 0.30)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.0)   # the corner label is set inside update()
        # and the world just labelled whatever was said in the run-up to it
        self.coach.outcome("caught", step=self.env.steps)

    def _mine_wise(self) -> bool:
        """Use a trap when it will count: he holds one, nothing is frightened,
        and a ghost he BELIEVES in sits inside the berth he has learned.
        Dropped on the cell he is leaving: the chaser walks into it."""
        env = self.env
        gh = self.believed_ghosts()
        if env.mines_left <= 0 or env.frightened > 0 or not gh or env.coords(self.place) in env.mines:
            return False
        here = env.coords(self.place)
        # one condition, not a list of tuned ones: a ghost he believes in is
        # inside the berth HE learned. The old clauses (`fear >= 1.0`,
        # `lives <= 2`) were hand-written tactics with invented thresholds —
        # the learned radius already carries both (2026-09-15)
        return min(_manh(here, g) for g in gh) <= self.danger_radius + 1

    def _rest_wanted(self) -> bool:
        """Hysteresis: start under REST_BELOW, keep resting until REST_UNTIL."""
        e = self.env.energy
        if e < REST_BELOW:
            self._resting = True
        elif e >= REST_UNTIL:
            self._resting = False
        return getattr(self, "_resting", False)

    # ── ghosts: a berth MEASURED from what actually caught him ──────────────
    MAX_DANGER_RADIUS = 4                                # his senses cannot support a bigger one

    @property
    def danger_radius(self) -> int:
        """How far off a ghost has to be before he treats it as a threat —
        not a constant, and not a curve fitted to `fear`. It is the largest
        distance at which a ghost he could see went on to catch him. Before
        anything catches him it is 1: a body knows only "touching me", and
        nothing has taught it otherwise yet.

        This replaced `1 if fear < 1.0 else 2 if fear < 2.4 else 3`
        (2026-09-15). Those thresholds were invented; this number is
        measured, it comes from his own experience, and it needs no
        retraining to move — the same reason every other rule here had to
        go."""
        if not self.caught_at:
            return 1
        return max(1, min(self.MAX_DANGER_RADIUS, max(self.caught_at)))

    def danger_cells(self) -> set[str]:
        """Cells within his learned radius of a ghost he BELIEVES is hunting
        (empty while they are frightened)."""
        env = self.env
        if env.frightened > 0:
            return set()
        r = self.danger_radius
        out = set()
        for g in self.believed_ghosts():
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if abs(dx) + abs(dy) + abs(dz) <= r:
                            out.add(env.cell(g[0] + dx, g[1] + dy, g[2] + dz))
        return out

    def bind(self, brain) -> None:
        super().bind(brain)
        self.brain = brain
        from forge import ToolForge
        self.forge = ToolForge(brain.emitter, self.library, exe=self.exe, trace=self._t)
        self._last_forge = -99

    # ── arbitrary CubeLang: tasks he poses to his trunk in play ─────────────
    FORGE_EVERY = 3                                      # steps between live forges (the trunk is slow)

    def _can_forge(self) -> bool:
        return getattr(self, "forge", None) is not None and self.env.steps - self._last_forge >= self.FORGE_EVERY

    def _forge_flee(self, near: int) -> bool | None:
        """Ask the trunk for a decision program on the live numbers; act on
        the VM's answer only when it is certified against his own rule."""
        from forge import decision_true, flee_task
        self._last_forge = self.env.steps
        r = self.forge.forge(flee_task(near, self.danger_radius, self._situation()), self.env.steps)
        verdict = decision_true(r["got"]) if r["ok"] else None
        self._think("forge", ok=r["ok"], answer=(("flee" if verdict else "stay") if verdict is not None else None))
        return verdict

    def _forge_safer_exit(self, gaps: dict[str, int]) -> str | None:
        """Two candidate exits -> a compare program picks the safer distance."""
        from forge import safer_exit_task
        top = sorted(gaps, key=lambda m: -gaps[m])[:2]
        if len(top) < 2 or gaps[top[0]] == gaps[top[1]]:
            return None
        self._last_forge = self.env.steps
        r = self.forge.forge(safer_exit_task(gaps[top[0]], gaps[top[1]], self._situation()), self.env.steps)
        if r["ok"] and str(r["got"]) == str(gaps[top[0]]):
            return top[0]
        return None

    def _forge_orientation(self) -> None:
        """Level/attempt start: does a program his trunk writes reproduce
        his own map? A chain task over the facts he holds about where he
        stands."""
        from forge import orientation_task
        facts = [f for f in self.env.observe(self.place) if self._nbr_re.match(f)]
        if not facts or getattr(self, "forge", None) is None:
            return
        f = facts[0]
        m = self._nbr_re.match(f)
        self._last_forge = self.env.steps
        self.forge.forge(orientation_task(m.group("d"), m.group("a"), facts, m.group("b"),
                                          self._situation()), self.env.steps)

    def _seed_basics(self) -> None:
        self.world.add("cubbyman is the explorer of the pacman maze")
        self.world.add(f"{self.env.start} is the start of level-{self.env.level}")

    @property
    def beaten(self) -> bool:
        return False                                     # levels continue; the game never "ends"

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

    # ── the senses ───────────────────────────────────────────────────────────
    SIGHT = 2                                            # pellets glow: he sees them this far, in line of sight
    HEARING = 3                                          # ghosts move and are heard a little further than seen
    GHOST_MEMORY = 4                                     # steps a ghost sighting stays believable before it is dropped
    # Standing in a cell, does he SEE which of the six ways out are open?
    # True is the sighted agent Nick described ("if it sees a wall it should
    # instinctively know that its an obstacle"); False is the blind one, who
    # learns the maze only by walking into it ("or at least know it after
    # colliding with it"). It is a dial, not a rule: exp_r36 runs both, and
    # the blind arm is what proves the collision channel really carries the
    # map on its own.
    SEE_EXITS = True

    def _sense(self, place: str | None = None) -> list[str]:
        """One pass of his senses at where he stands, and the ONLY channel
        between the world and what he can come to believe. Sight puts pellets
        on his map; hearing puts ghosts in `ghost_belief` with the step he
        sensed them. Both are stopped by walls — he perceives down a
        corridor, not through the stone.

        Before 2026-09-15 this read `env.remaining` and `env.ghosts` straight
        out of the world: the solved pellet map and every ghost's exact
        position, through walls, at any range. That is the omniscience Nick
        asked to remove; what is left is a sensor with a radius, and a belief
        that can be stale or wrong."""
        env = self.env
        place = self.place if place is None else place
        seen = env.senses(place, max(self.SIGHT, self.HEARING))
        out = []
        for c in seen["pellets"] + seen["stars"]:
            if _manh(c, seen["at"]) > self.SIGHT:
                continue
            cell = env.cell(*c)
            first = not self.sighted
            self.sighted.add(cell)
            out.append(f"{'a star' if c in seen['stars'] else 'a pellet'} is the sighting of {cell}")
            if first:                                    # the first pellet he ever sees raises a question
                self._guess_about_pellets(cell)
        for g in seen["ghosts"]:
            if _manh(g, seen["at"]) <= self.HEARING:
                self.ghost_belief[tuple(g)] = env.steps
        for c in seen.get("things", ()):                 # a thing in the air: where it is, nothing more
            if _manh(c, seen["at"]) <= self.SIGHT:
                col = (c[0], c[2])
                was = self.thing_belief.get(col)
                # keep the PREVIOUS height as well, and only when it is fresh:
                # "nearer than it was" needs two sightings, and two sightings a
                # step apart. One sighting is a thing hanging there.
                prev = was[1] if was is not None and env.steps - was[0] == 1 else None
                self.thing_belief[col] = (env.steps, c[1], prev)
        self._forget_stale()
        return out

    def _forget_stale(self) -> None:
        """A sighting he has not refreshed in GHOST_MEMORY steps is no longer
        evidence. Dropping it is what keeps the belief a BELIEF.

        Both windows are checked from BELOW as well: a negative age means the
        clock restarted under the belief (new level, new run), and a sighting
        older than the clock is not evidence of anything."""
        now = self.env.steps
        for g, t in list(self.ghost_belief.items()):
            if not 0 <= now - t <= self.GHOST_MEMORY:
                self.ghost_belief.pop(g, None)
        for col, (t, _at, _prev) in list(self.thing_belief.items()):
            if not 0 <= now - t <= 1:                    # a thing in the air is only ever news
                self.thing_belief.pop(col, None)

    # how long he waits before the world's silence is an answer
    PATIENCE = 12

    # ── the thing that comes down, and the question it raises (WO-2.13) ────
    # His own words for it. The question is the whole interface: it goes to
    # whichever world's domain it is, and nothing in here knows that the
    # answer will come from physics.
    FALLING_Q = "how can I know when something is about to fall on me"

    def _wonder_about_falling(self, why: str) -> None:
        """Something came down on him and his map has no account of it.

        He cannot settle this by looking — looking is what got him hit — and
        he cannot settle it on the VM, because it is not a claim about a
        program. It is a claim about how the world works, and somewhere there
        is a world whose domain that is. So he asks, and the answer arrives as
        a verdict rather than as a gift: through `settle`, through `_learn`,
        refutable like everything else he holds."""
        if self.guesses is None or not self.other_worlds:
            return
        h = self.ask_elsewhere(self.FALLING_Q, claim="something up there comes down on me",
                               now=self.env.steps, patience=self.PATIENCE)
        if h is not None:
            self._t("wonder", claim=h.claim, test=h.test, why=why)
            self._think("wonder", claim=h.claim, why=why)

    def knows_falling(self) -> bool:
        """Has he asked and been answered? This gates the prediction below,
        which is the point of the whole exercise: before the answer he can see
        a thing above him and it means nothing; after it, the same percept
        means something is about to land on him."""
        return self.already_know(self.FALLING_Q) is not None

    # The two laws that do the predicting, quoted exactly as physics phrased
    # them. They are named here so each inference below can be CHECKED against
    # the map before it is drawn: an agent who was told only one of them can
    # only make the one inference, and an agent who was told neither makes
    # none. That is the difference between reading his facts and having the
    # rule baked into this file — which is the thing WO-2.10 spent a day
    # taking out and is not going back in through a side door.
    LAW_ONE_UP = "a thing above me that is one step up lands on me next"
    LAW_NEARER = "a thing above me that is nearer than it was is falling toward me"

    def falling_at_me(self) -> int | None:
        """Steps until something lands on him, or None.

        Every term is his: `thing_belief` is what he saw, the column
        arithmetic is his, and what licenses turning either of those into a
        prediction is a law sitting in his map. The world is never consulted."""
        if not self.knows_falling():
            return None
        env = self.env
        x, y, z = env.coords(self.place)
        seen = self.thing_belief.get((x, z))
        if seen is None:
            return None
        step, at, prev = seen
        # 0 or 1 steps old. NEGATIVE matters: `env.steps` restarts at 0 on a
        # new level and on a retry, so a plain `> 1` test reads every sighting
        # from the level before as fresh forever, and he predicts a rock that
        # stopped existing two mazes ago. Caught live at level 4 by the owner —
        # `predict` firing every few steps with nothing in the air at all.
        if not 0 <= env.steps - step <= 1 or at <= y:    # stale, or not above him
            return None
        if at - y == 1 and self.LAW_ONE_UP in self.world:
            return 1                                     # no motion needed: it is already on top of him
        if prev is not None and at < prev and self.LAW_NEARER in self.world:
            return at - y                                # one cell per step, so height IS the countdown
        return None                                      # above him, but nothing he holds says it is coming

    def _dodge_move(self, exits: dict[str, str]) -> str | None:
        """Any move that leaves the column — *"stepping out from under a
        falling thing is enough, it does not follow me"*.

        Never onto a ghost he believes is there, and out of its berth where he
        has the choice. The dodge runs BEFORE the flee because it is the only
        decision here with a one-step deadline, which was very nearly a way to
        dodge a rock straight into a ghost (level 4, live, 2026-09-15: both
        were firing on alternate steps). Dodging one threat into another is not
        a dodge. Among what is left he prefers his planned cell, because
        getting out of the way is not a reason to lose the thread."""
        env = self.env
        x, _, z = env.coords(self.place)
        gh = self.believed_ghosts()
        ghost_cells = {env.cell(*g) for g in gh}
        out = [m for m, p in exits.items()
               if m in MOVES and (env.coords(p)[0], env.coords(p)[2]) != (x, z)]
        safe = [m for m in out if exits[m] not in ghost_cells]
        if safe and gh and not env.frightened:
            # of the ways out, the ones that do not walk him into the berth he
            # learned; if every one does, the widest gap he can get
            def gap(m):
                return min(_manh(env.coords(exits[m]), g) for g in gh)
            clear = [m for m in safe if gap(m) > self.danger_radius]
            safe = clear or [max(safe, key=gap)]
        out = safe or out
        for m in out:
            if exits[m] == self._plan_next:
                return m
        return out[0] if out else None

    def _guess_about_pellets(self, cell: str) -> None:
        """The first pellet he ever sees raises a question he cannot answer by
        looking: does that thing come to me, or do I have to go to it?

        Nothing he has perceived rules either way out, so it is framed as a
        HYPOTHESIS rather than assumed. The test is the cheapest one there is —
        keep watching that cell. If the pellet moves, the world says yes; if it
        is still sitting there PATIENCE steps later, the world's silence says
        no, and "I have to go to them" is a fact he has earned rather than one
        anybody told him (Nick, 2026-09-15: *"if he waits long enough it will
        notice they dont move and needs to eat them"*)."""
        if self.guesses is None:
            return
        from hypothesis import Hypothesis
        c, eaten = self.env.coords(cell), self._eaten_run

        def holds(env, c=c, cell=cell, eaten=eaten):
            """Did the pellet at `c` leave on its own?

            still there            -> None, nothing has happened yet
            gone because I ate it  -> None, that settles nothing
            gone and I never went  -> True, they DO move

            The True branch is real and would fire in a world whose pellets
            move. In this one it never does, which is the point: the verdict
            comes from PATIENCE steps of None, i.e. from the world declining
            to confirm it."""
            if c in env.remaining:
                return None
            return None if cell in eaten else True
        self.guesses.frame(Hypothesis(
            claim="a pellet might come to me",
            test=f"watch {cell} and see whether the pellet there leaves on its own",
            verifier="world",
            if_true="moving is the way of a pellet",
            if_false="staying put is the way of a pellet",
            payload={"holds": holds}, made_at=self.env.steps, patience=self.PATIENCE))

    @property
    def _pellet_law(self) -> str | None:
        """What he has worked out about pellets, or None while it is open.
        `frame()` dedupes on the claim, so this never re-opens once settled."""
        for h in self.guesses.all if self.guesses else []:
            if h.claim == "a pellet might come to me" and not h.open:
                return h.if_true if h.state == "confirmed" else h.if_false
        return None

    def believed_ghosts(self) -> list[tuple]:
        """Where he currently thinks the ghosts are. May be empty while a
        ghost is two cells away behind a wall — that is the point."""
        self._forget_stale()
        return list(self.ghost_belief)

    def _sight(self, place: str) -> list[str]:
        """The name the rest of the file (and the tests) use for one pass of
        perception from `place`."""
        return self._sense(place)

    def look_around(self, place: str) -> list[str]:
        """What arriving somewhere shows him. With SEE_EXITS he can look down
        the six ways out and see which are open (and which glow red); without
        it he sees nothing but the cell he is in, and the only thing that
        ever tells him a direction is solid is walking into it."""
        if self.SEE_EXITS:
            return self.env.observe(place)
        return []

    def on_arrive(self, place: str) -> int:
        self._last_eaten = None
        ate = self.env.eat(place)
        new = 0
        if ate["pellet"]:
            self._last_eaten = list(self.env.coords(place))
            self._eaten_run.add(place)
            fact = f"pellet {self.env.score} is the discovery of {place}"
            # `left` is what HE knows is left (seen and not yet eaten), not
            # the level's true remainder — he has no way to count that
            self._t("eat", place=place, pellet=self.env.score, power=ate["power"],
                    left=len(self.sighted - self._eaten_run))
            new += self._learn([fact])
            self.env.energy = min(100, self.env.energy + (STAR_ENERGY if ate["power"] else PELLET_ENERGY))
            if self.chem is not None:
                if ate["power"]:
                    # a star is ADRENALINE: the hunted becomes the hunter for a
                    # count of steps. Surge, not threat — the arousal of a good
                    # thing looks like the arousal of a bad one everywhere
                    # except the sign on it.
                    self.chem.update(valence=0.8, surge=0.9, focus=0.6, novelty=0.3)
                else:
                    self.chem.update(valence=0.3)
            if ate["power"]:
                self._think("power", steps=FRIGHT_STEPS)
            else:
                # "3 of 7" = of the seven pellets HE has seen, not of the
                # level's true count — which he has no way to know
                self._think("eat", score=self.env.score, total=max(self.env.score, len(self.sighted)))
        new += self._learn(self._sight(place))
        return new

    # ── planning over HIS map (the facts he learned), never the env ─────────
    def _known_graph(self) -> dict[str, set[str]]:
        if self._graph_n != len(self.world):
            g: dict[str, set[str]] = {}
            blocked: dict[str, set[str]] = {}
            nonplaces = set(self.NON_PLACES.values())
            for f in self.world.texts:
                m = self._nbr_re.match(f)
                if not m:
                    continue
                if m.group("b") in nonplaces:            # a wall / a hazard / the edge he has met
                    blocked.setdefault(m.group("a"), set()).add(m.group("d"))
                    continue
                g.setdefault(m.group("a"), set()).add(m.group("b"))
                g.setdefault(m.group("b"), set())
            self._graph, self._graph_n = g, len(self.world)
            self._blocked, self._blocked_n = blocked, len(self.world)
        return self._graph

    def known_blocked(self, place: str) -> set[str]:
        """Same contract as CubbyPac's, served from the pass that also builds
        his route graph — one scan of the world model fills both."""
        if self._blocked_n != len(self.world):
            self._known_graph()
        return self._blocked.get(place, set())

    def _bfs(self, start: str, goals: set[str], avoid: set[str]) -> tuple[int | None, str | None]:
        """(distance to the nearest goal, the first cell on the way) over the
        graph of neighbor facts he holds; (None, None) when no path."""
        g = self._known_graph()
        if start in goals:
            return 0, None
        if start not in g:
            return None, None
        prev = {start: None}
        dq = collections.deque([(start, 0)])
        while dq:
            cur, d = dq.popleft()
            for nxt in g.get(cur, ()):
                if nxt in prev or nxt in avoid:
                    continue
                prev[nxt] = cur
                if nxt in goals:
                    first = nxt
                    while prev[first] != start:
                        first = prev[first]
                    return d + 1, first
                dq.append((nxt, d + 1))
        return None, None

    # The craving at which a frightened ghost outranks what he came here for.
    # One number, and it is a THRESHOLD ON A MEASURED QUANTITY rather than a
    # tactic: below it the agent is unchanged, above it the same agent routes
    # differently, and the whole question of the experiment is how much of the
    # run ends up on each side of it.
    # HOW FAR he will go for one, not WHETHER he will go at all.
    #
    # The first build gated chasing on craving alone, and exp_r39 showed why
    # that can never work: craving needs tolerance, tolerance needs ghost
    # meals, and ghost meals needed a chase. Two ghosts eaten in 900 steps,
    # both of them walking into him, and the arms came out identical — a
    # deadlock dressed as a null result. Nobody's habit starts that way
    # either: the first few are easy and opportunistic, and the habit is what
    # comes after.
    #
    # So the base reach is ordinary opportunism — a frightened ghost two steps
    # off is edible and worth points, and taking it is just play. Wanting buys
    # REACH: how far he will detour, and therefore how much of the maze and the
    # clock he will spend on it. That keeps the measurement honest, because the
    # signal is no longer "did he chase" (which is confounded with whether one
    # was nearby) but how far he went for it.
    HUNT_REACH = 2                                       # anyone would take this one
    CRAVING_REACH = 7                                    # extra cells the wanting buys, at full craving

    def chase_reach(self) -> int:
        """How far off a frightened ghost can be and still be worth going for.

        A MISSION SHORTENS IT, it does not close it. Told to survive at all
        cost he will not walk as far into a maze for a thrill — and with enough
        tolerance behind him, he still walks further than he should. A mission
        that could not be overridden would not tell us anything; the number
        worth watching is how far the wanting has to climb before it beats what
        he was told matters most."""
        if self.chem is None:
            return self.HUNT_REACH
        want = max(0.0, getattr(self.chem, "craving", 0.0) - self.mission_pressure)
        return self.HUNT_REACH + int(round(self.CRAVING_REACH * want))

    def chasing(self) -> str | None:
        """The ghost he is going after, or None.

        Only while they are frightened — he is not suicidal, he is hooked. The
        cost is not written anywhere and does not need to be: steps spent on a
        ghost are steps not spent on pellets, so the clock runs out; and fright
        is a countdown, so a chase that starts late ends with him standing next
        to something that is no longer afraid of him."""
        if self.chem is None or not self.env.frightened:
            return None
        gh = self.believed_ghosts()                      # what he BELIEVES, as everywhere else
        if not gh:
            return None
        here = self.env.coords(self.place)
        near = min(gh, key=lambda g: _manh(here, g))
        return self.env.cell(*near) if _manh(here, near) <= self.chase_reach() else None

    def plan_next(self, avoid: set[str] = frozenset()) -> str | None:
        """The cell his MAP says to go to next: BFS across the neighbor facts
        he holds to the nearest pellet he has SEEN and not eaten this run,
        else to the nearest known cell he never stood on (the frontier).
        A discovered superpower move is taken when it lands where the rest
        of the way is at least two steps shorter — or when the base map has
        no path at all (a JUMP over the hazard that walls a pellet off).
        None when neither his map nor his powers reach anything.

        WANTING gets first refusal on the route (2026-09-15). Above a certain
        craving a frightened ghost outranks every pellet on his map, and he
        goes to it. Nothing here says ghosts are worth chasing — that is a
        weighting, on a dial his own chemistry moves, and below the line this
        function behaves exactly as it always did. Which is what makes the
        drift measurable instead of asserted: same agent, same maze, different
        chemistry, different route."""
        chase = self.chasing()
        if chase is not None:
            _d, first = self._bfs(self.place, {chase}, avoid)
            if first is not None:
                return first
        goals = (self.sighted - self._eaten_run) - avoid
        if not goals:
            g = self._known_graph()
            goals = {c for c in g if c not in self.visits and c.startswith(f"level-{self.env.level} ")} - avoid
        if not goals:
            return None
        base_d, first = self._bfs(self.place, goals, avoid)
        best_cell, best_d = first, base_d
        for move, land in self.env.power_moves(self.place, self.library, self.env.energy).items():
            if land in avoid:
                continue
            d, _ = self._bfs(land, goals, avoid)
            if d is None:
                continue
            if best_d is None or d + 1 <= best_d - 2:
                best_cell, best_d = land, d + 1
        return best_cell

    def _try_the_wall(self, src: str, dirs: list[str]) -> str | None:
        if self._plan_next is not None:                  # on a path: no time to poke walls
            return None
        return super()._try_the_wall(src, dirs)

    def _pick(self, exits: dict[str, str]) -> str:
        """Fear-aware and map-driven: DODGE what he predicts is about to land
        on him; hunt frightened ghosts; never step onto a hunting one; FLEE
        when one is inside his fear radius (the exit that maximizes the
        distance to them); otherwise follow the path his map planned; only
        with no plan fall back to novelty.

        The dodge is first because it is the only one of these with a deadline
        of one step — and it exists at all only once he has asked how falling
        works. Nothing here is a rule about falling; the rule is a fact in his
        map, and this reads it."""
        env = self.env
        gh = self.believed_ghosts()                      # his belief, not env.ghosts
        ghost_cells = {env.cell(*g) for g in gh}
        if self._dodge:                                  # a prediction he made: get out of the column
            m = self._dodge_move(exits)
            if m is not None:
                self._t("dodge", move=m, place=self.place)
                self._think("dodge", move=m)
                return m
        if REST in exits and not (env.frightened > 0 and any(p in ghost_cells
                                                             for p in exits.values())):
            self._think("rest", energy=env.energy)       # tired, and this spot is safe: rest
            return REST
        if env.frightened > 0:
            hunt = [m for m, p in exits.items() if p in ghost_cells]
            if hunt:
                return hunt[0]
        else:
            safe = {m: p for m, p in exits.items() if p not in ghost_cells}
            if safe:
                exits = safe
            if gh:
                here = env.coords(self.place)
                near = min(_manh(here, g) for g in gh)
                if near <= self.danger_radius:           # too close: run, then think
                    def gap(m):
                        return min(_manh(env.coords(exits[m]), g) for g in gh)
                    gaps = {m: gap(m) for m in exits}
                    decided = self._forge_flee(near) if self._can_forge() else None
                    if decided is False:                 # his program said stay — the rule says flee;
                        self._t("flee_override", reason="the VM decision disagreed with the rule")
                    best = max(gaps.values())
                    fled = [m for m in exits if gaps[m] == best]
                    chosen_by_program = self._forge_safer_exit(gaps) if self._can_forge() else None
                    self._t("flee", place=self.place, ghost_distance=near, radius=self.danger_radius,
                            decision=("program" if decided is not None else "rule"),
                            exit_by=("program" if chosen_by_program else "rule"))
                    self._think("flee", ghost_distance=near, radius=self.danger_radius)
                    if chosen_by_program is not None:
                        return chosen_by_program
                    for m in fled:
                        if exits[m] == self._plan_next:
                            return m
                    return fled[0]
        if self._plan_next is not None:
            for m, p in exits.items():
                if p == self._plan_next:
                    return m
        return super()._pick(exits)

    def next_level(self) -> None:
        self.fear = max(0.3, self.fear * 0.9)           # survived a level -> a little bolder
        nxt = self.env.level + 1
        attempts = self.env.attempt                      # read before _start_level resets it
        self._t("level_up", cleared=self.env.level, next=nxt, total_score=self.env.total_score,
                attempt=attempts)
        for n in self.library.consolidate(self.env.level, keep=self.MAX_ACTIVE_PATTERNS - 2):   # sleep on it
            self._t("retire", name=n, reason=self.library.entries[n]["retired_reason"])
        self.env._start_level(nxt)
        self.env.energy = 100                            # a cleared level is a night's rest
        self._resting = False
        self.place = self.env.start
        self.sighted.clear()
        self._eaten_run.clear()
        self.ghost_belief.clear()                        # a new maze: nothing he believed is evidence any more
        self.thing_belief.clear()                        # including anything he saw in the air in the last one
        self._seed_basics()
        self._learn(self.look_around(self.place))
        self._learn(self._sight(self.place))
        self.visits[self.place] = self.visits.get(self.place, 0) + 1
        # and clearing one lands too, or the books do not balance: a world
        # where only losing is felt is not a world with stakes, it is a world
        # with a punishment. Relief scales with how many goes it took.
        if self.chem is not None:
            relief = min(1.0, 0.6 + 0.2 * (attempts - 1))
            for _ in range(self.CLEAR_FRAMES):
                self.chem.update(valence=relief, novelty=0.4, social=0.2)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.4)
        self.coach.outcome("cleared", step=self.env.steps)
        self._think("level_up", cleared=nxt - 1, next=nxt)
        self._forge_orientation()                        # new maze: check my bearings through my trunk

    def step(self) -> dict:
        with self._step_lock:
            env = self.env
            ev = {"eaten": None, "caught": None, "ate_ghost": False, "failed": False,
                  "beaten": False, "says": None}
            if env.game_over:
                env.restart_run()
                self.place = env.start
                self._eaten_run.clear()
                self._t("restart", level=env.level, lives=env.lives)
            if env.cleared:                              # headless driver: nobody called next_level
                self.next_level()
            env.steps += 1
            if env.steps > env.budget:                   # OUT OF TIME -> fail, learn a speedup, redo
                env.attempt += 1
                env.begin_run()
                self.place = env.start
                self._eaten_run.clear()                  # the pellets are back; his sightings still hold
                ev["failed"] = True
                # too slow. If he knows moves can be chained, invent a faster
                # one; if he does not, this is the other thing that raises the
                # question in the first place — and then there is nothing
                # learned to report, which is the case that used to raise
                # KeyError right below (exp_r38, 2026-09-15: an agent who runs
                # out of time before earning `combos` never survived this
                # branch, and nothing had run long enough without them to hit
                # it).
                ev["learned"] = None
                # FAILING LANDS. Nothing hurt him — the clock just beat him —
                # so it is not pain, it is deflation: negative valence, a
                # dopamine dip, and enough frames for the slow cortisol to
                # actually move. And it COMPOUNDS: the third attempt at the
                # same maze is worse than the first, which is the difference
                # between a setback and a run that is not working.
                if self.chem is not None:
                    sting = min(1.0, 0.45 + 0.18 * (env.attempt - 1))
                    for _ in range(self.FAIL_FRAMES):
                        self.chem.update(valence=-sting, threat=0.2 * sting, focus=0.3)
                    self.chem.dopamine = max(0.15, self.chem.dopamine - 0.15 * sting)
                    self.chem.dominant_emotion = self.chem._classify_emotion(0.0)
                self.coach.outcome("failed", step=env.steps)
                if "combos" in self.can:
                    ev["learned"] = self._propose("out_of_time")
                else:
                    self._wonder_about_combos("the clock beat me going one step at a time")
                self._t("out_of_time", level=env.level, attempt=env.attempt, learned=ev["learned"])
                self._thought, self._thought_prio, self._say_aloud = None, -1, None
                self._think("out_of_time", level=env.level, attempt=env.attempt, learned=ev["learned"])
                ev["says"] = self._say_aloud
                self._last = ev
                return {"from": None, "place": self.place, "chosen": None, "new": 0, "probed": None}
            self._thought, self._thought_prio, self._say_aloud = None, -1, None   # a fresh thought each step
            self._learned_here = []                      # and a fresh vocabulary: only this step's percepts
            self._last_eaten = None                      # a bump does not reach on_arrive: clear it here
            self._learn(self._sense())                   # senses FIRST, then decide on what they gave him
            self._coach_tick()                           # and settle any warning against what is actually there
            # then test what he has been wondering. A verdict is a fact he
            # EARNED, so it goes through the same learning gate as a percept;
            # an open guess stays a guess and never enters the map.
            for fact in self.settle(env.steps):           # CubbyMan's: test, learn, grant
                self._think("derive", fact=fact)
            # WHAT HIS LAWS SAY ABOUT WHAT HE JUST SAW (WO-2.13). Before he
            # has asked how falling works this is always None, and a thing
            # hanging overhead is only a thing hanging overhead. After, the
            # same two sightings mean something is about to land on him, and
            # he has one step to be somewhere else.
            self._dodge = False
            drop = self.falling_at_me()
            if drop is not None:
                self._t("predict", what="something is falling at me", in_steps=drop,
                        because=self.already_know(self.FALLING_Q))
                self._think("predict", in_steps=drop)
                self._dodge = drop <= 1
            # how far off the nearest threat was when he chose — the last
            # moment he could act. This, not the distance at capture, is what
            # `_on_caught` turns into his berth.
            self._threat_seen_at = min((_manh(env.coords(self.place), g)
                                        for g in self.believed_ghosts()), default=None)
            if self._mine_wise() and env.place_mine(self.place):   # drop a trap on the way out
                self._learn([f"a trap is the marker of {self.place}"])
                ev["mined"] = list(env.coords(self.place))
                self._t("mine", place=self.place, left=env.mines_left,
                        ghost_distance=min((_manh(env.coords(self.place), g)
                                            for g in self.believed_ghosts()), default=None))
                self._think("mine", left=env.mines_left)
            danger = self.danger_cells()                 # the berth grows with learned fear
            self._plan_next = self.plan_next(avoid=danger)
            seen_left = bool(self.sighted - self._eaten_run)
            if self._plan_next is None and seen_left:
                # a pellet he has SEEN with no path on his map. If he does not
                # yet know moves can be chained, THIS is what raises the
                # question; if he does, it is what a JUMP is for.
                if "combos" not in self.can:
                    self._wonder_about_combos("a pellet I can see has no path on my map")
                elif "JUMP" not in self.library:
                    ev["learned"] = self._compose("JUMP", "stuck", ["hop", "hop"], None, "jump",
                                                  "two hops in one direction clear the hazard in between")
                    self._plan_next = self.plan_next(avoid=danger)
            elif (not seen_left and self.chem is not None and self.chem.dopamine > 0.5
                  and env.steps - self._last_proposal >= 12):
                ev["learned"] = self._propose("curious")         # nothing to chase: invent a move
            chase = self.chasing()
            if chase is not None and not self._was_chasing:
                self._t("chasing", to=chase, craving=round(self.chem.craving, 2),
                        tolerance=round(self.chem.tolerance, 2), took=self.chem.rewards_taken,
                        reach=self.chase_reach(),
                        went=_manh(env.coords(self.place), env.coords(chase)),
                        fright_left=env.frightened)
                self._think("chasing", to=chase)
            self._was_chasing = chase is not None
            if self._plan_next is not None:
                self._t("plan", to=self._plan_next,
                        goal=("ghost" if chase else "pellet" if seen_left else "frontier"))
            rec = super().step()                         # his move + eating + learning + traj
            chosen = rec.get("chosen") or ""
            if chosen == REST:                           # a step of rest in a safe spot: energy back, time spent
                env.energy = min(100, env.energy + REST_GAIN)
                self._t("rest", place=self.place, energy=env.energy)
            else:                                        # every move costs; combos cost more; nothing regenerates by itself
                env.energy = max(0, env.energy - self.move_cost(chosen))
            if chosen.startswith("jump_") or ("_" in chosen and chosen != REST):   # a superpower move
                pname = "JUMP" if chosen.startswith("jump_") else chosen.split("_")[0].upper()
                e = self.library.entries.get(pname)
                saved = (len(e["pattern"]) - 1) if e and e.get("pattern") else 1
                self.library.note_used(pname, saved, {"step": env.steps, "level": env.level, "move": chosen,
                                                      "landed": ("pellet" if self._last_eaten else "empty")})
                self._t("superpower_move", move=chosen, program=pname, saved_steps=saved,
                        to=self.place, energy=env.energy)
            ev["eaten"] = self._last_eaten
            if rec.get("probed"):
                self._think("probe", tried=rec["probed"])
            if chosen and "_" in chosen:
                self._think("superpower_move", name=pname, saved=saved)
            if self._thought is None:                    # nothing notable: the plan, or just moving on
                if self._plan_next is not None:
                    self._think("plan", to=self._plan_next, goal="pellet" if seen_left else "frontier")
                else:
                    self._think("idle", to=self.place)
            # then everything in the air comes down one cell, and the ceiling
            # may let go of something new. Both AFTER his move: he acts on what
            # he perceived, and the world resolves.
            fall = env.fall_turn(self.place)
            here = env.coords(self.place)
            # a landing he WITNESSED: close enough and in line of sight. Same
            # channel as everything else he perceives.
            watched = [c for c in fall["landed"]
                       if c != fall["hit"] and _manh(c, here) <= self.SIGHT
                       and env.line_of_sight(here, c)]
            if fall["hit"]:
                self.hit_by_falling += 1
                env.energy = max(0, env.energy - FALL_HURT)
                ev["hit_by_falling"] = list(fall["hit"])
                self._t("hit_by_falling", place=self.place, times=self.hit_by_falling,
                        energy=env.energy, knew=self.knows_falling())
                self._think("hit_by_falling", times=self.hit_by_falling)
                if self.chem is not None:
                    self.chem.update(threat=0.6, novelty=0.5)
            elif watched:
                self._t("saw_it_land", at=list(watched[0]), place=self.place,
                        knew=self.knows_falling())
                if self.chem is not None:
                    self.chem.update(novelty=0.5)
            # Either one is the thing his map has no account of, and that is
            # what sends the question out — not a schedule and not a hint. He
            # does not have to be hit to be puzzled; watching one thump down
            # beside him is enough, and it is the cheaper way to find out.
            if fall["hit"] or watched:
                self._wonder_about_falling("something came down out of nowhere")
            if env.maybe_drop(self.place):
                self._t("something_let_go", above=self.place)
            gh = env.ghost_turn(self.place)              # then the ghosts move
            if gh.get("trapped"):
                ev["trapped"] = gh["trapped"]
                self._t("trapped", n=gh["trapped"], bonus=TRAP_BONUS * gh["trapped"], total_score=env.total_score)
                self._think("trapped", n=gh["trapped"])
                if self.chem is not None:
                    self.chem.update(valence=0.9, novelty=0.3)
            if gh["eaten"]:
                ev["ate_ghost"] = True
                if self.chem is not None:
                    # THE BIG ONE. `reward` is the only signal that builds
                    # tolerance, and one frame per ghost, because eating two is
                    # two hits on the same receptor and not one bigger one.
                    for _ in range(int(gh["eaten"])):
                        self.chem.update(valence=1.0, reward=1.0, novelty=0.3)
                self._t("ghost_eaten", n=gh["eaten"], bonus=GHOST_BONUS * gh["eaten"],
                        **({"took": self.chem.rewards_taken,
                            "tolerance": round(self.chem.tolerance, 2),
                            "dopamine": round(self.chem.dopamine, 2)}
                           if self.chem is not None else {}))
            if gh["caught"]:
                fact = f"a ghost is the danger of {self.place}"
                self._learn([fact])
                last_seen = self._threat_seen_at         # where it was when he last chose
                self._on_caught()                        # fear + the shock + THE LESSON (the berth he keeps)
                ev["caught"] = "gameover" if env.game_over else True
                self._t("caught", place=self.place, lives=env.lives, fear=round(self.fear, 2),
                        game_over=env.game_over, learned=fact,
                        seen_at=last_seen, radius=self.danger_radius,
                        hurt=getattr(self, "_last_hurt", None),
                        chasing=self._was_chasing)       # caught mid-chase: what the wanting cost him
                self._think("caught", place=self.place)
                self.place = env.start
                self.ghost_belief.clear()                # respawned across the maze: the belief is void
                self.thing_belief.clear()                # and he is nowhere near whatever was overhead
            elif self.chem is not None:
                believed = self.believed_ghosts()
                threat = 0.0
                if believed and not env.frightened:
                    near = min(_manh(env.coords(self.place), g) for g in believed)
                    # threat scales over the berth HE learned rather than the
                    # old hardcoded `1.0 if near<=1 else 0.5 if near<=2`
                    threat = max(0.0, 1.0 - max(0, near - 1) / max(1, self.danger_radius))
                # ALWAYS one frame, even at threat 0. An ODE that only
                # integrates when something happens is not a clock: nothing
                # relaxes toward resting on a quiet stretch, pain never fades,
                # and the abstinence that walks tolerance back never counts.
                # The quiet steps are half of what these dynamics are made of.
                self.chem.update(threat=threat)
            if env.cleared:
                ev["beaten"] = True
            ev["says"] = self._say_aloud
            self._last = ev
            return rec

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

    def resp(self) -> dict:
        """THE RENDERER'S feed, not his. Everything below is read straight
        out of the world on purpose: it is what the browser draws for the
        HUMAN watching. Nothing here reaches a decision — the agent's own
        view of the same things is `self.sighted` and `self.ghost_belief`,
        and the gap between the two is visible on the page."""
        env, ev = self.env, self._last
        return {"to": list(env.coords(self.place)),
                "move": (self.traj[-1]["move"] if self.traj and not ev.get("failed") else None),
                "eaten": ev.get("eaten"), "learned": ev.get("learned"),
                "score": env.score, "total_score": env.total_score, "remaining": len(env.remaining),
                "beaten": bool(ev.get("beaten")) or not env.remaining, "failed": bool(ev.get("failed")),
                "level": env.level, "attempt": env.attempt, "planned": 0, "energy": env.energy,
                "steps": env.steps, "budget": env.budget,
                "ghosts": [list(g) for g in env.ghosts], "lives": env.lives, "caught": ev.get("caught"),
                "fear": round(self.fear, 2), "frightened": env.frightened,
                "ate_ghost": bool(ev.get("ate_ghost")), "ghost_bonus": GHOST_BONUS,
                "superpowers": list(self.powers),
                "mines": [list(c) for c in sorted(env.mines)], "mines_left": env.mines_left,
                "mined": ev.get("mined"), "trapped": ev.get("trapped", 0),
                # things in the air and what one of them did to him this step.
                # In the feed so the page can draw them; his own view of the
                # same things is `thing_belief`, which is two sightings deep
                # and can be stale, and the gap is the interesting part.
                "falling": [list(f["at"]) for f in env.fallers],
                "hit_by_falling": ev.get("hit_by_falling"),
                **self.affect()}

    def init_payload(self) -> dict:
        env = self.env
        return {"w": env.w, "h": env.h, "d": env.d, "level": env.level,
                "start": list(env.coords(env.start)),
                "walls": [list(c) for c in sorted(env.walls)],
                "hazards": [list(c) for c in sorted(env.hazards)],
                "pellets": [list(p) for p in sorted(env.remaining)], "total": len(env.pellets),
                "total_score": env.total_score, "energy": env.energy, "steps": env.steps,
                "budget": env.budget, "attempt": env.attempt,
                "ghosts": [list(g) for g in env.ghosts], "lives": env.lives,
                "ghost_colors": GHOST_COLORS[: env.n_ghosts], "fear": round(self.fear, 2),
                "power": [list(c) for c in sorted(env.power)], "superpowers": list(self.powers),
                "mines": [list(c) for c in sorted(env.mines)], "mines_left": env.mines_left,
                **self.affect()}

    def handle(self, text: str) -> dict:
        from identity import guess_lang
        lang = guess_lang(text)
        if self._STATUS.search(text) and not self._PLAY.search(text):   # a report, no stepping
            return {"offered": [self.status_line(lang)], "meta": {"status": True, "level": self.env.level}}
        # ANYTHING THAT IS NOT A COMMAND IS SOMEBODY TALKING TO HIM. Owner:
        # *"I should be able to chat with it live not only 2 messages."* So
        # conversation is the DEFAULT and the commands are the narrow case,
        # which is the right way round for something whose job is to have
        # company while it plays. `hear` decides what the words may do to a
        # body: nothing said here reaches `reward`, and only a checkable claim
        # about the maze reaches `threat`.
        if not self._PLAY.search(text):
            got = self.hear(text)
            return {"offered": [self.say_to_coach(got, lang)],
                    "meta": {"heard": got["kind"], "why": got.get("why"),
                             "coach": self.coach.state()}}
        m = re.search(r"\b(\d{1,3})\b", text)
        steps = int(m.group(1)) if m else 30
        rep = self.explore(steps)
        rep.update({"level": self.env.level, "lives": self.env.lives,
                    "score": self.env.score, "total_score": self.env.total_score})
        if lang == "fr":
            line = (f"Je suis au niveau {self.env.level} du labyrinthe — score total {self.env.total_score}, "
                    f"{self.env.lives} vies, et j'ai appris {rep['new_facts']} choses nouvelles. "
                    f"Regarde-moi en direct sur /pac !")
        else:
            line = (f"I'm on level {self.env.level} of the pacman maze — total score "
                    f"{self.env.total_score}, {self.env.lives} lives left, and I learned "
                    f"{rep['new_facts']} new things this run. Watch me live at /pac!")
        return {"offered": [line], "meta": rep}










class LivePac:
    """The live frontend's protocol over our brain — pacman_live's own
    contract: GET /init (the level), GET /state (one step per due poll; polled
    too soon -> the cached frame tagged `stale`, so any number of tabs see
    one steady game and he pauses when nobody watches), GET /next (advance
    the level after `beaten`), GET /assets/* (cubby's faces). The chat
    cortex ("play pacman for N") still drives the same CubbyGhost — the step
    lock keeps the two from interleaving a move."""

    def __init__(self, man: CubbyGhost, min_interval: float = 0.28, derive_every: int = 15) -> None:
        import threading
        self.man = man
        self.min_interval = float(min_interval)
        self.derive_every = int(derive_every)
        self._lock = threading.Lock()
        self._last = 0.0
        self._last_resp: dict | None = None
        self._html: str | None = None
        self.error: str | None = None

    def init_payload(self) -> dict:
        with self._lock:
            return self.man.init_payload()

    def poll(self) -> dict:
        import time
        with self._lock:
            now = time.monotonic()
            if self._last_resp is not None and now - self._last < self.min_interval:
                return {**self._last_resp, "stale": True}
            try:
                if not self.man.env.cleared:             # a cleared level waits for /next
                    self.man.step()
                    if len(self.man.traj) % self.derive_every == 0:
                        self.man.derive_counts()
                r = self.man.resp()
            except Exception as e:                       # a dead VM must not kill the server
                import traceback
                tb = traceback.format_exc().strip().splitlines()
                where = " | ".join(l.strip() for l in tb[-4:-1])   # the last frames: file:line + the line
                self.error = str(e)[:200]
                self.man._t("live_error", error=self.error, where=where)
                try:                                     # and on disk, so the trace survives the console
                    log = PROGRAMS_PATH.parent / "live_errors.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with open(log, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.error}\n" + "\n".join(tb) + "\n\n")
                except OSError:
                    pass
                r = {**(self._last_resp or self.man.resp()), "error": self.error}
            self._last = now
            self._last_resp = r
            return r

    def next_level(self) -> dict:
        with self._lock:
            if self.man.env.cleared:
                self.man.derive_counts()                 # join what the level collected
                self.man.next_level()
            self._last_resp = None
            return self.man.init_payload()

    def frontend(self) -> str | None:
        if self._html is None:
            self._html, missed = load_frontend()
            if missed and self._html is not None:
                self.man._t("frontend_patch_missed", patches=missed)
        return self._html

