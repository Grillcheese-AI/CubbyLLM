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
from pac_affect import AffectMixin  # noqa: E402
from pac_forge import ForgeMixin  # noqa: E402
from pac_nav import NavMixin  # noqa: E402
from pac_sense import SenseMixin  # noqa: E402
from pac_speech import SpeechMixin  # noqa: E402
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



































class CubbyGhost(ForgeMixin, SpeechMixin, SenseMixin, NavMixin, AffectMixin,
                 CubbyPac):
    """cubby-man in the big game: the same explorer brain, now hunted. Ghost
    proximity feeds THREAT into the neurochemistry (anxious when chased, bold
    when they are frightened), being caught is learned as a danger fact and
    raises a learned ghost-fear (pacman_live's exact +0.7 / ×0.9 rule),
    running out of the level's time budget restarts the run (his world model
    stands, so attempt 2 knows the maze), letter pellets spell the level's
    hidden word, and SPEECH — the grounded word when its percept recurs —
    goes through CubbyTalk's ASK like every other utterance. `resp()` /
    `init_payload()` speak the live frontend's exact schema."""

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

    def bind(self, brain) -> None:
        super().bind(brain)
        self.brain = brain
        from forge import ToolForge
        self.forge = ToolForge(brain.emitter, self.library, exe=self.exe, trace=self._t)
        self._last_forge = -99

    def _seed_basics(self) -> None:
        self.world.add("cubbyman is the explorer of the pacman maze")
        self.world.add(f"{self.env.start} is the start of level-{self.env.level}")

    @property
    def beaten(self) -> bool:
        return False                                     # levels continue; the game never "ends"

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
        da_before = self.chem.dopamine if self.chem is not None else None
        if self.chem is not None:
            relief = min(1.0, 0.6 + 0.2 * (attempts - 1))
            for _ in range(self.CLEAR_FRAMES):
                self.chem.update(valence=relief, novelty=0.4, social=0.2)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.4)
        self.coach.outcome("cleared", step=self.env.steps,
                           magnitude=self._da_moved(da_before))
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
                da_before = self.chem.dopamine if self.chem is not None else None
                if self.chem is not None:
                    sting = min(1.0, 0.45 + 0.18 * (env.attempt - 1))
                    for _ in range(self.FAIL_FRAMES):
                        self.chem.update(valence=-sting, threat=0.2 * sting, focus=0.3)
                    self.chem.dopamine = max(0.15, self.chem.dopamine - 0.15 * sting)
                    self.chem.dominant_emotion = self.chem._classify_emotion(0.0)
                self.coach.outcome("failed", step=env.steps,
                                   magnitude=self._da_moved(da_before))
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

    def say(self, text: str) -> dict:
        """Somebody at the side of the maze says something. -> his reply.

        STRAIGHT TO THE COACH, past `CubbyBrain.route`, and that is the point
        rather than a shortcut. Routing exists to decide what a sentence typed
        at a general-purpose assistant is FOR — a question, a fact to
        remember, a game command — and it scores the pac cortex on the game's
        own command words. So a warning about a ghost reached `hear()` and
        *"you can do it!!"* went to the reasoning cortex and came back with
        the don't-know line: the encouragement half of the channel, which is
        the half that was asked for first, never arrived.
        Owner: *"per example: you can do it!! or be careful of the ghost!"*

        A sentence typed into the MAZE WINDOW needs no such decision. That
        window is a person watching him play, so everything typed there is
        somebody talking to him and routing has nothing left to work out.
        `handle` still serves the console, commands and all.

        Under the step lock, so a reply cannot interleave with a move: `hear`
        runs the chemistry forward a few frames and `say_to_coach` reads the
        body it leaves behind."""
        from identity import guess_lang
        text = " ".join(str(text or "").split())
        if not text:
            return {"error": "empty text"}
        with self._lock:
            got = self.man.hear(text)
            reply = self.man.say_to_coach(got, guess_lang(text))
            return {"said": text, "reply": reply, "heard": got["kind"],
                    "why": got.get("why"), "coach": self.man.coach.state()}

    def frontend(self) -> str | None:
        if self._html is None:
            self._html, missed = load_frontend()
            if missed and self._html is not None:
                self.man._t("frontend_patch_missed", patches=missed)
        return self._html

