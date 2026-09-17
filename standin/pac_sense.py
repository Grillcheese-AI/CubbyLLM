"""pac_sense - what he perceives, what he believes, and the laws he works out.

Wired: WIRED (mixed into `pacman.CubbyGhost`).

THE RULE THE WHOLE PROJECT RESTS ON lives here: the world is the territory and
this is the map. Everything he decides reads `ghost_belief` and `thing_belief`,
never `env.ghosts` - a ghost he cannot sense is a ghost he does not know about,
and a belief that has gone stale is forgotten rather than trusted. The gap
between the two is the point of the exercise, and it is visible on the page.

The laws are the other half: he does not start knowing that things fall or that
pellets stay put. He notices, he frames a question, the VM settles it, and only
then does the same sighting mean something new.
"""
from __future__ import annotations

from pacworld import FRIGHT_STEPS, PELLET_ENERGY, STAR_ENERGY, _manh  # noqa: E402
from world import MOVES  # noqa: E402

__wiring__ = "WIRED"


class SenseMixin:

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
