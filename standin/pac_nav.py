"""pac_nav - where to go, what it costs, and how much room to give a ghost.

Wired: WIRED (mixed into `pacman.CubbyGhost`).

Planning over the map he has, not the maze that exists: `_known_graph` is built
from what he has learned, so a route through a corridor he has never seen is
not a route he can take. `danger_radius` is the berth he keeps, and it is
LEARNED from the distances he was actually caught at rather than fitted to how
frightened he feels - a berth fitted to fear is the thing that was replaced.
"""
from __future__ import annotations

import collections

from pacworld import REST, REST_BELOW, REST_UNTIL, _manh  # noqa: E402
from world import JUMP_COST, MOVE_COST, pattern_cost  # noqa: E402

__wiring__ = "WIRED"


class NavMixin:

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
    def learned_radius(self) -> int:
        """The berth his own experience has taught him, and nothing else.

        Not a constant, and not a curve fitted to `fear`. It is the largest
        distance at which a ghost he could see went on to catch him. Before
        anything catches him it is 1: a body knows only "touching me", and
        nothing has taught it otherwise yet.

        This replaced `1 if fear < 1.0 else 2 if fear < 2.4 else 3`
        (2026-09-15). Those thresholds were invented; this number is
        measured, it comes from his own experience, and it needs no
        retraining to move — the same reason every other rule here had to
        go.

        EVERY MEASUREMENT READS THIS ONE, never `danger_radius` below: what he
        learns from a catch has to come from the catch, and a berth that was
        temporarily wide because he was alarmed would otherwise bake its own
        alarm into the lesson. Behaviour reads the composite; learning reads
        this."""
        if not self.caught_at:
            return 1
        return max(1, min(self.MAX_DANGER_RADIUS, max(self.caught_at)))

    WARY_FROM = 0.10            # caution this far over its resting value buys a tile
    MAX_WARY = 2

    @property
    def wariness(self) -> int:
        """Extra berth because his body is alarmed RIGHT NOW. 0 when calm.

        THIS IS THE PATH FROM CHEMISTRY TO BEHAVIOUR, and until now there
        wasn't one. Two reads of `self.chem` existed in the whole decision
        path, both of `craving`, in `chase_reach`. So threat could spike, the
        compass could read fear, he could say frightened things — and take
        exactly the same step he would have taken calm. The affect stack was
        decorative at the point where it should have cost something.

        That is the open loop I described in GrillCheese's plasticity: state
        that is written every turn and read by nothing. It was true here too,
        and finding it while building an experiment ON this path is the reason
        the experiment was worth building before running.

        Driven by the CAUTION knob, not by the coach and no longer by raw
        noradrenaline. A warning is not special: a ghost he saw, a catch he is
        still carrying and a person shouting all raise the same signal, and all
        of them should make him keep more room.

        The knob rather than the hormone, because `modulation()` is the one
        place the hormones are turned into world-agnostic gains, and a second
        place that reads a raw hormone is a second opinion about what alarm
        means. `choosing` weighs PREFERENCES with the same knob; this sets
        where a CONSTRAINT trips. Those are different jobs — a filter may not
        talk a body out of fleeing — but they should not disagree about how
        alarmed it is.

        Transient by construction. NE decays back to quiescent within a few
        frames, so being told about a ghost buys caution for about as long as
        the warning is worth anything — it never becomes part of what he
        knows. What he KNOWS is `learned_radius`, and only being caught
        changes that."""
        if self.chem is None:
            return 0
        import choosing
        over = self.chem.modulation()["caution"] - choosing.neutral()["caution"]
        return 0 if over <= 0 else min(self.MAX_WARY, int(round(over / self.WARY_FROM)))

    @property
    def danger_radius(self) -> int:
        """What he actually keeps: what he has learned, plus how alarmed he is.

        Behaviour reads this. Measurement reads `learned_radius`."""
        return max(1, min(self.MAX_DANGER_RADIUS, self.learned_radius + self.wariness))


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
