"""pacworld - the maze itself: geometry, pellets, ghosts, falling things.

Wired: WIRED (the pac agent's env). This is ONE WORLD, implementing the
contract in `world.py`; nothing in it is meant to be reused by another.

Split out of `pacman.py` to put the boundary where it belongs: everything here
is ground truth about a maze, and nothing here is about an agent, a feeling or
a renderer. `PacVerse` is the plain grid; `GhostVerse` adds the hunt, the
power pellets, the mines and the fallers. The explorer reads this world only
through `observe` and `senses` - his beliefs are his own, and the gap between
the two is the point of the whole exercise.
"""
from __future__ import annotations

import collections
import os

from world import JUMP_COST, MOVES, OPP, _assignments, pattern_cost  # noqa: E402
__wiring__ = "WIRED"



def build_maze(w: int = 7, h: int = 7, d: int = 3, seed: int = 7,
               wall_p: float = 0.22, n_pellets: int = 16):
    """Verbatim port of pacman_3d.build_maze (same rng call order -> the
    same maze the planner demo runs on)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cells = [(x, y, z) for z in range(d) for y in range(h) for x in range(w)]
    start = (0, 0, 0)
    walls = set()
    for c in cells:
        if c != start and rng.random() < wall_p:
            walls.add(c)
    seen, dq = {start}, collections.deque([start])
    while dq:
        p = dq.popleft()
        for dx, dy, dz in MOVES.values():
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            if 0 <= q[0] < w and 0 <= q[1] < h and 0 <= q[2] < d and q not in walls and q not in seen:
                seen.add(q)
                dq.append(q)
    reach = seen - walls
    open_cells = [c for c in reach if c != start]
    rng.shuffle(open_cells)
    return start, walls, open_cells[:n_pellets], reach



class PacVerse:
    """The maze as a CubbyMan-shaped env: places are named cells, exits are
    the legal moves, observations are template facts. Ground truth (walls,
    pellet positions) lives HERE — his FactStore only ever holds what he
    discovered."""

    def __init__(self, w: int = 7, h: int = 7, d: int = 3, seed: int = 7,
                 wall_p: float = 0.22, n_pellets: int = 16) -> None:
        self.w, self.h, self.d = w, h, d
        start, self.walls, pellets, self.reach = build_maze(w, h, d, seed, wall_p, n_pellets)
        self.pellets = list(pellets)                     # initial, for the replay page
        self.remaining = set(pellets)
        self.score = 0
        self.start = self.cell(*start)

    @staticmethod
    def cell(x: int, y: int, z: int) -> str:
        return f"cell {x}-{y}-{z}"

    @staticmethod
    def coords(place: str) -> tuple[int, int, int]:
        x, y, z = place.split()[1].split("-")
        return int(x), int(y), int(z)

    def exits(self, place: str) -> dict[str, str]:
        x, y, z = self.coords(place)
        out = {}
        for m, (dx, dy, dz) in MOVES.items():
            q = (x + dx, y + dy, z + dz)
            if 0 <= q[0] < self.w and 0 <= q[1] < self.h and 0 <= q[2] < self.d \
                    and q not in self.walls:
                out[m] = self.cell(*q)
        return out

    def observe(self, place: str) -> list[str]:
        return [f"{nbr} is the {m} neighbor of {place}" for m, nbr in self.exits(place).items()]

    def try_move(self, place: str, move: str, offered: dict[str, str]) -> dict:
        """The world resolves a move ATTEMPT — he may try any direction his
        body has, and the world decides what happens. `offered` carries the
        superpower landings whose legality his library already resolved.
        A refusal is a result, not an error: it is how he learns a wall.

        A BASE direction is always re-resolved here, never taken from
        `offered` — `offered` is what he BELIEVES he can do, and the whole
        point is that his belief can be wrong."""
        if move not in MOVES:                            # a superpower move: the library already checked it
            if move in offered:
                return {"ok": True, "to": offered[move], "kind": "open"}
            return {"ok": False, "to": place, "kind": "wall"}
        x, y, z = self.coords(place)
        dx, dy, dz = MOVES[move]
        q = (x + dx, y + dy, z + dz)
        if not (0 <= q[0] < self.w and 0 <= q[1] < self.h and 0 <= q[2] < self.d):
            return {"ok": False, "to": place, "kind": "edge"}
        if q in self.walls:
            return {"ok": False, "to": place, "kind": "wall"}
        return {"ok": True, "to": self.cell(*q), "kind": "open"}

    def eat(self, place: str) -> bool:
        c = self.coords(place)
        if c in self.remaining:
            self.remaining.discard(c)
            self.score += 1
            return True
        return False

    @property
    def cleared(self) -> bool:
        """ONE BIT the world publishes: this maze is empty. The agent may
        perceive it — a cleared maze is visibly, audibly over — but not read
        `remaining` to work it out, which would hand him a pellet count he
        has no way to know (exp_r36 caught exactly that: 446 reads of
        `env.remaining` from inside his own step)."""
        return not self.remaining

    def progress(self) -> dict:
        """THE SCOREBOARD — what the cabinet displays, for the replay page and
        the run log. The renderer and the traces read this; his DECISIONS read
        his percepts. Keeping the two apart is what lets exp_r36's tripwire
        tell a leak from a legitimate read."""
        return {"score": self.score, "remaining": len(self.remaining),
                "total": len(self.pellets), "cleared": self.cleared}

    def all_facts(self) -> list[str]:
        return [f for c in sorted(self.reach) for f in self.observe(self.cell(*c))]



# ── the big game: ghosts, hazards, power stars, lives, levels ───────────────
FRIGHT_STEPS = 14                                        # pacman_live.py's constants


GHOST_BONUS = 5


GHOST_COLORS = ["#ff4d5e", "#27d3ff", "#ff9ff3", "#36e07a"]


PELLET_ENERGY, STAR_ENERGY = 4, 12                       # what eating gives back


REST_GAIN = 8                                            # per step of rest, in a safe spot


REST_BELOW, REST_UNTIL = 30, 60                          # start resting under 30, stop at 60 (hysteresis)


REST = "rest"                                            # the ASK candidate: stay put


TRAP_BONUS = 3                                           # a ghost walked into his trap


FALL_HURT = 15                                           # energy lost when something lands on him (WO-2.13)


MINE = "mine"                                            # the ASK candidate: drop a trap here, then move



class _Mulberry32:
    """pacman_live's seed-shared PRNG (cubbyverse/core/worldgen.py), ported
    bit-for-bit — same (W,H,D,seed) -> the same labyrinth the live game gets."""

    def __init__(self, seed: int) -> None:
        self.a = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def randint(self, n: int) -> int:
        return int(self.random() * n)



def _in(p, w, h, d):
    return 0 <= p[0] < w and 0 <= p[1] < h and 0 <= p[2] < d



def _reachable(start, blocked, w, h, d):
    seen, dq = {start}, collections.deque([start])
    while dq:
        p = dq.popleft()
        for dx, dy, dz in MOVES.values():
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            if _in(q, w, h, d) and q not in blocked and q not in seen:
                seen.add(q)
                dq.append(q)
    return seen



def carve_labyrinth(w, h, d, wall_fraction, seed, start=(0, 0, 0)):
    """Verbatim port of cubbyverse/core/worldgen.carve_labyrinth: straight
    wall runs, connectivity-checked (no pockets)."""
    rng = _Mulberry32(seed)
    runs = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))
    target = int(wall_fraction * w * h * d)
    walls: set = set()
    attempts = target * 10 + 40
    total = w * h * d
    while len(walls) < target and attempts > 0:
        attempts -= 1
        if d > 1 and rng.randint(8) == 0:
            dr = (0, 0, 1) if rng.randint(2) == 0 else (0, 0, -1)
        else:
            dr = runs[rng.randint(4)]
        length = 2 + rng.randint(3)
        x, y, z = rng.randint(w), rng.randint(h), rng.randint(d)
        run = []
        for i in range(length):
            c = (x + i * dr[0], y + i * dr[1], z + i * dr[2])
            if not _in(c, w, h, d) or c == start:
                break
            run.append(c)
        new = [c for c in run if c not in walls]
        if len(new) < 2:
            continue
        cand = walls | set(new)
        if len(_reachable(start, cand, w, h, d)) == total - len(cand):
            walls = cand
    return walls



def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])



class GhostVerse(PacVerse):
    """pacman_live.py's game, ported as a CubbyMan env: carved leveled
    labyrinths (exact level formulas + seed-shared walls), hazards, star
    power-pellets (farthest-point spread), ghosts with fright mode, lives.
    Simplified vs the original, on record: ghosts chase/flee greedily by
    Manhattan distance (theirs are mSA AdaptiveControllers with scent
    fields), and vaults/superpowers/letter-words/time-budget stay with the
    cubbyverse-side splice — without the jump superpower a vaulted pellet
    would be unreachable, so n_vaults=0 here."""

    # levels 1..N run with no ghosts: his classroom. Env-overridable
    # (CUBBYMAN_GHOST_FREE_LEVELS) so a test or a demo can turn the threat
    # back on at level 1 without editing the rule into the code.
    GHOST_FREE_LEVELS = 3

    def __init__(self, level: int = 1, seed_extra: int = 0,
                 ghost_free_levels: int | None = None,
                 fallers_from_level: int | None = None) -> None:
        self.total_score = 0
        self.lives = 3
        self.energy = 100
        self.mines_left = 0                              # traps: +1 per level, unused ones carry over
        self.mines: set = set()                          # cells holding a trap (ghosts only)
        self.seed_extra = int(seed_extra)
        self.ghost_free_levels = int(
            ghost_free_levels if ghost_free_levels is not None
            else os.environ.get("CUBBYMAN_GHOST_FREE_LEVELS", self.GHOST_FREE_LEVELS))
        self.fallers_from_level = int(
            fallers_from_level if fallers_from_level is not None
            else os.environ.get("CUBBYMAN_FALLERS_FROM_LEVEL", self.FALLERS_FROM_LEVEL))
        self._start_level(level)

    # ── traps (owner, 2026-09-02): ghosts only; 1 per level, cumulative ─────
    def place_mine(self, place: str) -> bool:
        c = self.coords(place)
        if self.mines_left <= 0 or c in self.mines or c in self.walls or c in self.hazards:
            return False
        self.mines.add(c)
        self.mines_left -= 1
        return True

    def _start_level(self, level: int) -> None:
        import numpy as np
        self.level = int(level)
        self.w = self.h = min(5 + level, 16)
        self.d = 3
        wall_p = min(0.10 + 0.012 * level, 0.20)
        hazard_p = min(0.04 + 0.010 * level, 0.10)
        n_pellets = 9 + 3 * level
        seed = level * 1313 + 7 + self.seed_extra
        start = (0, 0, 0)
        self.walls = carve_labyrinth(self.w, self.h, self.d, wall_p, seed, start)
        rng = np.random.default_rng(seed)
        self.hazards = set()
        for z in range(self.d):
            for y in range(self.h):
                for x in range(self.w):
                    c = (x, y, z)
                    if c != start and c not in self.walls and rng.random() < hazard_p:
                        self.hazards.add(c)
        blocked = self.walls | self.hazards
        self.reach = _reachable(start, blocked, self.w, self.h, self.d) - blocked
        opens = [c for c in self.reach if c != start]
        rng.shuffle(opens)
        self.pellets = list(opens[:n_pellets])
        self.remaining = set(self.pellets)
        n_power = min(1 + level // 2, 4)
        self.power = set()
        for _ in range(min(n_power, len(self.pellets))):
            ref = self.power or {start}
            self.power.add(max((p for p in self.pellets if p not in self.power),
                               key=lambda p: min(_manh(p, q) for q in ref)))
        self._power_all = set(self.power)
        self.score = 0
        self.frightened = 0
        self.start = self.cell(*start)
        # ghosts: count/speed per the live game; spawns = farthest open cells.
        # The first GHOST_FREE_LEVELS levels run with NO ghosts at all (Nick,
        # 2026-09-15): learning what a maze IS and surviving a threat at the
        # same time produces neither. An empty maze is his classroom; the
        # ghosts arrive once he has a map to run on.
        self.n_ghosts = 0 if level <= self.ghost_free_levels else min(2 + level // 2, 4)
        self.ghost_speed = min(0.55 + 0.035 * level, 0.85)
        self._grng = np.random.default_rng(level * 31 + 1)
        far = sorted(self.reach, key=lambda c: -_manh(c, start))
        self.ghost_spawn = ([far[i % len(far)] for i in range(self.n_ghosts)]
                            if (far and self.n_ghosts) else [])
        self.ghosts = list(self.ghost_spawn)
        self.fallers: list[dict] = []                    # things in the air, one cell down per step
        self.game_over = False
        self.mines = set()                               # a new maze: the floor is clean
        self.mines_left = getattr(self, "mines_left", 0) + 1   # one more trap; unused ones carry over
        # the live game's time budget (the hidden-word/letter mechanic was
        # dropped 2026-09-02 — his THOUGHTS are the speech surface now)
        self.budget = int(46 + 8 * level)
        self.attempt = 1
        self.steps = 0

    def begin_run(self) -> None:
        """Out of time: same level, fresh run (pacman_live's _begin_run)."""
        self.remaining = set(self.pellets)
        self.power = set(self._power_all)
        self.score = 0
        self.steps = 0
        self.frightened = 0
        self.ghosts = list(self.ghost_spawn)
        self.fallers = []

    # level-scoped names so facts stay true forever across levels
    def cell(self, x: int, y: int, z: int) -> str:
        return f"level-{self.level} cell {x}-{y}-{z}"

    @staticmethod
    def coords(place: str) -> tuple[int, int, int]:
        x, y, z = place.split()[-1].split("-")
        return int(x), int(y), int(z)

    def exits(self, place: str) -> dict[str, str]:
        x, y, z = self.coords(place)
        out = {}
        for m, (dx, dy, dz) in MOVES.items():
            q = (x + dx, y + dy, z + dz)
            if _in(q, self.w, self.h, self.d) and q not in self.walls and q not in self.hazards:
                out[m] = self.cell(*q)
        return out

    def _open(self, q) -> bool:
        return _in(q, self.w, self.h, self.d) and q not in self.walls and q not in self.hazards

    def try_move(self, place: str, move: str, offered: dict[str, str]) -> dict:
        """Same contract as PacVerse.try_move, with this world's two kinds of
        obstacle. A hazard is SEEN (it glows) before it is hit; a plain wall
        is not, so walking into one is the only way to find it."""
        if move not in MOVES:                            # a superpower move / REST: already resolved
            if move in offered:
                return {"ok": True, "to": offered[move], "kind": "open"}
            return {"ok": False, "to": place, "kind": "wall"}
        x, y, z = self.coords(place)
        dx, dy, dz = MOVES[move]
        q = (x + dx, y + dy, z + dz)
        if not _in(q, self.w, self.h, self.d):
            return {"ok": False, "to": place, "kind": "edge"}
        if q in self.walls:
            return {"ok": False, "to": place, "kind": "wall"}
        if q in self.hazards:
            return {"ok": False, "to": place, "kind": "hazard"}
        return {"ok": True, "to": self.cell(*q), "kind": "open"}

    # ── vision: a ray that a wall stops, not a map lookup ───────────────────
    def line_of_sight(self, a: tuple, b: tuple) -> bool:
        """Can a straight ray from `a` reach `b` without a wall in the way?
        A 3-D DDA sampled along the longest axis. This is what makes SIGHT a
        SENSOR instead of a map read: he sees down a corridor, not through
        the stone."""
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        n = max(abs(d[0]), abs(d[1]), abs(d[2]))
        if n == 0:
            return True
        for i in range(1, n + 1):
            c = tuple(a[k] + round(d[k] * i / n) for k in range(3))
            if c == b:
                break
            if c in self.walls:
                return False
        return True

    def senses(self, place: str, radius: int) -> dict:
        """What a sensor AT `place` picks up: pellets, stars and ghosts that
        are inside `radius` AND in line of sight. Nothing else about the
        level is readable from here — this is the whole channel between the
        world and what he can come to believe (2026-09-15: before this, the
        agent read env.remaining and env.ghosts directly, i.e. the solved map
        and every ghost's exact position through walls at any range)."""
        here = self.coords(place)
        seen = {"pellets": [], "stars": [], "ghosts": [], "at": here, "radius": radius}
        for c in self.remaining:
            if _manh(c, here) <= radius and self.line_of_sight(here, c):
                (seen["stars"] if c in self.power else seen["pellets"]).append(c)
        for g in self.ghosts:
            if _manh(g, here) <= radius and self.line_of_sight(here, g):
                seen["ghosts"].append(g)
        # Things in the air, as POSITIONS and nothing else. The world does not
        # say a thing is falling — it says where the thing is. That it is
        # falling is his to work out from it being lower than it was, which is
        # exactly what the law he has to go and ask for tells him to look at.
        seen["things"] = [tuple(f["at"]) for f in self.fallers
                          if _manh(f["at"], here) <= radius and self.line_of_sight(here, f["at"])]
        return seen

    # ── things that fall (WO-2.13) ─────────────────────────────────────────
    # The ceiling gives way from level 2 on. There is no announcement, no
    # warning glow and no timer: a thing appears up the column and comes down
    # one cell per step, which is enough to be hit by and enough to reason
    # about — and the reasoning is not in here. The world falls; understanding
    # falling is his problem, and the first time it is solved by asking a world
    # that already knows (knowledge.PhysicsWorld).
    FALLERS_FROM_LEVEL = 2
    FALL_EVERY = 8                                       # steps between drops
    FALL_HEIGHT = 3                                      # how far up one starts

    @property
    def fallers_on(self) -> bool:
        return self.level >= self.fallers_from_level

    def maybe_drop(self, place: str) -> tuple | None:
        """Start something falling down the column above him, if the column is
        clear and nothing is already in the air."""
        if not self.fallers_on or self.fallers or self.steps % self.FALL_EVERY:
            return None
        x, y, z = self.coords(place)
        top = min(y + self.FALL_HEIGHT, self.h - 1)
        if top <= y:
            return None
        if any((x, k, z) in self.walls for k in range(y + 1, top + 1)):
            return None                                  # it would be caught on the way down
        self.fallers.append({"at": (x, top, z), "let_go": (x, top, z), "since": self.steps})
        return (x, top, z)

    def fall_turn(self, place: str) -> dict:
        """Every falling thing comes down one cell. It stops on stone, and it
        does not steer: where it lands is straight below where it let go —
        which is what makes stepping aside enough, and what makes the law worth
        holding."""
        here = self.coords(place)
        hit, landed = None, []
        for f in list(self.fallers):
            x, y, z = f["at"]
            below = (x, y - 1, z)
            if not _in(below, self.w, self.h, self.d) or below in self.walls:
                self.fallers.remove(f)                   # it came to rest on something
                landed.append(f["at"])
                continue
            f["at"] = below
            if below == here:
                self.fallers.remove(f)
                landed.append(below)
                hit = below
        return {"hit": hit, "landed": landed, "flying": [tuple(f["at"]) for f in self.fallers]}

    def power_moves(self, place: str, library, energy: int) -> dict[str, str]:
        """The superpower moves legal HERE from his program library: JUMP =
        a hop over ONE hazard (costs energy); a pattern = its slots
        instantiated with directions, every cell on the way open. Move names
        carry the instantiation (`knight_right_right_up`); `jump_*` keeps the
        live page's hop animation."""
        x, y, z = self.coords(place)
        out = {}
        for name, e in library.active().items():         # never retired ones; never his TOOLS (forge entries have no pattern)
            if e["kind"] == "jump":
                if energy < JUMP_COST:
                    continue
                for d, (dx, dy, dz) in MOVES.items():
                    mid, land = (x + dx, y + dy, z + dz), (x + 2 * dx, y + 2 * dy, z + 2 * dz)
                    if _in(mid, self.w, self.h, self.d) and mid in self.hazards and self._open(land):
                        out[f"jump_{d}"] = self.cell(*land)
                continue
            if e["kind"] != "pattern" or not e.get("pattern"):
                continue
            pattern = e["pattern"]
            if energy < pattern_cost(pattern):          # can't afford it: not offered
                continue
            for asg in _assignments(pattern):
                cur, ok = (x, y, z), True
                for ch in pattern:
                    dx, dy, dz = MOVES[asg[ch]]
                    cur = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                    if not self._open(cur):
                        ok = False
                        break
                if ok and cur != (x, y, z):
                    out[f"{name.lower()}_{'_'.join(asg[ch] for ch in pattern)}"] = self.cell(*cur)
        return out

    def observe(self, place: str) -> list[str]:
        """Open-neighbor facts, plus the hazards he can SEE glowing red next
        to him — plain walls stay invisible until a refused move teaches them."""
        obs = super().observe(place)
        x, y, z = self.coords(place)
        for m, (dx, dy, dz) in MOVES.items():
            q = (x + dx, y + dy, z + dz)
            if _in(q, self.w, self.h, self.d) and q in self.hazards:
                obs.append(f"a hazard is the {m} neighbor of {place}")
        return obs

    def eat(self, place: str) -> dict:
        c = self.coords(place)
        out = {"pellet": False, "power": False}
        if c in self.remaining:
            self.remaining.discard(c)
            self.score += 1
            self.total_score += 1
            out["pellet"] = True
            if c in self.power:
                self.power.discard(c)
                self.frightened = FRIGHT_STEPS
                out["power"] = True
        out["cleared"] = self.cleared
        return out

    def _respawn(self, i: int) -> tuple:
        """Where ghost `i` reappears. With no spawns (a ghost-free level there
        can be no ghost to respawn) this is never reached from a live ghost,
        but it must not divide by zero if it is."""
        return self.ghost_spawn[i % len(self.ghost_spawn)] if self.ghost_spawn else self.coords(self.start)

    def ghost_turn(self, cubby: str) -> dict:
        """One ghost move each (probabilistic speed), then resolve contact:
        frightened -> eaten (+bonus, respawn); else -> caught (a life)."""
        pos = self.coords(cubby)
        if self.frightened > 0:
            self.frightened -= 1
        placed = []
        for i, g in enumerate(self.ghosts):
            if self._grng.random() >= self.ghost_speed:
                placed.append(g)
                continue
            cands = [q for q in
                     [(g[0] + dx, g[1] + dy, g[2] + dz) for dx, dy, dz in MOVES.values()]
                     if _in(q, self.w, self.h, self.d) and q not in self.walls
                     and q not in self.hazards and q not in placed] or [g]
            key = (lambda q: _manh(q, pos)) if self.frightened == 0 else (lambda q: -_manh(q, pos))
            placed.append(min(cands, key=lambda q: (key(q), self._grng.random())))
        self.ghosts = placed
        out = {"caught": False, "eaten": 0, "trapped": 0}
        for i, g in enumerate(self.ghosts):              # a trap fires on the ghost that steps on it
            if g in self.mines:
                self.mines.discard(g)
                self.ghosts[i] = self._respawn(i)
                self.total_score += TRAP_BONUS
                out["trapped"] += 1
        for i, g in enumerate(self.ghosts):
            if g == pos:
                if self.frightened > 0:
                    self.total_score += GHOST_BONUS
                    self.ghosts[i] = self._respawn(i)
                    out["eaten"] += 1
                else:
                    out["caught"] = True
        if out["caught"]:
            self.lives -= 1
            self.frightened = 0
            self.ghosts = list(self.ghost_spawn)
            if self.lives <= 0:
                self.game_over = True
        return out

    def restart_run(self) -> None:
        """After game over: same level, fresh lives — his world model stands."""
        self.lives = 3
        self.begin_run()
        self.game_over = False
