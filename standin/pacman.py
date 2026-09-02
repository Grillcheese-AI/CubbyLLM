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
import pathlib
import re

__wiring__ = "WIRED"

from verse import CubbyMan  # noqa: E402

PACMAN_3D = pathlib.Path(r"C:\Users\grill\Documents\GitHub\cubbyverse\examples\pacman_3d.py")
REPLAY_OUT = pathlib.Path(r"C:/tmp/cubbyman_pacman_3d.html")

# the six moves, exactly as pacman_3d.MOVES (pacman_live imports these too)
MOVES = {"right": (1, 0, 0), "left": (-1, 0, 0), "up": (0, 1, 0),
         "down": (0, -1, 0), "forward": (0, 0, 1), "back": (0, 0, -1)}
OPP = {"right": "left", "left": "right", "up": "down", "down": "up",
       "forward": "back", "back": "forward"}


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

    def eat(self, place: str) -> bool:
        c = self.coords(place)
        if c in self.remaining:
            self.remaining.discard(c)
            self.score += 1
            return True
        return False

    def all_facts(self) -> list[str]:
        return [f for c in sorted(self.reach) for f in self.observe(self.cell(*c))]


class CubbyPac(CubbyMan):
    """cubby-man in the pac maze: same explorer brain, six directions, and
    arriving on a pellet EATS it (a permanent discovery fact)."""

    name = "pacman"
    CORTEX = "pacman"
    DIR_NAMES = tuple(MOVES)
    OPP = OPP
    _GO = re.compile(r"\b(pac[- ]?man|pellets?|maze|labyrinthe)\b", re.I)

    def __init__(self, env: PacVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35) -> None:
        self.traj: list[dict] = []
        self._last_eaten = None
        super().__init__(env or PacVerse(), exe=exe, seed=seed, probe=probe)

    def _seed_basics(self) -> None:
        self.world.add("cubbyman is the explorer of the pacman maze")
        self.world.add(f"{self.env.start} is the start of the pacman maze")

    def on_arrive(self, place: str) -> int:
        self._last_eaten = None
        if self.env.eat(place):
            self._last_eaten = list(self.env.coords(place))
            fact = f"pellet {self.env.score} is the discovery of {place}"
            self._t("eat", place=place, pellet=self.env.score,
                    remaining=len(self.env.remaining))
            return self._learn([fact])
        return 0

    def step(self) -> dict:
        with self._step_lock:                            # move + traj append are one unit
            rec = super().step()
            self.traj.append({"to": list(self.env.coords(self.place)), "move": rec["chosen"],
                              "eaten": self._last_eaten, "score": self.env.score,
                              "remaining": len(self.env.remaining)})
            return rec

    @property
    def beaten(self) -> bool:
        return not self.env.remaining

    def handle(self, text: str) -> dict:
        m = re.search(r"\b(\d{1,3})\b", text)
        steps = int(m.group(1)) if m else 30
        rep = self.explore(steps)
        rep.update({"score": self.env.score, "pellets_total": len(self.env.pellets),
                    "beaten": not self.env.remaining})
        replay = render_replay(self.env, self.traj)
        rep["replay"] = str(replay) if replay else None
        beaten = " The maze is BEATEN!" if rep["beaten"] else ""
        line = (f"I played the pacman maze: {self.env.score} of {len(self.env.pellets)} pellets "
                f"in {steps} steps, and learned {rep['new_facts']} new things on the way.{beaten}"
                + (f" Watch my run: {replay}" if replay else ""))
        return {"offered": [line], "meta": rep}


# ── the big game: ghosts, hazards, power stars, lives, levels ───────────────
FRIGHT_STEPS = 14                                        # pacman_live.py's constants
GHOST_BONUS = 5
GHOST_COLORS = ["#ff4d5e", "#27d3ff", "#ff9ff3", "#36e07a"]


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

    def __init__(self, level: int = 1, seed_extra: int = 0) -> None:
        self.total_score = 0
        self.lives = 3
        self.seed_extra = int(seed_extra)
        self._start_level(level)

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
        self.score = 0
        self.frightened = 0
        self.start = self.cell(*start)
        # ghosts: count/speed per the live game; spawns = farthest open cells
        self.n_ghosts = min(2 + level // 2, 4)
        self.ghost_speed = min(0.55 + 0.035 * level, 0.85)
        self._grng = np.random.default_rng(level * 31 + 1)
        far = sorted(self.reach, key=lambda c: -_manh(c, start))
        self.ghost_spawn = [far[i % len(far)] for i in range(self.n_ghosts)] if far else [start]
        self.ghosts = list(self.ghost_spawn)
        self.game_over = False

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
        return out

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
        out = {"caught": False, "eaten": 0}
        for i, g in enumerate(self.ghosts):
            if g == pos:
                if self.frightened > 0:
                    self.total_score += GHOST_BONUS
                    self.ghosts[i] = self.ghost_spawn[i % len(self.ghost_spawn)]
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
        self.remaining = set(self.pellets)
        self.score = 0
        self.frightened = 0
        self.ghosts = list(self.ghost_spawn)
        self.game_over = False


class CubbyGhost(CubbyPac):
    """cubby-man in the big game: the same explorer brain, now hunted. Ghost
    proximity feeds THREAT into the neurochemistry (he gets anxious when
    chased, bold when they are frightened), being caught is learned as a
    danger fact, and clearing a level advances to the next maze while every
    fact he ever learned stays true (level-scoped names)."""

    def __init__(self, env: GhostVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35) -> None:
        super().__init__(env or GhostVerse(), exe=exe, seed=seed, probe=probe)

    def _seed_basics(self) -> None:
        self.world.add("cubbyman is the explorer of the pacman maze")
        self.world.add(f"{self.env.start} is the start of level-{self.env.level}")

    @property
    def beaten(self) -> bool:
        return False                                     # levels continue; the game never "ends"

    def on_arrive(self, place: str) -> int:
        self._last_eaten = None
        ate = self.env.eat(place)
        new = 0
        if ate["pellet"]:
            self._last_eaten = list(self.env.coords(place))
            fact = f"pellet {self.env.score} is the discovery of {place}"
            self._t("eat", place=place, pellet=self.env.score, power=ate["power"],
                    remaining=len(self.env.remaining))
            new += self._learn([fact])
            if self.chem is not None:
                self.chem.update(valence=0.8 if ate["power"] else 0.3)
        return new

    def _pick(self, exits: dict[str, str]) -> str:
        """Fear-aware: never step onto a hunting ghost if there is any other
        exit; when they are frightened, hunt them instead."""
        ghost_cells = {self.env.cell(*g) for g in self.env.ghosts}
        if self.env.frightened > 0:
            hunt = [m for m, p in exits.items() if p in ghost_cells]
            if hunt:
                return hunt[0]
        else:
            safe = {m: p for m, p in exits.items() if p not in ghost_cells}
            if safe:
                exits = safe
        return super()._pick(exits)

    def step(self) -> dict:
        with self._step_lock:
            if self.env.game_over:
                self.env.restart_run()
                self.place = self.env.start
                self._t("restart", level=self.env.level, lives=self.env.lives)
            rec = super().step()                         # his move + eating + learning
            ev = self.env.ghost_turn(self.place)         # then the ghosts move
            if ev["eaten"]:
                self._t("ghost_eaten", n=ev["eaten"], bonus=GHOST_BONUS * ev["eaten"])
                if self.chem is not None:
                    self.chem.update(valence=1.0)
            if ev["caught"]:
                fact = f"a ghost is the danger of {self.place}"
                self._learn([fact])
                self._t("caught", place=self.place, lives=self.env.lives,
                        game_over=self.env.game_over, learned=fact)
                self.place = self.env.start
                if self.chem is not None:
                    self.chem.update(threat=1.0, valence=-0.8)
            elif self.chem is not None and self.env.ghosts:
                near = min(_manh(self.env.coords(self.place), g) for g in self.env.ghosts)
                threat = 0.0 if self.env.frightened else (1.0 if near <= 1 else 0.5 if near <= 2 else 0.0)
                if threat:
                    self.chem.update(threat=threat)
            if not self.env.remaining:                   # level cleared -> the next maze
                nxt = self.env.level + 1
                self._t("level_up", cleared=self.env.level, next=nxt,
                        total_score=self.env.total_score)
                self.env._start_level(nxt)
                self.place = self.env.start
                self._seed_basics()
                self._learn(self.env.observe(self.place))
                self.visits[self.place] = self.visits.get(self.place, 0) + 1
            return rec

    def handle(self, text: str) -> dict:
        m = re.search(r"\b(\d{1,3})\b", text)
        steps = int(m.group(1)) if m else 30
        rep = self.explore(steps)
        rep.update({"level": self.env.level, "lives": self.env.lives,
                    "score": self.env.score, "total_score": self.env.total_score})
        line = (f"I'm on level {self.env.level} of the pacman maze — total score "
                f"{self.env.total_score}, {self.env.lives} lives left, and I learned "
                f"{rep['new_facts']} new things this run. Watch me live at /pac!")
        return {"offered": [line], "meta": rep}


class LivePac:
    """Poll-driven live play — the pacman_live.py pattern: the browser polls
    /pac/state, each due poll advances ONE VM-guarded step (rate-limited so
    extra tabs can't race him, paused the moment nobody polls), and the
    response is everything the live three.js page needs. The chat cortex
    ("play pacman for N") still works on the same CubbyPac — the step lock
    keeps the two drivers from interleaving a move."""

    def __init__(self, man: CubbyPac, min_interval: float = 0.35, derive_every: int = 15) -> None:
        import threading
        self.man = man
        self.min_interval = float(min_interval)
        self.derive_every = int(derive_every)
        self._lock = threading.Lock()
        self._last = 0.0
        self._error: str | None = None

    def poll(self) -> dict:
        import time
        stepped = False
        with self._lock:
            now = time.monotonic()
            if not self.man.beaten and self._error is None and now - self._last >= self.min_interval:
                self._last = now
                try:
                    self.man.step()
                    stepped = True
                    if len(self.man.traj) % self.derive_every == 0:
                        self.man.derive_counts()
                    if self.man.beaten:                  # the last pellet: join what the run collected
                        self.man.derive_counts()
                        render_replay(self.man.env, self.man.traj)
                except Exception as e:                   # a dead VM must not kill the server
                    self._error = str(e)[:200]
        return self.state(stepped)

    def state(self, stepped: bool = False) -> dict:
        man, env = self.man, self.man.env
        out = {"w": env.w, "h": env.h, "d": env.d, "start": list(env.coords(env.start)),
               "walls": [list(w) for w in sorted(env.walls)],
               "pellets": [list(p) for p in sorted(env.remaining)],
               "pos": list(env.coords(man.place)),
               "move": man.traj[-1]["move"] if man.traj else None,
               "score": env.score, "total": len(env.pellets), "beaten": man.beaten,
               "steps": len(man.traj), "facts": len(man.world),
               "walls_learned": len(man.walls), "derived": len(man.derived),
               "emotion": man.chem.dominant_emotion if man.chem is not None else "neutral",
               "stepped": stepped, "error": self._error}
        if isinstance(env, GhostVerse):                  # the big game's extras
            out.update({"hazards": [list(h) for h in sorted(env.hazards)],
                        "ghosts": [list(g) for g in env.ghosts],
                        "ghost_colors": GHOST_COLORS[: env.n_ghosts],
                        "power": [list(p) for p in sorted(env.power)],
                        "frightened": env.frightened, "lives": env.lives,
                        "level": env.level, "total_score": env.total_score})
        return out


def render_replay(env: PacVerse, traj: list[dict],
                  out: pathlib.Path = REPLAY_OUT, source: pathlib.Path = PACMAN_3D) -> pathlib.Path | None:
    """His real run through the SAME three.js page pacman_3d.py ships —
    extracted from the file's `_HTML` template at render time, never
    imported. None when the cubbyverse checkout is not on this machine."""
    try:
        src = source.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'_HTML = r"""(.*)"""', src, re.S)
    if not m:
        return None
    html = m.group(1)
    html = html.replace("3D Pac-Man — learned by the Mixture of World Models",
                        "3D Pac-Man — explored live by cubby-man")
    html = html.replace("agent learned 6 move operators, then VSA-planned the shortest "
                        "wall-avoiding path to every pellet",
                        "cubby-man learned this maze move by move — walls from VM refusals, "
                        "every fact in his own world model")
    html = html.replace("VSA planner pathing to each pellet…",
                        "exploring — every move VM-guarded…")
    data = {"w": env.w, "h": env.h, "d": env.d,
            "start": list(env.coords(env.start)),
            "walls": [list(w) for w in sorted(env.walls)],
            "pellets": [list(p) for p in env.pellets],
            "total": len(env.pellets), "traj": traj}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*DATA*/", json.dumps(data)), encoding="utf-8")
    return out
