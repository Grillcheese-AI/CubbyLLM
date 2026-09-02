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
        return {"w": env.w, "h": env.h, "d": env.d, "start": list(env.coords(env.start)),
                "walls": [list(w) for w in sorted(env.walls)],
                "pellets": [list(p) for p in sorted(env.remaining)],
                "pos": list(env.coords(man.place)),
                "move": man.traj[-1]["move"] if man.traj else None,
                "score": env.score, "total": len(env.pellets), "beaten": man.beaten,
                "steps": len(man.traj), "facts": len(man.world),
                "walls_learned": len(man.walls), "derived": len(man.derived),
                "emotion": man.chem.dominant_emotion if man.chem is not None else "neutral",
                "stepped": stepped, "error": self._error}


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
