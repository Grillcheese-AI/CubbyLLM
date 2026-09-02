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

_CV = pathlib.Path(r"C:\Users\grill\Documents\GitHub\cubbyverse")
PACMAN_3D = _CV / "examples" / "pacman_3d.py"
PACMAN_LIVE = _CV / "examples" / "pacman_live.py"        # the exact frontend is lifted from here
ASSETS_DIR = _CV / "examples" / "assets"                 # cubby's face textures
PLUTCHIK_JSON = _CV / "cubbyverse" / "core" / "emotion" / "plutchik.json"
REPLAY_OUT = pathlib.Path(r"C:/tmp/cubbyman_pacman_3d.html")

# the live game's hidden word per level and what each word NAMES (ostensive
# definition): collect its letters and the word is grounded -> cubby SAYS it
# when the percept recurs. Speech goes through CubbyTalk's ASK like all speech.
_WORDS = ["HELLO", "CUBBY", "MAZE", "LEARN", "GHOST", "POWER", "SMART"]
_WORD_CONCEPT = {"HELLO": "player_present", "CUBBY": "proud", "MAZE": "new_maze",
                 "LEARN": "discovery", "GHOST": "ghost_near", "POWER": "power_up",
                 "SMART": "solved"}
_CONCEPT_WORD = {v: k for k, v in _WORD_CONCEPT.items()}

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
JUMP_COST = 20                                           # our energy price for a hop (theirs is unrecorded here)
PROGRAMS_PATH = pathlib.Path(__file__).resolve().parent / "data" / "out" / "cubbyman_programs.json"

# ── generative superpowers: patterns over move SLOTS ────────────────────────
# A pattern is a word over slots A/B/C ("AAA" = one direction three times,
# "AB" = two perpendicular directions, "ABC" = three axes). At use time the
# slots are instantiated with any assignment of distinct, non-opposite
# directions, so one program covers a whole family of moves. The live game's
# DASH / BLINK / COMBO are three points in this space; the rest is his to
# find — combos that did not exist until he composed them.
_FLAVOR = {"AAA": "DASH", "AAAAA": "BLINK", "AB": "COMBO", "AAB": "KNIGHT", "ABC": "WARP",
           "AAAA": "SPRINT", "AABB": "ZIGZAG", "ABAB": "STAIRS", "ABA": "HOOK", "ABB": "ELBOW"}


def pattern_name(pattern: str) -> str:
    return _FLAVOR.get(pattern, f"COMBO-{pattern}")


def _assignments(pattern: str):
    """Every injective slot -> direction assignment with no two slots on
    opposite directions (a back-and-forth is not a move)."""
    slots = []
    for ch in pattern:
        if ch not in slots:
            slots.append(ch)
    dirs = list(MOVES)

    def rec(i, chosen):
        if i == len(slots):
            yield dict(zip(slots, chosen))
            return
        for d in dirs:
            if d in chosen or OPP[d] in chosen:
                continue
            yield from rec(i + 1, chosen + [d])
    yield from rec(0, [])


class ProgramLibrary:
    """His generated programs — each with its source, the REASONING that led
    to it (what triggered it, the situation he was in, how the proposal was
    sampled, the VM's verdict) and a log of every use and what it earned.
    Nothing is ever deleted: a pattern that never pays is RETIRED (kept in
    the notebook, no longer offered) — he does not forget what he learned.
    Kept on disk as JSON plus a readable Markdown notebook (pacman_live
    persists procedural memory the same way) so the knowledge survives a
    restart and can be read afterwards."""

    def __init__(self, path: pathlib.Path | None = None) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if path is not None:
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
            except (OSError, ValueError):
                self.entries = {}

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def names(self) -> list[str]:
        return [n for n, e in self.entries.items() if not e.get("retired")]

    def active(self) -> dict[str, dict]:
        return {n: e for n, e in self.entries.items() if not e.get("retired")}

    def add(self, name: str, pattern: str | None, program: str, kind: str, step: int,
            reasoning: dict) -> None:
        self.entries[name] = {"pattern": pattern, "kind": kind, "program": program,
                              "born": step, "legal": 0, "used": 0, "saved": 0,
                              "reasoning": reasoning, "uses": [], "retired": False}
        self.save()

    def note_legal(self, name: str) -> None:
        if name in self.entries:
            self.entries[name]["legal"] += 1

    def note_used(self, name: str, saved: int, context: dict) -> None:
        if name in self.entries:
            e = self.entries[name]
            e["used"] += 1
            e["saved"] += saved
            if len(e["uses"]) < 200:
                e["uses"].append({"saved": saved, **context})
            self.save()

    def value(self, name: str) -> float:
        e = self.entries[name]
        return e["saved"] / (e["used"] + 1) + 0.05 * min(e["legal"], 20)

    def retire(self, step: int, keep: int = 10) -> str | None:
        """Stop OFFERING the least valuable never-used pattern once the active
        set is crowded (every legal move is an ASK candidate; a useless one is
        noise). It stays in the notebook with the reason — not forgotten."""
        act = self.active()
        cands = [n for n, e in act.items()
                 if e["kind"] == "pattern" and e["used"] == 0 and step - e["born"] > 40]
        if len(act) > keep and cands:
            worst = min(cands, key=self.value)
            self.entries[worst]["retired"] = True
            self.entries[worst]["retired_reason"] = (f"never used in {step - self.entries[worst]['born']} "
                                                     f"steps; legal {self.entries[worst]['legal']} times")
            self.save()
            return worst
        return None

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"entries": self.entries}, indent=1), encoding="utf-8")
            self.path.with_suffix(".md").write_text(self.notebook(), encoding="utf-8")
        except OSError:
            pass

    def notebook(self) -> str:
        """The readable record: one section per program, reasoning first."""
        out = ["# cubby-man's programs\n",
               "Every move he composed himself: why, the situation, the program the VM certified, "
               "and what it earned. Retired programs are kept — nothing learned is forgotten.\n"]
        for name, e in self.entries.items():
            r = e.get("reasoning", {})
            out.append(f"\n## {name}  ({'pattern ' + e['pattern'] if e.get('pattern') else e['kind']})"
                       f"{'  — RETIRED: ' + e.get('retired_reason', '') if e.get('retired') else ''}\n")
            out.append(f"- **trigger:** {r.get('why', '?')} — {r.get('because', '')}\n")
            s = r.get("situation", {})
            if s:
                out.append("- **situation:** " + ", ".join(f"{k} {v}" for k, v in s.items()) + "\n")
            if r.get("rationale"):
                out.append(f"- **proposal:** {r['rationale']}\n")
            out.append(f"- **VM verdict:** {r.get('verdict', '?')}\n")
            if e["kind"] == "tool":
                out.append(f"- **task:** expected `{r.get('expected')}` · got `{r.get('got')}` · "
                           f"{'PASS' if r.get('ok') else 'FAIL'}\n")
            else:
                out.append(f"- **earned:** used {e['used']}× · saved {e['saved']} steps · legal {e['legal']}×\n")
            out.append("\n```cubelang\n" + e["program"].rstrip() + "\n```\n")
            if e.get("uses"):
                out.append("\n| step | level | move | landed on | saved |\n|---|---|---|---|---|\n")
                for u in e["uses"][-12:]:
                    out.append(f"| {u.get('step')} | {u.get('level')} | {u.get('move')} | "
                               f"{u.get('landed', '')} | {u.get('saved')} |\n")
        return "".join(out)


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
        self.energy = 100
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
        self._power_all = set(self.power)
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
        # the live game's time budget + the hidden word (letter pellets = the
        # first len(word) non-star pellets, exactly as pacman_live assigns them)
        self.budget = int(46 + 8 * level)
        self.attempt = 1
        self.steps = 0
        self.word = _WORDS[(level - 1) % len(_WORDS)]
        cells = [c for c in sorted(self.pellets) if c not in self.power][: len(self.word)]
        self.letter_at = {c: (i, self.word[i]) for i, c in enumerate(cells)}
        self.collected: dict[int, str] = {}

    def begin_run(self) -> None:
        """Out of time: same level, fresh run (pacman_live's _begin_run)."""
        self.remaining = set(self.pellets)
        self.power = set(self._power_all)
        self.score = 0
        self.steps = 0
        self.frightened = 0
        self.collected = {}
        self.ghosts = list(self.ghost_spawn)

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

    def power_moves(self, place: str, library, energy: int) -> dict[str, str]:
        """The superpower moves legal HERE from his program library: JUMP =
        a hop over ONE hazard (costs energy); a pattern = its slots
        instantiated with directions, every cell on the way open. Move names
        carry the instantiation (`knight_right_right_up`); `jump_*` keeps the
        live page's hop animation."""
        x, y, z = self.coords(place)
        out = {}
        for name, e in library.entries.items():
            if e["kind"] == "jump":
                if energy < JUMP_COST:
                    continue
                for d, (dx, dy, dz) in MOVES.items():
                    mid, land = (x + dx, y + dy, z + dz), (x + 2 * dx, y + 2 * dy, z + 2 * dz)
                    if _in(mid, self.w, self.h, self.d) and mid in self.hazards and self._open(land):
                        out[f"jump_{d}"] = self.cell(*land)
                continue
            pattern = e["pattern"]
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
        out = {"pellet": False, "power": False, "letter": None, "word_done": False}
        if c in self.remaining:
            self.remaining.discard(c)
            self.score += 1
            self.total_score += 1
            out["pellet"] = True
            if c in self.power:
                self.power.discard(c)
                self.frightened = FRIGHT_STEPS
                out["power"] = True
            if c in self.letter_at:                      # a LETTER pellet
                i, ch = self.letter_at[c]
                self.collected[i] = ch
                out["letter"] = ch
                out["word_done"] = len(self.collected) == len(self.word)
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
        self.begin_run()
        self.game_over = False


# our Lövheim readout -> the live game's Plutchik compass (petal, angle, petal
# color, the three intensity tiers whose color/message come from plutchik.json)
_PETAL = {"joy": ("joy", 0, "#ffca05", ("serenity", "joy", "ecstasy")),
          "warm": ("trust", 45, "#8ac650", ("acceptance", "trust", "admiration")),
          "anxious": ("fear", 90, "#00a551", ("apprehension", "fear", "terror")),
          "surprise": ("surprise", 135, "#0099cd", ("distraction", "surprise", "amazement")),
          "sad": ("sadness", 180, "#2983c5", ("pensiveness", "sadness", "grief")),
          "shame": ("sadness", 180, "#2983c5", ("pensiveness", "sadness", "grief")),
          "contempt": ("disgust", 225, "#8973b3", ("boredom", "disgust", "loathing")),
          "angry": ("anger", 270, "#f05b61", ("annoyance", "anger", "rage")),
          "curious": ("anticipation", 315, "#f6923d", ("interest", "anticipation", "vigilance"))}
_PLUTCHIK: dict | None = None


def _plutchik() -> dict:
    """cubbyverse's Plutchik metadata (tier color + felt message), loaded by
    file path — data, not a package import. {} when the checkout is absent."""
    global _PLUTCHIK
    if _PLUTCHIK is None:
        try:
            _PLUTCHIK = json.loads(PLUTCHIK_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _PLUTCHIK = {}
    return _PLUTCHIK


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

    def __init__(self, env: GhostVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35, memory: pathlib.Path | None = None) -> None:
        self.brain = None
        self.fear = 0.3                                  # pacman_live's ghost_penalty, learned
        self.vocab: list[str] = []                       # words grounded by collecting their letters
        self.talk: dict | None = None
        self._talk_id = 0
        self._cool: dict[str, int] = {}
        self._last: dict = {}
        self._pending_says: str | None = None
        self.sighted: set[str] = set()                   # cells where he SAW a pellet (facts too)
        self._eaten_run: set[str] = set()                # eaten THIS run (pellets respawn on a retry)
        self._plan_next: str | None = None               # the cell his map says to go to next
        self._graph_n = -1
        self._graph: dict[str, set[str]] = {}
        self.library = ProgramLibrary(memory)            # his generated programs (+ stats), persisted
        self._last_proposal = -99
        super().__init__(env or GhostVerse(), exe=exe, seed=seed, probe=probe)

    @property
    def powers(self) -> list[str]:
        return self.library.names()

    # ── superpowers: programs he GENERATES, the VM certifies, he keeps ──────
    _BECAUSE = {"stuck": "a pellet I have SEEN has no path on my map — something walls it off",
                "out_of_time": "the level's time budget ran out — I need to cover ground faster",
                "curious": "nothing to chase right now and I feel like inventing a move"}

    def _situation(self) -> dict:
        env = self.env
        near = min((_manh(env.coords(self.place), g) for g in env.ghosts), default=None)
        st = {"level": env.level, "attempt": env.attempt, "step": env.steps, "budget": env.budget,
              "pellets_left": len(env.remaining), "seen_unreached": len(self.sighted - self._eaten_run),
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
        if name in self.library:
            return None
        binds = "".join(f'        bind frame, H{i + 1}_STEP, "{s}";\n' for i, s in enumerate(steps))
        program = ("use vsa;\n\nprogram Superpower implements ISolve {\n"
                   "    public function solve(mention: str): str {\n"
                   "        create frame: number;\n"
                   f"{binds}"
                   f'        bind frame, H{len(steps) + 1}_NAME, "{name}";\n'
                   f"        return recover(frame, H{len(steps) + 1}_NAME);\n    }}\n}}\n")
        reasoning = {"why": why, "because": self._BECAUSE.get(why, why), "situation": self._situation(),
                     "rationale": rationale, "steps": steps}
        ok = self._certify_join(program, name)
        reasoning["verdict"] = ("certified: the VM bound the steps and recovered the name" if ok
                                else "REJECTED by the VM (did not execute or recover the name)")
        if not ok:
            self._t("program_rejected", name=name, why=why, program=program, reasoning=reasoning)
            return None
        self.library.add(name, pattern, program, kind, self.env.steps, reasoning)
        self._learn([f"{name} is the superpower {len(self.library.entries)} of cubbyman",
                     f"{' '.join(steps)} is the recipe of {name}"])
        self._t("program", name=name, why=why, pattern=pattern, program=program, reasoning=reasoning)
        if self.chem is not None:
            self.chem.update(novelty=1.0, valence=0.9)
        return name

    def _propose(self, why: str) -> str | None:
        """GENERATE a new composition. Lengths are sampled from what has paid
        off (steps saved per use of the patterns he has), with a prior toward
        3; out-of-time asks for longer ones. Slots are drawn from A/B/C. A
        pattern he already holds is skipped; dead weight is retired (not
        forgotten) first. The sampling rationale is written into the
        program's reasoning trace."""
        retired = self.library.retire(self.env.steps)
        if retired:
            self._t("retire", name=retired, reason=self.library.entries[retired]["retired_reason"])
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

    def candidate_moves(self, exits: dict[str, str]) -> dict[str, str]:
        """The ASK offers the base exits PLUS the superpower moves from his
        library that are legal here (and notes which programs were usable)."""
        if not self.library.entries:
            return exits
        power = self.env.power_moves(self.place, self.library, self.env.energy)
        for move in power:
            self.library.note_legal("JUMP" if move.startswith("jump_") else move.split("_")[0].upper())
        return {**exits, **power}

    # ── ghosts: a berth that GROWS with learned fear ────────────────────────
    @property
    def danger_radius(self) -> int:
        return 1 if self.fear < 1.0 else 2 if self.fear < 2.4 else 3

    def danger_cells(self) -> set[str]:
        """Cells within the fear radius of a hunting ghost (empty while they
        are frightened)."""
        env = self.env
        if env.frightened > 0:
            return set()
        r = self.danger_radius
        out = set()
        for g in env.ghosts:
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
        return decision_true(r["got"]) if r["ok"] else None

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

    # ── speech: grounded words through CubbyTalk ─────────────────────────────
    def _say(self, word: str, why: str) -> dict | None:
        if self.brain is None:
            return None
        try:
            rec = self.brain.chat.mediate(f"[{why}]", [f"{word}!"], rejected=[])
        except Exception as e:                           # speech must never stop the game
            self._t("talk_error", word=word, error=str(e)[:120])
            return None
        self._talk_id += 1
        self.talk = {"word": word, "why": why, "sim": 1.0, "id": self._talk_id,
                     "trace": f"ASK offered {rec['offered']} -> chose {rec['reply']!r} -> act remembered"}
        self._t("talk", word=word, why=why)
        return self.talk

    def _percept(self, why: str) -> None:
        """A percept a word may NAME: if he has grounded that word (collected
        its letters) and the cooldown passed, he says it."""
        word = _CONCEPT_WORD.get(why)
        if word in self.vocab and self.env.steps - self._cool.get(why, -99) >= 6:
            self._cool[why] = self.env.steps
            self._say(word, why)

    # ── the moves ────────────────────────────────────────────────────────────
    SIGHT = 2                                            # pellets glow: he sees them this far

    def _sight(self, place: str) -> list[str]:
        """Pellets within SIGHT (Manhattan) of where he stands become
        sighting facts — perception, learned like everything else."""
        x, y, z = self.env.coords(place)
        out = []
        for c in self.env.remaining:
            if _manh(c, (x, y, z)) <= self.SIGHT:
                cell = self.env.cell(*c)
                self.sighted.add(cell)
                out.append(f"{'a star' if c in self.env.power else 'a pellet'} is the sighting of {cell}")
        return out

    def on_arrive(self, place: str) -> int:
        self._last_eaten = None
        ate = self.env.eat(place)
        new = 0
        if ate["pellet"]:
            self._last_eaten = list(self.env.coords(place))
            self._eaten_run.add(place)
            fact = f"pellet {self.env.score} is the discovery of {place}"
            self._t("eat", place=place, pellet=self.env.score, power=ate["power"],
                    letter=ate["letter"], remaining=len(self.env.remaining))
            new += self._learn([fact])
            if self.chem is not None:
                self.chem.update(valence=0.8 if ate["power"] else 0.3)
            if ate["power"]:
                self._percept("power_up")
            if ate["word_done"]:                         # the hidden word is complete -> grounded + said
                word = self.env.word
                if word not in self.vocab:
                    self.vocab.append(word)
                self._pending_says = word
                self._say(word, _WORD_CONCEPT.get(word, "learned"))
        new += self._learn(self._sight(place))
        return new

    # ── planning over HIS map (the facts he learned), never the env ─────────
    def _known_graph(self) -> dict[str, set[str]]:
        if self._graph_n != len(self.world):
            g: dict[str, set[str]] = {}
            for f in self.world.texts:
                m = self._nbr_re.match(f)
                if m and m.group("b") not in ("a wall", "a hazard"):
                    g.setdefault(m.group("a"), set()).add(m.group("b"))
                    g.setdefault(m.group("b"), set())
            self._graph, self._graph_n = g, len(self.world)
        return self._graph

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

    def plan_next(self, avoid: set[str] = frozenset()) -> str | None:
        """The cell his MAP says to go to next: BFS across the neighbor facts
        he holds to the nearest pellet he has SEEN and not eaten this run,
        else to the nearest known cell he never stood on (the frontier).
        A discovered superpower move is taken when it lands where the rest
        of the way is at least two steps shorter — or when the base map has
        no path at all (a JUMP over the hazard that walls a pellet off).
        None when neither his map nor his powers reach anything."""
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
        """Fear-aware and map-driven: hunt frightened ghosts; never step onto
        a hunting one; FLEE when one is inside his fear radius (the exit that
        maximizes the distance to them); otherwise follow the path his map
        planned; only with no plan fall back to novelty."""
        env = self.env
        ghost_cells = {env.cell(*g) for g in env.ghosts}
        if env.frightened > 0:
            hunt = [m for m, p in exits.items() if p in ghost_cells]
            if hunt:
                return hunt[0]
        else:
            safe = {m: p for m, p in exits.items() if p not in ghost_cells}
            if safe:
                exits = safe
            if env.ghosts:
                here = env.coords(self.place)
                near = min(_manh(here, g) for g in env.ghosts)
                if near <= self.danger_radius:           # too close: run, then think
                    def gap(m):
                        return min(_manh(env.coords(exits[m]), g) for g in env.ghosts)
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
        self._t("level_up", cleared=self.env.level, next=nxt, total_score=self.env.total_score)
        self.env._start_level(nxt)
        self.place = self.env.start
        self.sighted.clear()
        self._eaten_run.clear()
        self._seed_basics()
        self._learn(self.env.observe(self.place))
        self._learn(self._sight(self.place))
        self.visits[self.place] = self.visits.get(self.place, 0) + 1
        self._percept("new_maze")
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
            if not env.remaining:                        # headless driver: nobody called next_level
                self.next_level()
            env.steps += 1
            if env.steps > env.budget:                   # OUT OF TIME -> fail, learn a speedup, redo
                env.attempt += 1
                env.begin_run()
                self.place = env.start
                self._eaten_run.clear()                  # the pellets are back; his sightings still hold
                ev["failed"] = True
                ev["learned"] = self._propose("out_of_time")     # too slow -> generate a faster move
                self._t("out_of_time", level=env.level, attempt=env.attempt, learned=ev["learned"])
                self._last = ev
                return {"from": None, "place": self.place, "chosen": None, "new": 0, "probed": None}
            self._pending_says = None
            danger = self.danger_cells()                 # the berth grows with learned fear
            self._plan_next = self.plan_next(avoid=danger)
            seen_left = bool(self.sighted - self._eaten_run)
            if self._plan_next is None and seen_left and "JUMP" not in self.library:
                # a pellet he has SEEN with no path on his map: hazard-walled -> compose a JUMP
                ev["learned"] = self._compose("JUMP", "stuck", ["hop", "hop"], None, "jump",
                                              "two hops in one direction clear the hazard in between")
                self._plan_next = self.plan_next(avoid=danger)
            elif (not seen_left and self.chem is not None and self.chem.dopamine > 0.5
                  and env.steps - self._last_proposal >= 12):
                ev["learned"] = self._propose("curious")         # nothing to chase: invent a move
            if self._plan_next is not None:
                self._t("plan", to=self._plan_next,
                        goal="pellet" if seen_left else "frontier")
            rec = super().step()                         # his move + eating + learning + traj
            chosen = rec.get("chosen") or ""
            if chosen.startswith("jump_"):
                env.energy = max(0, env.energy - JUMP_COST)
            else:
                env.energy = min(100, env.energy + 2)    # simplified regen (theirs: +1/pellet, +4 resting)
            if "_" in chosen:                            # only superpower moves carry an underscore
                pname = "JUMP" if chosen.startswith("jump_") else chosen.split("_")[0].upper()
                e = self.library.entries.get(pname)
                saved = (len(e["pattern"]) - 1) if e and e.get("pattern") else 1
                self.library.note_used(pname, saved, {"step": env.steps, "level": env.level, "move": chosen,
                                                      "landed": ("pellet" if self._last_eaten else "empty")})
                self._t("superpower_move", move=chosen, program=pname, saved_steps=saved,
                        to=self.place, energy=env.energy)
            ev["eaten"] = self._last_eaten
            ev["says"] = self._pending_says
            gh = env.ghost_turn(self.place)              # then the ghosts move
            if gh["eaten"]:
                ev["ate_ghost"] = True
                self._t("ghost_eaten", n=gh["eaten"], bonus=GHOST_BONUS * gh["eaten"])
                if self.chem is not None:
                    self.chem.update(valence=1.0)
            if gh["caught"]:
                self.fear = min(4.0, self.fear + 0.7)    # LEARN: that hurt -> fear ghosts more
                fact = f"a ghost is the danger of {self.place}"
                self._learn([fact])
                ev["caught"] = "gameover" if env.game_over else True
                self._t("caught", place=self.place, lives=env.lives, fear=round(self.fear, 2),
                        game_over=env.game_over, learned=fact)
                self.place = env.start
                if self.chem is not None:
                    self.chem.update(threat=1.0, valence=-0.8)
            elif env.ghosts:
                near = min(_manh(env.coords(self.place), g) for g in env.ghosts)
                if self.chem is not None:
                    threat = 0.0 if env.frightened else (1.0 if near <= 1 else 0.5 if near <= 2 else 0.0)
                    if threat:
                        self.chem.update(threat=threat)
                if near <= 2 and not env.frightened:
                    self._percept("ghost_near")
            if rec["new"] >= 3:
                self._percept("discovery")
            if not env.remaining:
                ev["beaten"] = True
                self._percept("solved")
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
        if name not in _PETAL:
            ar, val = chem.affect_arousal, chem.valence
            ne, da, ot = chem.noradrenaline - 0.15, chem.dopamine - 0.30, chem.oxytocin - 0.20
            if ar >= 0.5 and val <= 0 and ne > 0.05:
                name = "anxious"
            elif ar >= 0.5 and val > 0:
                name = "curious"
            elif ot > 0.10:
                name = "warm"
            elif da > 0.10 and val >= 0:
                name = "joy"
            elif da < -0.05 or val < -0.3:
                name = "sad"
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
        near = min((_manh(env.coords(self.place), g) for g in env.ghosts), default=99)
        scared = bool(env.ghosts) and env.frightened == 0 and near <= 2
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
                "word": env.word,
                "collected": "".join(env.collected.get(i, "_") for i in range(len(env.word))),
                "says": ev.get("says"), "thought": None, "vocab": list(self.vocab), "talk": self.talk,
                "lay_low": len(self.walls),              # relabeled: refused moves (walls learned)
                "pursuit": sum(1 for g in env.ghosts if _manh(env.coords(self.place), g) <= 2)}

    def resp(self) -> dict:
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
                **self.affect()}

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


# the lifted page's strings we change: endpoint prefixes (so it can live under
# /pac next to the console), the subtitle/foot (say what actually drives him),
# and the two HUD rows we back with different real numbers than the original
_FRONTEND_PATCHES = [
    ("fetch('/state')", "fetch('/pac/state')"),
    ("fetch('/init')", "fetch('/pac/init')"),
    ("fetch('/next')", "fetch('/pac/next')"),
    ("'/assets/", "'/pac/assets/"),
    ('<div id="sub">live MoWM + VSA planner · CVL value</div>',
     '<div id="sub">live CubbyLLM stand-in brain · every move VM-guarded · learns as he explores</div>'),
    ("isometric · drag to orbit · CVL picks the target (evades ghosts) · BFS plans + discovers JUMP · "
     "mSA ghosts flank · fear is learned",
     "isometric · drag to orbit · explorer brain: the VM offers the exits and guards the choice · "
     "walls learned from refused moves · joins are his own programs · fear is learned"),
    ('R-STDP learned: <b id="wd">dopa +0</b> · <b id="wc">cort 0</b> · <b id="wt">treat 0</b>',
     'world model: <b id="wd">facts 0</b> · <b id="wc">derived 0</b> · <b id="wt">walls 0</b>'),
    ("$('wd').textContent='dopa '+sg(s.w_dopa); $('wc').textContent='cort '+sg(s.w_cort); "
     "$('wt').textContent='treat '+sg(s.w_treat);",
     "$('wd').textContent='facts '+s.w_dopa; $('wc').textContent='derived '+s.w_cort; "
     "$('wt').textContent='walls '+s.w_treat;"),
    ('covers tracks (learned): <b id="laylow">0.3</b>', 'refused moves (walls learned): <b id="laylow">0</b>'),
    ("$('laylow').textContent=s.lay_low.toFixed(2)", "$('laylow').textContent=String(s.lay_low)"),
    ("INTERVAL=320", "INTERVAL=600"),                   # our VM-guarded steps are slower than the planner's
]


def load_frontend(source: pathlib.Path = PACMAN_LIVE) -> tuple[str | None, list[str]]:
    """pacman_live.py's EXACT page, extracted at serve time (never imported),
    with the patches above applied. -> (html, patches that did not apply)."""
    try:
        src = source.read_text(encoding="utf-8")
    except OSError:
        return None, ["source missing"]
    m = re.search(r'FRONTEND = r"""(.*?)"""\s*\n', src, re.S)
    if not m:
        return None, ["FRONTEND block not found"]
    html, missed = m.group(1), []
    for old, new in _FRONTEND_PATCHES:
        if old in html:
            html = html.replace(old, new)
        else:
            missed.append(old[:40])
    return html, missed


def load_asset(rel: str, base: pathlib.Path = ASSETS_DIR) -> bytes | None:
    """A face texture from cubbyverse's assets dir (png only, path-safe)."""
    try:
        fp = (base / rel.split("?")[0]).resolve()
        if fp.suffix.lower() == ".png" and base.resolve() in fp.parents and fp.is_file():
            return fp.read_bytes()
    except OSError:
        pass
    return None


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
                if self.man.env.remaining:               # a cleared level waits for /next
                    self.man.step()
                    if len(self.man.traj) % self.derive_every == 0:
                        self.man.derive_counts()
                r = self.man.resp()
            except Exception as e:                       # a dead VM must not kill the server
                self.error = str(e)[:200]
                self.man._t("live_error", error=self.error)
                r = {**(self._last_resp or self.man.resp()), "error": self.error}
            self._last = now
            self._last_resp = r
            return r

    def next_level(self) -> dict:
        with self._lock:
            if not self.man.env.remaining:
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
