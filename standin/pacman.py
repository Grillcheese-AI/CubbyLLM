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
                 probe: float = 0.35) -> None:
        self.traj: list[dict] = []
        self._last_eaten = None
        super().__init__(env or PacVerse(), exe=exe, seed=seed, probe=probe)

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


# ── the big game: ghosts, hazards, power stars, lives, levels ───────────────
FRIGHT_STEPS = 14                                        # pacman_live.py's constants
GHOST_BONUS = 5
GHOST_COLORS = ["#ff4d5e", "#27d3ff", "#ff9ff3", "#36e07a"]
JUMP_COST = 20                                           # our energy price for a hop (theirs is unrecorded here)
# the energy economy (owner, 2026-09-02: every move costs, combos cost more,
# and energy only comes back by RESTING in a safe place — or a little from pellets)
MOVE_COST = 1                                            # a base step
PELLET_ENERGY, STAR_ENERGY = 4, 12                       # what eating gives back
REST_GAIN = 8                                            # per step of rest, in a safe spot
REST_BELOW, REST_UNTIL = 30, 60                          # start resting under 30, stop at 60 (hysteresis)
REST = "rest"                                            # the ASK candidate: stay put
TRAP_BONUS = 3                                           # a ghost walked into his trap
MINE = "mine"                                            # the ASK candidate: drop a trap here, then move


def pattern_cost(pattern: str | None) -> int:
    """A combo costs twice its length (AAB -> 6, AAAAA -> 10); a jump 20."""
    return JUMP_COST if pattern is None else 2 * len(pattern)
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

    def __init__(self, path: pathlib.Path | None = None, ledger=None) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.ledger = ledger                             # ledger.Ledger: decisions hashed + signed in SQLite (2026-09-04)
        if path is not None:
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
            except (OSError, ValueError):
                self.entries = {}
        if self.ledger is not None:
            self.audit()

    def audit(self) -> dict[str, str]:
        """Every entry that carries a certificate hash is checked against the ledger: the program text must hash to a
        signed, certified row made by the current VM build — otherwise the entry is RETIRED (never deleted) with the
        reason. Entries without a certificate (pre-ledger) are left as they are, flagged `uncertified`. -> {name: reason}."""
        out = {}
        for name, e in self.entries.items():
            cert = e.get("cert")
            if not cert:
                e["uncertified"] = True
                continue
            ok, why = self.ledger.verify(cert, e.get("program"))
            if not ok and not e.get("retired"):
                e["retired"] = True
                e["retired_reason"] = f"certificate: {why}"
                out[name] = why
            elif ok:
                e.pop("uncertified", None)
        if out:
            self.save()
        return out

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def names(self) -> list[str]:
        return [n for n, e in self.entries.items() if not e.get("retired")]

    def active(self) -> dict[str, dict]:
        return {n: e for n, e in self.entries.items() if not e.get("retired")}

    def add(self, name: str, pattern: str | None, program: str, kind: str, step: int,
            reasoning: dict, cert: str | None = None) -> None:
        self.entries[name] = {"pattern": pattern, "kind": kind, "program": program,
                              "born": step, "legal": 0, "used": 0, "saved": 0,
                              "reasoning": reasoning, "uses": [], "retired": False,
                              "cert": cert}                # the ledger's decision hash (None: pre-ledger / no VM run)
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

    # ── ONE program, a reusable function per combo ─────────────────────────
    @staticmethod
    def fn_name(name: str) -> str:
        """A move's function in the Moves program: `COMBO-AAB` -> `combo_aab`."""
        return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_") or "move"

    def moves_program(self, extra: dict | None = None) -> str:
        """The one program he keeps editing: `program Moves implements ISolve`
        with a function per ACTIVE move (jump + patterns) — `extra` = {name:
        steps} adds the candidate function being certified. solve() is the
        catalogue smoke: it recovers how many moves the program holds."""
        fns = {}
        for name, e in self.active().items():
            if e["kind"] == "jump":
                fns[name] = ["hop", "hop"]
            elif e["kind"] == "pattern" and e.get("pattern"):
                fns[name] = list(e["pattern"])
        if extra:
            fns.update(extra)
        out = ["use vsa;\n\n# cubby-man's moves: one program, a reusable function per combo (edited in play)\n"
               "program Moves implements ISolve {\n"
               "    public function solve(mention: str): str {\n        create frame: number;\n"
               f'        bind frame, H1_MOVES, "{len(fns)}";\n        return recover(frame, H1_MOVES);\n    }}\n']
        for name, steps in fns.items():
            binds = "".join(f'        bind frame, H{i + 1}_STEP, "{s}";\n' for i, s in enumerate(steps))
            out.append(f"\n    public function {self.fn_name(name)}(): str {{\n        create frame: number;\n"
                       f"{binds}        bind frame, H{len(steps) + 1}_NAME, \"{name}\";\n"
                       f"        return recover(frame, H{len(steps) + 1}_NAME);\n    }}\n")
        out.append("}\n")
        return "".join(out)

    def consolidate(self, level: int, keep: int = 6) -> list[str]:
        """After a level: keep the `keep` most valuable ACTIVE patterns, retire
        the rest with the reason (never deleted). JUMP is structural and
        always stays. Returns the names retired."""
        act = {n: e for n, e in self.active().items() if e["kind"] == "pattern"}
        if len(act) <= keep:
            return []
        ranked = sorted(act, key=self.value, reverse=True)
        out = []
        for n in ranked[keep:]:
            e = self.entries[n]
            e["retired"] = True
            e["retired_reason"] = (f"consolidated after level {level}: value {self.value(n):.2f} below the "
                                   f"top {keep} (used {e['used']}x, saved {e['saved']}, legal {e['legal']}x)")
            out.append(n)
        if out:
            self.save()
        return out

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"entries": self.entries}, indent=1), encoding="utf-8")
            self.path.with_suffix(".md").write_text(self.notebook(), encoding="utf-8")
            self.path.with_name("cubbyman_moves.cube").write_text(self.moves_program(), encoding="utf-8")
        except OSError:
            pass

    def notebook(self) -> str:
        """The readable record: the ONE Moves program as it stands, then one
        section per move/tool with its reasoning (each stores the program as
        it was when certified, so the evolution is readable)."""
        out = ["# cubby-man's programs\n",
               "Every move he composed himself: why, the situation, the program the VM certified, "
               "and what it earned. Retired moves are kept — nothing learned is forgotten.\n",
               "\n## The Moves program (as it stands — a reusable function per active combo)\n",
               "\n```cubelang\n" + self.moves_program().rstrip() + "\n```\n"]
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

    # levels 1..N run with no ghosts: his classroom. Env-overridable
    # (CUBBYMAN_GHOST_FREE_LEVELS) so a test or a demo can turn the threat
    # back on at level 1 without editing the rule into the code.
    GHOST_FREE_LEVELS = 3

    def __init__(self, level: int = 1, seed_extra: int = 0,
                 ghost_free_levels: int | None = None) -> None:
        self.total_score = 0
        self.lives = 3
        self.energy = 100
        self.mines_left = 0                              # traps: +1 per level, unused ones carry over
        self.mines: set = set()                          # cells holding a trap (ghosts only)
        self.seed_extra = int(seed_extra)
        self.ghost_free_levels = int(
            ghost_free_levels if ghost_free_levels is not None
            else os.environ.get("CUBBYMAN_GHOST_FREE_LEVELS", self.GHOST_FREE_LEVELS))
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
        return seen

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

# Lövheim's SOCIAL corners. Contempt/disgust and shame/humiliation need
# someone to feel them ABOUT; a maze has nobody in it, so the corner fires on
# hormone geometry alone and the compass reports an emotion he has no reason
# to have. Nick, 2026-09-15, on a "(sick of it)" two steps into a fresh level:
# *"sick of it should be neutral."* They stay in `_PETAL` because a world with
# social input can read them; here they read calm.
_SOCIAL_CORNERS = frozenset({"contempt", "shame"})

# The THINGS this world contains. He may name one only when the step's record
# or the facts he learned in it mention it — exp_r37 caught him saying "without
# triggering the ghost" on a ghost-free level, which passed every other check
# because "ghost" is neither a figure nor a cell name. Invented entities are
# the same defect as invented numbers; this is the clause that says so.
_ENTITIES = {"ghost": ("ghost", "ghosts"), "pellet": ("pellet", "pellets"),
             "star": ("star", "stars"), "wall": ("wall", "walls"),
             "hazard": ("hazard", "hazards"), "trap": ("trap", "traps"),
             "edge": ("edge",), "level": ("level",)}
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


class _Safe(dict):
    """format_map helper: a missing field renders as ? instead of raising."""

    def __missing__(self, k):
        return "?"


_MOOD_TAG = re.compile(r"^\(([^)]{1,40})\)\s*")
_ECHO = re.compile(r"own words|first person|short sentence|keeping every|propres mots|premi\u00e8re personne|phrase courte|"
                   r"\breword|\brephras|here'?s the sentence|\bsure\b|\bcertainly\b|\bi understand you\b|\bi see, so\b|"
                   r"\byou'?re\b|\byou are\b|\byour\b|\byou\b|\bvous\b|\btu\b|\bton\b|\bta\b|\btes\b", re.I)


def split_mood(line: str) -> tuple[str, str]:
    """'(at ease) Pellet 6/12, nice.' -> ('(at ease) ', 'Pellet 6/12, nice.'); no tag -> ('', line)."""
    m = _MOOD_TAG.match(line)
    return (line[:m.end()], line[m.end():]) if m else ("", line)


def _content_words(text: str) -> set[str]:
    """Stems (first five letters) of the words that carry content (>= 5 letters)."""
    return {w[:5] for w in re.findall(r"[a-z\u00e0-\u00ff]{5,}", text.lower())}


_NAME_RE = re.compile(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b")

# the shapes of a COPIED prompt rather than a spoken sentence: the record's
# own header, its field markers, a bullet, a markdown heading
_COPY = re.compile(r"here is what i just perceived|voici ce que je viens de perceptoir|"
                   r"voici ce que je viens de percevoir|^\s*#|\n\s*[-*]\s|"
                   r"\b(about|i am at|how i feel|pellets i can see|goal|tried|hit)\s*:", re.I | re.M)


# "level-1 cell 0-2-0" is 4 tokens and he must be able to say it; beyond that
# he is reciting. Measured on exp_r37's first run, where the copies ran 20+.
MAX_RUN = 6


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9à-ÿ\-]+", s.lower())


def longest_run(a: str, b: str) -> int:
    """Longest run of consecutive tokens `a` shares with `b`, in order. A
    sentence in his own words reuses the names; it does not reproduce the
    record's word ORDER for long stretches."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0
    best = 0
    prev = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def grounded_ok(record_line: str, text: str, facts, learned: str = "", slack: int = 5) -> bool:
    """Is `text` something he could honestly say about this step?

    The referent is the PERCEPT RECORD, not a host-written sentence, so the
    test is factual rather than lexical (2026-09-15):

      * every number in the sentence is in the record, AND no number in the
        sentence is absent from it — the old check was one-directional and let
        an invented figure through as long as the host's own numbers survived;
      * same both ways for cell and move NAMES;
      * content words come from the record or from the facts he learned this
        step, plus room for the words a sentence needs to BE a sentence. The
        exact figures and names carry the anti-invention work; this clause
        only stops a reply that is mostly words from nowhere, so the budget
        is proportional (a fluent short sentence really does add about five);
      * it is a SENTENCE, not the record read back. exp_r37's first run
        scored 100% "spoke" while the model was reproducing the prompt
        verbatim — a copy passes every grounding test there is, because
        everything in it came from the record. So: no header, no `key:`
        field markers, no bullets, and no run of `MAX_RUN` consecutive
        tokens shared with the record in order.
      * the voice rules, no echo of the instruction, no second person, no
        question back, no base-model guard, not a bio.

    A sentence that fails is refused, not repaired: the host then states the
    record flatly. A refusal is a result; a made-up thought is a defect."""
    from identity import has_non_latin, is_identity_reply, is_model_guard, voice_ok
    from forge import numbers
    if not text or "?" in text or _ECHO.search(text) or has_non_latin(text):
        return False
    if _COPY.search(text):
        return False                                     # the record read back, not a thought
    n_words = len(text.split())
    if n_words > 40 or n_words < 2:
        return False
    if longest_run(text, record_line) > MAX_RUN:
        return False                                     # his own words, not the record's word order
    if set(numbers(text)) != set(numbers(text)) & set(numbers(record_line)):
        return False                                     # a figure the step did not contain
    if set(_NAME_RE.findall(text)) - set(_NAME_RE.findall(record_line)):
        return False                                     # a cell or move he did not meet
    ground = f"{record_line} {learned}".lower()          # everything he perceived this step, verbatim
    mine = _content_words(text)
    if len(mine - _content_words(ground)) > max(slack, int(0.7 * len(mine))):
        return False                                     # mostly words for things he has not perceived
    said = text.lower()
    for thing, forms in _ENTITIES.items():
        if any(re.search(rf"\b{f}\b", said) for f in forms) and thing not in ground:
            return False                                 # a thing this step did not contain
    return voice_ok(text, facts) and not is_model_guard(text) and not is_identity_reply(text, facts)


def rephrase_ok(line: str, text: str, facts) -> bool:
    """Is `text` an acceptable rephrasing of the host's `line` (mood tag already
    stripped)? Every number and name kept, the voice rules, no base-model guard,
    not a bio, at most 40 words and under 2x the host line, no echo of the
    instruction, no second person, no question back, and at most ONE content
    word the host line did not have — the live trace's failures were invented
    meaning around kept numbers ('6 grains weighing 12 grams', '7 fingers')."""
    from forge import numbers
    from identity import has_non_latin, is_identity_reply, is_model_guard, voice_ok
    if not text or "?" in text or _ECHO.search(text) or has_non_latin(text):   # "建筑师: 8/15, nice." (Qwen arm, 2026-09-04)
        return False
    n_words = len(text.split())
    if n_words > 40 or n_words > max(12, 2 * len(line.split()) + 4):
        return False
    names = re.findall(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b", line)   # cells and move NAMES, not sentence-initial words
    if not (set(numbers(line)) <= set(numbers(text)) and all(n in text for n in names)):
        return False
    if len(_content_words(text) - _content_words(line)) > 1:
        return False
    return voice_ok(text, facts) and not is_model_guard(text) and not is_identity_reply(text, facts)


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
                 probe: float = 0.35, memory: pathlib.Path | None = None, ledger=None) -> None:
        self.brain = None
        self.fear = 0.3                                  # pacman_live's ghost_penalty, learned
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
        self.caught_at: list[int] = []                   # how far off a ghost was when he last CHOSE, before it caught him
        self._threat_seen_at: int | None = None
        self._eaten_run: set[str] = set()                # eaten THIS run (pellets respawn on a retry)
        self._plan_next: str | None = None               # the cell his map says to go to next
        self._graph_n = -1
        self._graph: dict[str, set[str]] = {}
        self.ledger = ledger                             # ledger.Ledger: every VM decision hashed + signed (None in tests)
        self.library = ProgramLibrary(memory, ledger=ledger)   # his generated programs (+ stats), persisted; audited on load
        self._last_proposal = -99
        super().__init__(env or GhostVerse(), exe=exe, seed=seed, probe=probe)

    @property
    def powers(self) -> list[str]:
        """His MOVES (jump + patterns), not his forged tools — the HUD's list."""
        return [n for n, e in self.library.active().items() if e["kind"] in ("jump", "pattern")]

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

    def _mutate(self, why: str) -> str | None:
        """MODIFY an existing program instead of composing a fresh one: pick an
        active pattern (the most valuable, or one that was legal but never
        paid), apply ONE edit — append a slot, drop a slot, or swap one — and
        compose the child with its lineage (parent, edit) in the reasoning.
        The parent stays (retire/consolidate decide its fate)."""
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
        """Improve what he has before inventing: MODIFY an existing program
        (one edit, lineage recorded); only with nothing to edit — or when the
        active set is at its cap — sample a fresh composition. Lengths are
        sampled from what has paid off; out-of-time asks for longer ones. The
        rationale is written into the program's reasoning trace."""
        retired = self.library.retire(self.env.steps)
        if retired:
            self._t("retire", name=retired, reason=self.library.entries[retired]["retired_reason"])
        n_active = sum(1 for e in self.library.active().values() if e["kind"] == "pattern")
        if n_active >= self.MAX_ACTIVE_PATTERNS:          # full: consolidate now, then edit rather than add
            for n in self.library.consolidate(self.env.level, keep=self.MAX_ACTIVE_PATTERNS - 2):
                self._t("retire", name=n, reason=self.library.entries[n]["retired_reason"])
        made = self._mutate(why)
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
        if not self.library.entries:
            return exits
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

    CAUGHT_FEAR = 0.7                                    # the live game's ghost_penalty increment
    CAUGHT_SHOCK_FRAMES = 6                              # frames of full threat: NE surges, cortisol integrates

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
        lesson = self.danger_radius + 1 if seen is None else max(seen, self.danger_radius)
        self.caught_at.append(max(1, min(self.MAX_DANGER_RADIUS, lesson)))
        self.fear = min(4.0, self.fear + self.CAUGHT_FEAR)
        if self.chem is not None:
            for _ in range(self.CAUGHT_SHOCK_FRAMES):
                self.chem.update(threat=1.0, valence=-0.8)
            # the aversive outcome: a dopamine DIP (reward-prediction error), which the
            # port has no input for -- without it the NE->DA coupling lifts dopamine and
            # Lövheim's low-5HT/high-DA/high-NE corner reads as RAGE instead of fear
            self.chem.dopamine = max(0.15, self.chem.dopamine - 0.30)
            self.chem.dominant_emotion = self.chem._classify_emotion(0.0)   # the corner label is set inside update()

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
    _PRIO = {"caught": 6, "level_up": 6, "out_of_time": 5, "program": 5, "modify": 5, "power": 4,
             "forge": 4, "flee": 4, "trapped": 5, "mine": 4, "superpower_move": 3, "eat": 3, "rest": 3,
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
              "mine": "dropped a trap", "trapped": "a ghost hit my trap"}

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
                   "left", "n", "seen_at")

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
        if self.chem is not None:
            rec["how i feel"] = self.emotion()["name"]
        return rec

    @staticmethod
    def render_percepts(rec: dict) -> str:
        """The record as a flat line — the fallback when the model's sentence
        is refused, and the referent the guard checks against. Generated from
        the data every time, so unlike a template it cannot say anything the
        step did not contain."""
        return "; ".join(f"{k}: {v}" for k, v in rec.items() if k != "about") or str(rec.get("about", ""))

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
        if self.lang == "fr":
            return (f"Situation : {situation}.\n"
                    "Que penses-tu ? Réponds en une phrase courte, à la première personne, "
                    "sans répéter la liste. N'invente rien que tu ne perçois pas.")
        return (f"Situation: {situation}.\n"
                "What are you thinking? Answer in one short sentence, first person, "
                "without repeating the list back. Do not invent anything you do not perceive.")

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
        flat = self.render_percepts(rec)
        if self.brain is None or not self.verbalize:
            return flat, False
        from identity import identity_system
        # what he may speak about: the record, plus the facts he learned this
        # step. Passed verbatim so the guard can check whole words ("ghost")
        # as well as stems.
        learned = " ".join(self._learned_here)
        try:
            raw = self.brain.emitter.emit(
                self._speak_prompt(rec), context="talk", max_new_tokens=48,
                system=identity_system(self.brain.facts, self.brain.chat.state),
                temperature=0.85, seed=self.env.steps * 7919 + self.env.level)
        except Exception as e:
            # traced, not swallowed: a broken emitter used to look exactly
            # like a refused sentence, which cost an afternoon
            self._t("thought_error", error=f"{type(e).__name__}: {e}"[:160])
            return flat, False
        from emitter import clean_reply
        text = " ".join(clean_reply(raw).split())
        if grounded_ok(flat, text, self.brain.facts, learned):
            return text, True
        self._t("thought_refused", said=text[:160], record=flat[:200])
        return flat, False

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
        flat = self.render_percepts(rec)
        text, spoke = self._speak(rec) if prio >= self.SPEAK_ABOVE else (flat, False)
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
            self.sighted.add(cell)
            out.append(f"{'a star' if c in seen['stars'] else 'a pellet'} is the sighting of {cell}")
        for g in seen["ghosts"]:
            if _manh(g, seen["at"]) <= self.HEARING:
                self.ghost_belief[tuple(g)] = env.steps
        self._forget_stale()
        return out

    def _forget_stale(self) -> None:
        """A sighting he has not refreshed in GHOST_MEMORY steps is no longer
        evidence. Dropping it is what keeps the belief a BELIEF."""
        now = self.env.steps
        for g in [g for g, t in self.ghost_belief.items() if now - t > self.GHOST_MEMORY]:
            self.ghost_belief.pop(g, None)

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
                self.chem.update(valence=0.8 if ate["power"] else 0.3)
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
        gh = self.believed_ghosts()                      # his belief, not env.ghosts
        ghost_cells = {env.cell(*g) for g in gh}
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
        self._t("level_up", cleared=self.env.level, next=nxt, total_score=self.env.total_score)
        for n in self.library.consolidate(self.env.level, keep=self.MAX_ACTIVE_PATTERNS - 2):   # sleep on it
            self._t("retire", name=n, reason=self.library.entries[n]["retired_reason"])
        self.env._start_level(nxt)
        self.env.energy = 100                            # a cleared level is a night's rest
        self._resting = False
        self.place = self.env.start
        self.sighted.clear()
        self._eaten_run.clear()
        self.ghost_belief.clear()                        # a new maze: nothing he believed is evidence any more
        self._seed_basics()
        self._learn(self.look_around(self.place))
        self._learn(self._sight(self.place))
        self.visits[self.place] = self.visits.get(self.place, 0) + 1
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
                ev["learned"] = self._propose("out_of_time")     # too slow -> generate a faster move
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
            gh = env.ghost_turn(self.place)              # then the ghosts move
            if gh.get("trapped"):
                ev["trapped"] = gh["trapped"]
                self._t("trapped", n=gh["trapped"], bonus=TRAP_BONUS * gh["trapped"], total_score=env.total_score)
                self._think("trapped", n=gh["trapped"])
                if self.chem is not None:
                    self.chem.update(valence=0.9, novelty=0.3)
            if gh["eaten"]:
                ev["ate_ghost"] = True
                self._t("ghost_eaten", n=gh["eaten"], bonus=GHOST_BONUS * gh["eaten"])
                if self.chem is not None:
                    self.chem.update(valence=1.0)
            if gh["caught"]:
                fact = f"a ghost is the danger of {self.place}"
                self._learn([fact])
                last_seen = self._threat_seen_at         # where it was when he last chose
                self._on_caught()                        # fear + the shock + THE LESSON (the berth he keeps)
                ev["caught"] = "gameover" if env.game_over else True
                self._t("caught", place=self.place, lives=env.lives, fear=round(self.fear, 2),
                        game_over=env.game_over, learned=fact,
                        seen_at=last_seen, radius=self.danger_radius)
                self._think("caught", place=self.place)
                self.place = env.start
                self.ghost_belief.clear()                # respawned across the maze: the belief is void
            else:
                believed = self.believed_ghosts()
                if believed and self.chem is not None:
                    near = min(_manh(env.coords(self.place), g) for g in believed)
                    # threat scales over the berth HE learned rather than the
                    # old hardcoded `1.0 if near<=1 else 0.5 if near<=2`
                    r = self.danger_radius
                    threat = 0.0 if env.frightened else max(0.0, 1.0 - max(0, near - 1) / max(1, r))
                    if threat:
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
                               if _manh(env.coords(self.place), g) <= self.danger_radius)}

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


# the lifted page's strings we change: endpoint prefixes (so it can live under
# /pac next to the console), the subtitle/foot (say what actually drives him),
# and the two HUD rows we back with different real numbers than the original
_FRONTEND_PATCHES = [
    ("fetch('/state')", "fetch('/pac/state')"),
    ('function build(D){', 'function build(D){ if(window.cbLevelStart) cbLevelStart(D);'),   # the level-start jingle
    ("  if(s.lives!=null) $('lives').innerHTML=",
     "  if(window.cbSfx) cbSfx(s);\n  if(s.lives!=null) $('lives').innerHTML="),   # the sound layer reads every frame
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
    # the letter/word mechanic is gone: the speech line shows his THOUGHT, the flash says so
    ("if(s.word){ const prog=(s.collected||'').split('').join(' '); $('speaktxt').textContent = s.says ? "
     "('💬 says: \"'+s.says+'\"') : ('letters: '+(prog||'—')); }",
     "$('speaktxt').textContent = s.thought ? ('💭 '+s.thought) : '';"),
    ("flash('💬 CUBBY SAYS: \"'+s.says+'\"','#7fe0ff')", "flash('💭 '+s.says,'#7fe0ff')"),
    # traps (owner, 2026-09-02): a flashing red vortex per mine, drawn in the page's own scene
    ("let scene,cam,rnd,ctr,composer,vintagePass,agent,group,pellets,trailDots=[]",
     "let scene,cam,rnd,ctr,composer,vintagePass,agent,group,pellets,mines=new Map(),trailDots=[]"),
    ("function drawCompass(angle,intensity,name,color,msg){",
     "function syncMines(list){ const want=new Set((list||[]).map(c=>c.join(','))); "
     "for(const [k,m] of mines){ if(!want.has(k)){ group.remove(m); mines.delete(k); } } "
     "for(const c of (list||[])){ const k=c.join(','); if(mines.has(k)) continue; "
     "const m=new THREE.Mesh(new THREE.TorusGeometry(0.30,0.07,10,28), new THREE.MeshStandardMaterial({color:0xff2d2d,emissive:0xff1a1a,emissiveIntensity:1.0,metalness:0.3,roughness:0.4})); "
     "m.position.set(...W3(...c)); m.rotation.x=Math.PI/2; m.userData.vortex=true; group.add(m); mines.set(k,m); } }\n"
     "function drawCompass(angle,intensity,name,color,msg){"),
    ("pellets.set(p.join(','),m);}", "pellets.set(p.join(','),m);}\n  syncMines(D.mines);"),
    ("if(s.eaten){const k=s.eaten.join(',');", "syncMines(s.mines); if(s.eaten){const k=s.eaten.join(',');"),
    ("'  ·  pellets left: '+s.remaining", "'  ·  pellets left: '+s.remaining+'  ·  ◎ traps: '+(s.mines_left??0)"),
    ("<b>● pellet</b> · <i>◆ hazard</i>", "<b>● pellet</b> · <b style=\"color:#ff3b3b\">◎ trap</b> · <i>◆ hazard</i>"),
    ("pellets.forEach(m=>{ if(m.userData.star) m.rotation.z=now/600; });",
     "pellets.forEach(m=>{ if(m.userData.star) m.rotation.z=now/600; }); "
     "mines.forEach(m=>{ m.rotation.z=now/220; const sc=1+0.22*Math.sin(now/130); m.scale.set(sc,sc,1); "
     "m.material.emissiveIntensity=0.6+0.9*Math.abs(Math.sin(now/160)); });"),
]


# the console panel (owner, 2026-09-02): the brain's event feed (GET /events)
# under the big stage number, so the game and what Cubby does are one screen.
# Injected before </body>; everything is ours, nothing of the lifted page moves.
_CONSOLE_PANEL = r"""
<style>
  #cubbycon{position:fixed;right:3vw;top:71vh;bottom:12px;width:min(36vw,480px);z-index:12;background:rgba(8,12,20,.86);
    border:1px solid #2a3a55;border-radius:12px;box-shadow:0 8px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;
    font:11.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#c9d1d9}
  #cubbycon .hd{padding:7px 12px;border-bottom:1px solid #2a3a55;color:#8b949e;display:flex;gap:10px;align-items:center}
  #cubbycon .hd b{color:#58a6ff}#cubbycon .hd a{color:#58a6ff;text-decoration:none;margin-left:auto}
  #cubbycon .hd label{cursor:pointer}
  #cubbylog{flex:1;overflow-y:auto;padding:8px 12px;white-space:pre-wrap;word-break:break-word}
  #cubbylog .ev{margin:1px 0}#cubbylog .k{display:inline-block;min-width:88px}
  .k-user{color:#d29922}.k-sense{color:#bc8cff}.k-route{color:#58a6ff}.k-walk{color:#3fb950}.k-emit{color:#8b949e}
  .k-vm{color:#3fb950}.k-gate{color:#d29922}.k-learn{color:#3fb950}.k-speak{color:#c9d1d9}.k-explore{color:#58a6ff}
  .k-probe{color:#f85149}.k-derive{color:#bc8cff}.k-eat{color:#ffe66d}.k-caught{color:#f85149}.k-flee{color:#ff9f43}
  .k-plan{color:#8b949e}.k-program{color:#bc8cff}.k-forge{color:#bc8cff}.k-superpower_move{color:#27d3ff}
  .k-level_up{color:#2ce6a8}.k-out_of_time{color:#ff7043}.k-live_error{color:#f85149}.k-ghost_eaten{color:#2ce6a8}
  .k-says{color:#ffe66d}.k-retire{color:#8b949e}.k-restart{color:#ff7043}
  .k-thought{color:#ffe66d;font-style:italic}.k-modify{color:#bc8cff}
</style>
<div id="cubbycon"><div class="hd"><b>cubby</b> console · what he does, as he does it
  <label><input id="cubbyauto" type="checkbox" checked> follow</label><a href="/" target="_blank">full console ↗</a></div>
  <div id="cubbylog"></div></div>
<script>
(function(){
  const log=document.getElementById('cubbylog'), auto=document.getElementById('cubbyauto');
  let since=0; const H=['dopamine','serotonin','cortisol','oxytocin','noradrenaline'];
  const esc=s=>{const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;};
  function fmt(e){const d={...e};delete d.i;delete d.t;delete d.kind;
    if(e.kind==='sense')return `emotion=${d.emotion} `+H.map(h=>`${h.slice(0,4)}=${(d.state?.[h]??0).toFixed(2)}`).join(' ');
    if(e.kind==='walk')return `${d.verified?'VERIFIED':(d.reason||'walked')} answer=${d.answer??'-'} facts=[${(d.facts||[]).join(' | ')}]`;
    if(e.kind==='speak')return `(${d.turn_kind}, ${d.register||'-'}) "${d.reply}"`;
    if(e.kind==='program')return `${d.name} (${d.pattern||d.why}) — ${d.why}: ${(d.reasoning&&d.reasoning.verdict)||''}`;
    if(e.kind==='forge')return `${d.name} ${d.ok?'CERTIFIED':'REJECTED'} got=${d.got} expected=${d.expected}`;
    if(e.kind==='explore')return `${d.from||''} —${d.chosen}→ ${d.place} new=${d.new}${d.probed?' probed='+d.probed:''}`;
    if(e.kind==='thought')return `“${d.text}”${d.verbalized?'':' (host)'}`;
    return Object.entries(d).filter(([k])=>k!=='program'&&k!=='reasoning').map(([k,v])=>`${k}=${typeof v==='object'?JSON.stringify(v):v}`).join(' ');}
  async function poll(){
    try{const r=await fetch(`/events?since=${since}`);const j=await r.json();
      for(const e of j.events){log.insertAdjacentHTML('beforeend',`<div class="ev"><span class="k k-${e.kind}">${esc(e.kind)}</span> ${esc(fmt(e))}</div>`);}
      while(log.children.length>400)log.removeChild(log.firstChild);
      since=j.next; if(auto.checked&&j.events.length)log.scrollTop=log.scrollHeight;
    }catch(err){}
    setTimeout(poll,700);
  }
  poll();
})();
</script>
"""

# the sound layer (2026-09-02): the owner's soundtrack loops quietly under
# 8-bit oscillator effects for the game's events (a step ticks, a pellet is
# the two-note waka, a star an arpeggio, a trap a noise burst, a catch a
# falling saw…). Browsers need a gesture before audio: the toggle is that
# gesture; the choice and the music volume are remembered per browser.
_SOUND_PANEL = r"""
<audio id="bgm" loop preload="auto" src="/pac/music"></audio>
<audio id="jingle" preload="auto" src="/pac/sfx/game_start"></audio>
<div id="sndbox" style="position:fixed;left:14px;bottom:14px;z-index:60;display:flex;gap:8px;align-items:center;
  background:rgba(8,10,24,.78);border:1px solid rgba(120,140,255,.35);border-radius:10px;padding:6px 10px;
  font:600 12px system-ui,sans-serif;color:#dfe6ff;backdrop-filter:blur(6px)">
  <button id="sndbtn" style="background:#22264a;color:#dfe6ff;border:1px solid #5a63b0;border-radius:6px;padding:3px 9px;cursor:pointer;font:inherit">♪ sound off</button>
  <label style="display:flex;align-items:center;gap:5px;opacity:.9">music <input id="bgmvol" type="range" min="0" max="60" value="18" style="width:70px"></label>
</div>
<script>
(function(){
  const bgm=document.getElementById('bgm'), btn=document.getElementById('sndbtn'), vol=document.getElementById('bgmvol');
  let ac=null, on=false;
  try{ on=localStorage.getItem('cb_sound')==='1'; const v=localStorage.getItem('cb_bgm'); if(v!=null) vol.value=v; }catch(e){}
  bgm.volume=vol.value/100;                                   // the soundtrack sits UNDER the effects
  function ctx(){ if(!ac){ const AC=window.AudioContext||window.webkitAudioContext; if(AC) ac=new AC(); }
    if(ac&&ac.state==='suspended') ac.resume(); return ac; }
  function paint(){ btn.textContent=on?'♪ sound on':'♪ sound off'; btn.style.background=on?'#2f8a5a':'#22264a'; }
  function start(){ ctx(); bgm.play().catch(()=>{}); }
  function stop(){ bgm.pause(); }
  btn.onclick=function(){ on=!on; try{ localStorage.setItem('cb_sound',on?'1':'0'); }catch(e){} paint(); if(on){ start(); SFX.pellet(); } else stop(); };
  vol.oninput=function(){ bgm.volume=vol.value/100; try{ localStorage.setItem('cb_bgm',vol.value); }catch(e){} };
  if(on){ const kick=()=>{ start(); window.removeEventListener('pointerdown',kick); window.removeEventListener('keydown',kick); };
    window.addEventListener('pointerdown',kick); window.addEventListener('keydown',kick); }
  paint();
  // 8-bit voice: square/triangle/saw oscillators with short envelopes
  function tone(f0, ms, type, vol, f1){ const a=ctx(); if(!a||!on) return; const t=a.currentTime, o=a.createOscillator(), g=a.createGain();
    o.type=type||'square'; o.frequency.setValueAtTime(f0,t); if(f1) o.frequency.exponentialRampToValueAtTime(f1,t+ms/1000);
    g.gain.setValueAtTime(vol||0.1,t); g.gain.exponentialRampToValueAtTime(0.0001,t+ms/1000);
    o.connect(g); g.connect(a.destination); o.start(t); o.stop(t+ms/1000+0.02); }
  function seq(notes, step, type, vol){ notes.forEach((f,i)=>setTimeout(()=>tone(f, step*1.7, type, vol), i*step)); }
  function noise(ms, vol){ const a=ctx(); if(!a||!on) return; const n=Math.floor(a.sampleRate*ms/1000), b=a.createBuffer(1,n,a.sampleRate), d=b.getChannelData(0);
    for(let i=0;i<n;i++) d[i]=(Math.random()*2-1)*(1-i/n); const src=a.createBufferSource(); src.buffer=b; const g=a.createGain(); g.gain.value=vol||0.16;
    src.connect(g); g.connect(a.destination); src.start(); }
  const SFX={
    move:      ()=>tone(170,45,'triangle',0.05),
    jump:      ()=>tone(280,140,'square',0.09,1100),
    rest:      ()=>tone(330,260,'sine',0.05,495),
    pellet:    ()=>{ tone(880,60,'square',0.11); setTimeout(()=>tone(1320,80,'square',0.10),60); },
    star:      ()=>seq([523,659,784,1047,1319],70,'square',0.11),
    ghost:     ()=>seq([1047,1319,1568,2093],55,'square',0.11),
    caught:    ()=>tone(540,650,'sawtooth',0.13,55),
    gameover:  ()=>seq([392,349,311,262,196],170,'sawtooth',0.12),
    trapped:   ()=>{ noise(200,0.2); tone(130,240,'square',0.12,40); },
    mine:      ()=>tone(250,100,'square',0.08,125),
    level:     ()=>seq([523,659,784,1047,784,1047,1319],95,'square',0.12),
    fail:      ()=>seq([440,415,392,370],150,'triangle',0.10),
    superpower:()=>seq([659,880,1175,1568],65,'square',0.10),
  };
  // level start: the jingle plays over the ducked soundtrack (owner's game-start sample)
  const jingle=document.getElementById('jingle'); jingle.volume=0.55;
  jingle.addEventListener('ended', ()=>{ bgm.volume=vol.value/100; });
  window.cbLevelStart=function(D){ if(!on) return; try{ bgm.volume=(vol.value/100)*0.3; jingle.currentTime=0; jingle.play().catch(()=>{ bgm.volume=vol.value/100; }); }catch(e){} };
  let last={fr:0, to:null, rest:false};
  window.cbSfx=function(s){
    const fr=s.frightened||0, star=!!s.eaten && fr>last.fr, moved=s.to && (!last.to || s.to[0]!==last.to[0]||s.to[1]!==last.to[1]||s.to[2]!==last.to[2]);
    if(on){
      if(s.caught) SFX[s.caught==='gameover'?'gameover':'caught']();
      else if(s.trapped>0) SFX.trapped();
      else if(s.ate_ghost) SFX.ghost();
      else if(s.beaten) SFX.level();
      else if(s.failed) SFX.fail();
      else if(star) SFX.star();
      else if(s.learned) SFX.superpower();
      else if(s.eaten) SFX.pellet();
      else if(s.mined) SFX.mine();
      else if(s.move==='rest'){ if(!last.rest) SFX.rest(); }
      else if(s.move && s.move.indexOf('jump_')===0) SFX.jump();
      else if(moved) SFX.move();
    }
    last={fr:fr, to:s.to, rest:s.move==='rest'};
  };
})();
</script>
"""


def load_frontend(source: pathlib.Path = PACMAN_LIVE) -> tuple[str | None, list[str]]:
    """pacman_live.py's EXACT page, extracted at serve time (never imported),
    with the patches above applied and our console panel injected before
    </body>. -> (html, patches that did not apply)."""
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
    if "</body>" in html:
        html = html.replace("</body>", _CONSOLE_PANEL + _SOUND_PANEL + "</body>", 1)
    else:
        html += _CONSOLE_PANEL + _SOUND_PANEL
    return html, missed


MUSIC_PATH = pathlib.Path(os.environ.get("CB_PAC_MUSIC") or
                          (pathlib.Path(__file__).resolve().parent / "data" / "music" / "neon_pixel_dash.mp3"))


def load_music(path: pathlib.Path = MUSIC_PATH) -> bytes | None:
    """The background soundtrack (the owner's "Neon Pixel Dash", 2026-09-02;
    override with CB_PAC_MUSIC). None when absent -> the page stays silent
    but the 8-bit effects still play."""
    try:
        return path.read_bytes() if path.suffix.lower() in (".mp3", ".ogg", ".wav") and path.is_file() else None
    except OSError:
        return None


def load_sfx(name: str, base: pathlib.Path = MUSIC_PATH.parent) -> bytes | None:
    """A sampled effect from the music dir: /pac/sfx/<name> -> <name>.mp3
    (name-safe). Today: `game_start` (the level-start jingle)."""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", name or ""):
        return None
    try:
        fp = base / f"{name}.mp3"
        return fp.read_bytes() if fp.is_file() else None
    except OSError:
        return None


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
