"""verse — the toy cubbyverse: cubby-man's OWN world model, learned by exploring.

Wired: WIRED (the reference `worlds.CubbyPlugin`; mounted on `serve.CubbyBrain`
by tests and demos; nothing in cubbyllm/ imports this).

Owner's rule (2026-08-31): the cubbyverse has its own world model, and
cubby-man starts knowing ONLY the basics — everything else it must discover.
So:

  * `ToyVerse` is a deterministic little world (a seeded grid of places, each
    with a color and a treasure). Its ground truth lives in the env, NOT in
    any store the brain can read.
  * `CubbyMan` starts with two facts (who he is, where the start is). Every
    move is VM-MEDIATED — the same IAgent ASK machinery as speech: the
    program offers the exits, `VM::resume` rejects a direction it did not
    offer. On arrival he OBSERVES the place, and each observation is written
    into his own `FactStore` AT THAT MOMENT, through the same anti-poisoning
    gate user facts pass (dedup + contradiction) — learning happens AS he
    explores, not as a batch.
  * Curiosity is count-based novelty, modulated by the REAL hormones when
    mounted: dopamine-leaning states prefer unvisited exits, a stressed
    (high-cortisol) state prefers known ground; discoveries feed novelty
    back into the ODE, so finding new rooms literally makes Cubby curious.
  * He TRIES THINGS OUTSIDE THE SCOPE of what is offered: sometimes he
    answers the ASK with a direction the program did NOT offer — the VM's
    chosen-not-invented guard rejects it, and the rejection itself is
    learned ("a wall is the west neighbor of the hall", which is env-true).
    Deny-by-default as a teacher.
  * He WRITES HIS OWN PROGRAMS to join known facts: a counting program the
    VM executes (the exit count is COMPUTED on the VM, not by the host) and
    symmetry-join certificates ("B is the north neighbor of A" ⇒ "A is the
    south neighbor of B" — knowledge about places not yet visited, accepted
    only if the join program binds and recovers cleanly). The semantic
    axioms (opposite directions, counting) are his; the VM executes and
    certifies each derivation — the CotChain trust model.
  * Once discovered or derived, facts answer through the NORMAL reasoning
    cortex (walk → emitter → VM → gates). Before discovery, the same
    question gets the don't-know line. That arc — can't answer, explores,
    can answer — is the demo.

The real cubbyverse repo implements this same contract over its own worldgen
and cubby-man game; this file is the runnable reference (and the toy's
numbers are [stand-in] like everything else here).
"""
from __future__ import annotations

import random
import re
import time

__wiring__ = "WIRED"

from chat import render_talk_program  # noqa: E402
from worlds import FactStore  # noqa: E402

_PLACES = ["the hall", "the garden", "the kitchen", "the attic", "the cellar",
           "the tower", "the library", "the workshop", "the pond"]
_TREASURES = ["a golden key", "a silver lantern", "a red balloon", "an old map",
              "a music box", "a glass marble", "a copper bell", "a paper crown", "a tiny boat"]
_COLORS = ["blue", "green", "yellow", "purple", "orange", "teal", "crimson", "grey", "pink"]
_DIRS = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


class ToyVerse:
    """A seeded w×h grid world. Ground truth stays here — the env is the
    territory, the explorer's FactStore is his map of it."""

    def __init__(self, w: int = 3, h: int = 2, seed: int = 0) -> None:
        rng = random.Random(seed)
        cells = [(x, y) for y in range(h) for x in range(w)]
        places = _PLACES[: len(cells)]
        rng.shuffle(places)
        self.at = dict(zip(cells, places))
        self.pos = {p: c for c, p in self.at.items()}
        treasures = rng.sample(_TREASURES, len(cells))
        colors = rng.sample(_COLORS, len(cells))
        self.treasure = dict(zip(places, treasures))
        self.color = dict(zip(places, colors))
        self.start = self.at[(0, 0)]

    def exits(self, place: str) -> dict[str, str]:
        """direction -> neighboring place."""
        x, y = self.pos[place]
        out = {}
        for (dx, dy), d in _DIRS.items():
            nbr = self.at.get((x + dx, y + dy))
            if nbr:
                out[d] = nbr
        return out

    def observe(self, place: str) -> list[str]:
        """What standing in `place` reveals — template facts the reasoning
        cortex can walk."""
        obs = [f"{self.color[place]} is the color of {place}",
               f"{self.treasure[place]} is the treasure of {place}"]
        for d, nbr in self.exits(place).items():
            obs.append(f"{nbr} is the {d} neighbor of {place}")
        return obs

    def all_facts(self) -> list[str]:
        return [f for p in self.pos for f in self.observe(p)]


class CubbyMan:
    """The explorer-cortex: owns the cubbyverse world model, starts with the
    basics, learns as he goes. Implements the plugin contract in one object
    (worlds/cortices/bind/on_turn-free). Subclasses re-skin it for other
    worlds (see standin/pacman.py) by overriding the class attrs, the env,
    and the `on_arrive` hook."""

    name = "cubbyverse"
    CORTEX = "game"
    DIR_NAMES = ("north", "south", "east", "west")
    OPP = _OPP
    _GO = re.compile(r"\b(explore|wander|play|adventure|explore[rz]?|balade|visite|joue|aventure)\b", re.I)

    def __init__(self, env: ToyVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35) -> None:
        self.env = env or ToyVerse()
        self.exe = exe
        self.rng = random.Random(seed)
        self.probe = float(probe)                        # chance of trying an unoffered direction
        self.world = FactStore(name=self.name)
        self._nbr_re = re.compile(rf"^(?P<b>.+?) is the (?P<d>{'|'.join(self.DIR_NAMES)}) "
                                  rf"neighbor of (?P<a>.+)$")
        self._seed_basics()                              # everything else must be discovered
        self.place = self.env.start
        self.visits: dict[str, int] = {}
        self.chem = None                                 # set by bind(); optional
        self.derived: set[str] = set()                   # facts he PROVED rather than saw
        self.walls: set[str] = set()                     # learned from VM rejections
        self.anomalies: list[str] = []                   # a guard that FAILED to reject
        self.log: list[dict] = []
        self._learn(self.env.observe(self.place))        # he can see where he stands
        self.on_arrive(self.place)
        self.visits[self.place] = 1

    def _seed_basics(self) -> None:
        self.world.add(f"cubbyman is the explorer of the {self.name}")
        self.world.add(f"{self.env.start} is the start of the {self.name}")

    def on_arrive(self, place: str) -> int:
        """Hook for worlds where arriving DOES something (eating a pellet,
        triggering an event). Returns extra facts learned."""
        return 0

    # ── plugin contract ─────────────────────────────────────────────────────
    def bind(self, brain) -> None:
        self.chem = brain.chat.chem
        self.exe = self.exe or brain.exe
        self._trace = getattr(brain, "trace", None)      # the console feed

    def _t(self, kind: str, **data) -> None:
        if getattr(self, "_trace", None):
            self._trace(kind, **data)

    def worlds(self) -> dict[str, FactStore]:
        return {self.name: self.world}

    def cortices(self) -> dict[str, object]:
        return {self.CORTEX: self}

    def match(self, text: str) -> float:
        return 1.0 if self._GO.search(text) else 0.0

    def handle(self, text: str) -> dict:
        m = re.search(r"\b(\d{1,2})\b", text)
        steps = int(m.group(1)) if m else 6
        rep = self.explore(steps)
        line = (f"I explored the cubbyverse: {rep['new_facts']} new things learned "
                f"across {rep['places_visited']} places, {len(self.derived)} of them "
                f"worked out on my own. Ask me about them!")
        return {"offered": [line], "meta": rep}

    # ── the exploration loop: move (VM-mediated) → observe → learn ──────────
    def _learn(self, observations: list[str]) -> int:
        """Write observations into the world model AT DISCOVERY TIME, through
        the same gate user facts pass."""
        from serve import MemoryCortex
        new = 0
        for fact in observations:
            if MemoryCortex.contradiction(fact, self.world) is None:
                new += int(self.world.add(fact))
        return new

    def _pick(self, exits: dict[str, str]) -> str:
        """Curiosity policy: prefer the least-visited destination; a stressed
        state (cortisol ≥ 0.35) flips to preferring known ground. The pick is
        only a PREFERENCE — the VM's ASK still guards that the chosen
        direction was actually offered."""
        prefer_known = self.chem is not None and self.chem.cortisol >= 0.35
        ranked = sorted(exits, key=lambda d: (self.visits.get(exits[d], 0), self.rng.random()),
                        reverse=prefer_known)
        return ranked[0]

    def _try_the_wall(self, src: str, dirs: list[str]) -> str | None:
        """Try a direction the program did NOT offer. The VM's resume guard
        must reject it; the rejection teaches him where the walls are. If
        the guard ever ACCEPTS, that is an anomaly worth recording, not a
        fact worth learning."""
        from cubbyllm.bridges import cubelang_client as cc
        missing = sorted(set(self.DIR_NAMES) - set(dirs))
        if not missing or self.rng.random() >= self.probe:
            return None
        d = self.rng.choice(missing)
        try:
            cc.resume_program_proto(src, fn="think", args=[self.place, "explore"],
                                    answers=[d], exe=self.exe)
            self.anomalies.append(f"resume accepted unoffered '{d}' at {self.place}")
            return None
        except cc.CubelangRunError:
            fact = f"a wall is the {d} neighbor of {self.place}"
            if self._learn([fact]):
                self.walls.add(fact)
            self._t("probe", place=self.place, tried=d, rejected=True, learned=fact)
            return d

    def step(self) -> dict:
        from cubbyllm.bridges import cubelang_client as cc
        exits = self.env.exits(self.place)
        dirs = sorted(exits)
        src = render_talk_program(dirs, question="which way next")
        asked = cc.run_program_proto(src, fn="think", args=[self.place, "explore"], exe=self.exe)
        if not asked.get("suspended") or asked["candidates"] != dirs:
            raise RuntimeError(f"the VM did not offer the exits: {asked}")
        probed = self._try_the_wall(src, dirs)
        chosen = self._pick(exits)
        res = cc.resume_program_proto(src, fn="think", args=[self.place, "explore"],
                                      answers=[chosen], exe=self.exe)
        cc.run_program_proto(src, fn="act", args=[f"went {res['result']} from {self.place}", "explore"],
                             exe=self.exe)
        came_from = self.place
        self.place = exits[chosen]
        self.visits[self.place] = self.visits.get(self.place, 0) + 1
        obs = self.env.observe(self.place)
        new = self._learn(obs)
        new += self.on_arrive(self.place)                # world-specific arrival effects (eating…)
        new += self.derive_symmetry(obs)                 # join facts the moment they land
        if self.chem is not None:                        # discovery feeds curiosity
            self.chem.update(novelty=new / max(1, len(obs)), valence=0.2 * min(1, new))
        rec = {"from": came_from, "place": self.place, "chosen": chosen, "new": new,
               "probed": probed}
        self._t("explore", **rec)
        self.log.append(rec)
        return rec

    # ── his own programs: join known facts on the VM ────────────────────────
    def _certify_join(self, program: str, expect: str) -> bool:
        """Run one of his own join programs; the derivation is accepted only
        if it executes and recovers exactly the proposed object."""
        from cubbyllm.bridges import cubelang_client as cc
        try:
            out = cc.run_program_proto(program, fn="solve", exe=self.exe)
            return out.get("result") is not None and str(out["result"]) == expect
        except cc.CubelangRunError:
            return False

    def derive_symmetry(self, facts: list[str]) -> int:
        """"B is the d neighbor of A" ⇒ "A is the opp(d) neighbor of B" —
        knowledge about places not yet visited. The opposite-direction axiom
        is his; each join runs as a bind/recover certificate on the VM."""
        from serve import MemoryCortex
        n = 0
        for f in facts:
            m = self._nbr_re.match(" ".join(f.split()))
            if not m or m.group("b") == "a wall":
                continue
            derived = f"{m.group('a')} is the {self.OPP[m.group('d')]} neighbor of {m.group('b')}"
            if derived in self.world or MemoryCortex.contradiction(derived, self.world) is not None:
                continue
            program = ("use vsa;\n\nprogram JoinSym implements ISolve {\n"
                       "    public function solve(mention: str): str {\n"
                       "        create frame: number;\n"
                       f'        bind frame, H1_SOURCE, "{m.group("b")}";\n'
                       f'        bind frame, H2_DERIVED, "{m.group("a")}";\n'
                       "        return recover(frame, H2_DERIVED);\n    }\n}\n")
            if self._certify_join(program, m.group("a")) and self.world.add(derived):
                self.derived.add(derived)
                self._t("derive", rule="symmetry", fact=derived)
                n += 1
        return n

    def derive_counts(self) -> int:
        """For each place he knows neighbors of, WRITE a counting program —
        the exit count is computed BY THE VM (one `add` per known neighbor
        fact), then stored as a joined fact he was never shown."""
        n = 0
        by_place: dict[str, int] = {}
        for f in self.world.texts:
            m = self._nbr_re.match(f)
            if m and m.group("b") != "a wall":
                by_place[m.group("a")] = by_place.get(m.group("a"), 0) + 1
        for place, count in sorted(by_place.items()):
            derived = f"{count} is the exit count of {place}"
            if derived in self.world:
                continue
            adds = "        add n, 1;\n" * count
            program = ("program JoinCount implements ISolve {\n"
                       "    public function solve(mention: str): str {\n"
                       "        create n : symbol;\n        assign n = 0;\n"
                       f"{adds}        return n;\n    }}\n}}\n")
            if self._certify_join(program, str(count)) and self.world.add(derived):
                self.derived.add(derived)
                self._t("derive", rule="count", fact=derived, vm_computed=count)
                n += 1
        return n

    def explore(self, steps: int = 6) -> dict:
        t0 = time.perf_counter()
        before = len(self.world)
        for _ in range(steps):
            self.step()
        counted = self.derive_counts()                   # join what the walk collected
        total = len(self.env.all_facts())
        return {"steps": steps, "new_facts": len(self.world) - before,
                "places_visited": len(self.visits), "coverage": round(self.coverage(), 3),
                "derived": len(self.derived), "counted": counted, "walls": len(self.walls),
                "anomalies": list(self.anomalies),
                "world_size": len(self.world), "total_env_facts": total,
                "wall_s": round(time.perf_counter() - t0, 3)}

    def coverage(self) -> float:
        """Fraction of the env's ground-truth facts now in his world model."""
        env_facts = self.env.all_facts()
        return sum(1 for f in env_facts if f in self.world) / max(1, len(env_facts))
