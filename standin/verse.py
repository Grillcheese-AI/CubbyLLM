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
  * He is offered WHAT HIS BODY CAN DO, not what the world says is legal,
    so a move can FAIL. The ASK lists every direction his own map has not
    ruled out; the WORLD resolves the attempt and may refuse it; the
    refusal is the percept ("a wall is the west neighbor of the hall",
    which is env-true because the world is the one that said so). Walking
    into things is how the map gets made. Before 2026-09-15 the offer was
    pre-filtered to the legal moves, which made a collision impossible and
    handed him the maze for free.
  * He also TRIES THINGS OUTSIDE THE SCOPE of what is offered, as a guard
    audit: the VM's chosen-not-invented rule must reject a direction the
    program did not list, and an acceptance is recorded as an anomaly. That
    probe no longer TEACHES anything — the world is the only authority on
    what is solid.
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

import os
import random
import re
import threading
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

    def try_move(self, place: str, move: str, offered: dict[str, str]) -> dict:
        """The world resolves a move ATTEMPT. This grid has no stone, only an
        edge — walk off it and the world says so, which is how he learns the
        shape of the place instead of being handed it."""
        nbr = self.exits(place).get(move)
        if nbr:
            return {"ok": True, "to": nbr, "kind": "open"}
        return {"ok": False, "to": place, "kind": "edge"}

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
    # what a refused move leaves in his map. These are NOT places: the
    # opposite-direction axiom must not run through them, and his route
    # planner must not path through them. `kind` -> the words he learns it in.
    NON_PLACES = {"wall": "a wall", "hazard": "a hazard", "edge": "the edge"}

    # ── what he is born able to do, and what he arrives with ───────────────
    # These live HERE, on the agent, not on any one world. Nick, 2026-09-15:
    # *"these behaviours should be global not one per context... the same
    # (exactly the same) should be portable to other contexts and vice-versa."*
    # Cubby-man is A context. A capability he earns in it is a capability he
    # has, and the map he builds is ONE map — so the gate, the blank slate and
    # the hypothesis ledger belong to him and travel with him.
    BORN_WITH: frozenset = frozenset()                   # capabilities he starts with; everything else is earned

    # WHAT HE IS HERE FOR (owner, 2026-09-15: *"we give it an important
    # mission: survive the maze at all cost!"*). A standing goal, in his own
    # words, on the agent rather than in any world — the same sentence travels
    # with him. It is deliberately NOT a rule and nothing branches on its text:
    # it reaches the percept record so it colours what he says and what he
    # weighs, and it raises `mission_pressure`, the extra wanting he has to
    # feel before he will take a risk against it. That is the honest shape of a
    # mission an agent can also fail to keep — and the tension is the point,
    # because a mission that cannot be overridden is not a mission, it is a
    # guard clause.
    mission: str = ""
    mission_pressure: float = 0.0                        # how much harder it is to talk himself into a risk
    # the game's own words claim a turn outright; a bare command only claims a
    # statement (the brain never hands a plugin a question at 0.6); a turn about
    # something else that happens to say "explore" (the web, a plugin) is not ours
    _GO = re.compile(r"\b(explore (for )?\d+|adventure|aventure|balade|visite)\b", re.I)   # not the world's NAME: a question about it is identity
    _CMD = re.compile(r"\b(explore[rz]?|wander|play|joue[rz]?)\b", re.I)
    _NOT_GAME = re.compile(r"\b(web|internet|online|browser?|site|plugin|world ?wide|api|file|dossier|fichier)\b", re.I)

    def __init__(self, env: ToyVerse | None = None, exe: str | None = None, seed: int = 0,
                 probe: float = 0.35, blank: bool | None = None) -> None:
        # BLANK: nothing inherited from an earlier run. Nick: *"we should be
        # able to clear the memories when testing it."* An agent that loads a
        # library off disk is not a fresh agent, and an experiment that starts
        # from one is measuring the disk. CUBBYMAN_BLANK=1 forces it globally;
        # a subclass that persists anything must honour `self.blank`.
        self.blank = (bool(int(os.environ.get("CUBBYMAN_BLANK", "0")))
                      if blank is None else bool(blank))
        self.can: set[str] = set(self.BORN_WITH)         # what he has EARNED the ability to do
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
        self.walls: set[str] = set()                     # obstacles learned by walking into them
        self._blocked: dict[str, set[str]] = {}          # his map's blocked directions, by place
        self._nbrs: dict[str, dict[str, str]] = {}       # his map's named neighbours, by place
        self._blocked_n = -1                             # the world size both caches were built at
        self.anomalies: list[str] = []                   # a guard that FAILED to reject
        self._learned_here: list[str] = []               # facts learned THIS step — the vocabulary he may speak from
        self.log: list[dict] = []
        self._step_lock = threading.RLock()              # one move at a time (live poller vs chat turn)
        # his open questions, and what settling them earned. On the AGENT, not
        # the world: a question raised in one context and a verdict reached in
        # another are the same ledger, because there is one of him.
        from hypothesis import Hypotheses, ask_verifier, vm_verifier, world_verifier
        from worlds import Worlds
        # the worlds OUTSIDE his own that he can put a question to. On the agent
        # for the same reason the ledger is: a law he gets from physics while
        # playing cubby-man is a law he has, and he takes it to the next world
        # with him because there is only ever one map (WO-2.13).
        self.other_worlds = Worlds(trace=self._t)
        self.guesses = Hypotheses({"world": world_verifier(self.env),
                                   "vm": vm_verifier(self.exe),
                                   "ask": ask_verifier(self.other_worlds)}, trace=self._t)
        self._learn(self.look_around(self.place))        # what he perceives where he starts
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
        """1.0 on the game's own words, 0.6 on a bare command, 0 when the turn
        is about something else (the web, a plugin) or asks a question."""
        if self._NOT_GAME.search(text):
            return 0.0
        if self._GO.search(text):
            return 1.0
        return 0.6 if self._CMD.search(text) else 0.0

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
                if self.world.add(fact):
                    new += 1
                    # kept for this step only: these are the words he has
                    # earned the right to use when he speaks about it
                    self._learned_here.append(fact)
        return new

    def _pick(self, exits: dict[str, str]) -> str:
        """Curiosity, through the affect filter. The pick is only a PREFERENCE
        — the VM's ASK still guards that the chosen direction was offered.

        THIS IS WHERE EVERY WORLD'S DISCRETIONARY CHOICE PASSES, which is why
        the filter goes here and not in a subclass. ToyVerse, the maze, and
        anything written later all inherit it, and none of them has to know
        that a body is involved.

        WHAT IT REPLACES: `prefer_known = chem.cortisol >= 0.35`, which flipped
        the sort. One hormone, one hand-placed cliff, a binary answer to a
        graded question — the same shape as the `fear`-fitted danger radius
        this project already threw out for being invented rather than measured.
        The behaviour it reached for is real (frightened animals prefer known
        ground) and survives: `novelty` is the one term the state may push
        negative, so a calm body is drawn to the unvisited and an alarmed one
        is pushed away from it. Graded, from all seven knobs, with no threshold
        anywhere.

        `visits` is turned into novelty as `1 / (1 + visits)`: unseen is 1.0,
        seen once is 0.5. The world supplies the number; the body supplies what
        it is worth."""
        if self.chem is None:                            # no body: plain curiosity
            return sorted(exits, key=lambda d: (self.visits.get(exits[d], 0),
                                                self.rng.random()))[0]
        import choosing
        opts = [choosing.Option(d, novelty=1.0 / (1.0 + self.visits.get(exits[d], 0)))
                for d in exits]
        self.rng.shuffle(opts)                           # ties break freely, as they did before
        winner, _ = choosing.choose(opts, self.chem.modulation())
        return winner.key

    def _try_the_wall(self, src: str, dirs: list[str]) -> str | None:
        """Try a direction the program did NOT offer, and check the VM's
        resume guard rejects it. This is a GUARD AUDIT, not a way to learn
        the map: if the guard ever ACCEPTS, that is an anomaly worth
        recording.

        It used to learn `a wall is the d neighbor of here` from the
        rejection — which made the VM a second authority on what is solid,
        and a wrong one (an unoffered direction is only a direction the
        program did not list, and out-of-bounds came back labelled "a wall").
        Since 2026-09-15 the WORLD refuses a move for real and the collision
        is the percept, so there is exactly one source of truth about stone
        and this is no longer it."""
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
            self._t("probe", place=self.place, tried=d, rejected=True, guard="held")
            return d

    def step(self) -> dict:
        with self._step_lock:
            return self._step()

    def body_moves(self) -> dict[str, str]:
        """Every direction his BODY has, whatever the world holds. Where his
        own map already names the neighbour, that name is the destination;
        where it does not, the destination is a placeholder he has never
        visited (so curiosity prefers it) and never arrives at — the WORLD
        says where he actually ends up."""
        mine = self.known_neighbors(self.place)
        return {d: mine.get(d, f"? {d} of {self.place}") for d in self.DIR_NAMES}

    def candidate_moves(self, exits: dict[str, str]) -> dict[str, str]:
        """What his BODY can try, minus what HIS MAP has ruled out.

        `exits` — the world's legal-move list — is deliberately ignored
        (2026-09-15). Pre-filtering the offer to the legal moves meant he
        could never walk into anything, so the world's shape was a gift
        rather than something he learned. Now an untried direction stays
        offered, the world refuses it if it is solid, and the refusal is the
        percept. The filter is his belief doing the work the rule used to do,
        and unlike the rule it can be wrong."""
        offers = self.body_moves()
        blocked = self.known_blocked(self.place)
        return {d: p for d, p in offers.items() if d not in blocked} or offers

    def look_around(self, place: str) -> list[str]:
        """Hook: what he PERCEIVES on arriving somewhere. The default is the
        world's own `observe` — an agent with eyes. A subclass can narrow it
        (see CubbyGhost.SEE_EXITS) so the map arrives only through what he
        walks into."""
        return self.env.observe(place)

    def _rebuild_beliefs(self) -> None:
        """One scan of his world model into two indexes: which directions he
        has found solid, and which neighbours he can name."""
        nonplaces = set(self.NON_PLACES.values())
        blocked: dict[str, set[str]] = {}
        nbrs: dict[str, dict[str, str]] = {}
        for f in self.world.texts:
            m = self._nbr_re.match(f)
            if not m:
                continue
            if m.group("b") in nonplaces:
                blocked.setdefault(m.group("a"), set()).add(m.group("d"))
            else:
                nbrs.setdefault(m.group("a"), {})[m.group("d")] = m.group("b")
        self._blocked, self._nbrs, self._blocked_n = blocked, nbrs, len(self.world)

    def known_blocked(self, place: str) -> set[str]:
        """The directions HIS OWN MAP says are solid here — each one learned
        from a refused move he made, or from something he saw. Nothing is
        read from the env: this is a belief, an untried direction is simply
        absent from it, and it can be wrong."""
        if self._blocked_n != len(self.world):
            self._rebuild_beliefs()
        return self._blocked.get(place, set())

    def known_neighbors(self, place: str) -> dict[str, str]:
        """direction -> the place HIS MAP says is that way (only the ones he
        has learned)."""
        if self._blocked_n != len(self.world):
            self._rebuild_beliefs()
        return self._nbrs.get(place, {})

    # ── settling what he wonders, in any world ─────────────────────────────
    def settle(self, now: int) -> list[str]:
        """Test every open question that can be tested and learn the verdicts.

        Generic on purpose: the ledger, the gate and the learning all belong to
        the agent, so a world only has to call this. `on_capability` is the one
        hook a world needs if a verdict grants him something."""
        earned: list[str] = []
        for fact in self.guesses.sweep(now):
            if self._learn([fact]):
                earned.append(fact)
            gained = self.UNLOCKS.get(fact)
            if gained and gained not in self.can:
                self.can.add(gained)
                self._t("capability", gained=gained, because=fact)
                self.on_capability(gained, fact)
        return earned

    # ── asking a world that already knows ──────────────────────────────────
    # Deliberately high. A settled ask leaves a fact carrying the question
    # verbatim, so the question he really has asked comes back at ~1.0 while
    # anything else sits near 0.2 — the margin is wide because the two errors
    # are both bad: re-asking forever teaches him nothing, and deciding he
    # knows something he does not is how a wrong answer gets spoken.
    KNOWN_TAU = 0.6

    def already_know(self, question: str) -> str | None:
        """The fact in HIS OWN map that already answers this, if there is one.

        This is the whole of *"then cubby stores it in long term memory so it
        knows that part and dont have to ask already"*, and it is also the
        kill criterion for WO-2.13: a fresh agent restored from his map must
        not re-ask. It is deliberately his ordinary retrieval and nothing
        special — the map answers the question, or it does not."""
        hits = self.world(question, 1)
        return hits[0][1] if hits and hits[0][0] >= self.KNOWN_TAU else None

    def ask_elsewhere(self, question: str, claim: str = "", *, now: int = 0,
                      patience: int = 3):
        """He hit something his map cannot explain, so he frames the question
        in his own words and sends it to whoever's domain it is.

        It goes in as a HYPOTHESIS, not as a lookup, and that is the point: an
        answer from a world is a claim that got settled, it lands through the
        same gate as a percept, and if no world knows it stays an open question
        instead of becoming a fact. Returns the hypothesis, or None when he
        already knows (he does not ask twice) or has already asked this."""
        known = self.already_know(question)
        if known is not None:
            self._t("ask_skipped", question=question, because=known)
            return None
        from hypothesis import Hypothesis
        return self.guesses.frame(Hypothesis(
            claim=claim or question, test=f"ask whoever knows: {question}",
            verifier="ask", payload={"question": question}, made_at=now, patience=patience))

    # fact -> the capability it grants. A capability is EARNED by a verdict,
    # never set: being handed one is the maze's legal-move list in a hat.
    UNLOCKS: dict[str, str] = {}

    def on_capability(self, name: str, because: str) -> None:
        """Hook: he can now do something he could not before."""

    def able(self, name: str) -> bool:
        return name in self.can

    @staticmethod
    def ask_label(i: int, move: str, salt: int = 0) -> str:
        """The short, distinct token the ASK offers for a move: index + the
        initials of the move's words (`07-carrdrr` for
        `combo-aabaa_right_right_down_right_right`). The VM decodes string
        values through its symbol space and unrelated strings can collide
        there (2026-09-02: a long name came back as another's duplicate;
        later a label came back as the ASK's own question text), so the
        program offers labels, the host maps the choice back, and on a
        mismatch the labels are RE-ROLLED with a salt (`07a-…`) — a
        collision depends on the exact strings. The guard is unchanged: the
        answer must be one of the offered labels, and a raw direction that
        was not offered is still refused."""
        initials = "".join(w[0] for w in move.replace("-", "_").split("_") if w)
        return f"{i:02d}{'' if salt == 0 else chr(96 + salt)}-{initials[:10]}"

    ASK_RETRIES = 3

    def _ask_exits(self, dirs: list[str]) -> tuple[str, list[str]]:
        """think() with labels for `dirs`; re-rolled on a VM decode glitch.
        -> (program source, labels). Every retry is traced (`ask_retry`) so
        the glitch rate is a number, not a feeling."""
        from cubbyllm.bridges import cubelang_client as cc
        last = ""
        for salt in range(self.ASK_RETRIES):
            labels = [self.ask_label(i, d, salt) for i, d in enumerate(dirs)]
            src = render_talk_program(labels, question="which way next")
            asked = cc.run_program_proto(src, fn="think", args=[self.place, "explore"], exe=self.exe)
            got = asked.get("candidates") or []
            if asked.get("suspended") and got == labels:
                return src, labels
            last = (f"embedded {len(labels)}, back {len(got)}; missing {[d for d in labels if d not in got][:3]}, "
                    f"extra {[g for g in got if g not in labels][:3]}; suspended={asked.get('suspended')}")
            self._t("ask_retry", salt=salt + 1, why=last)
        raise RuntimeError(f"the VM did not offer the exits after {self.ASK_RETRIES} label sets: {last}")

    def _step(self) -> dict:
        from cubbyllm.bridges import cubelang_client as cc
        exits = self.candidate_moves(self.env.exits(self.place))
        dirs = sorted(exits)
        src, labels = self._ask_exits(dirs)
        by_label = dict(zip(labels, dirs))
        probed = self._try_the_wall(src, dirs)
        chosen = self._pick(exits)
        label = labels[dirs.index(chosen)]
        res = cc.resume_program_proto(src, fn="think", args=[self.place, "explore"],
                                      answers=[label], exe=self.exe)
        if res.get("result") != label or by_label.get(res.get("result")) != chosen:
            # the same decode glitch on the way back: re-ask with fresh labels once, then give up loudly
            self._t("ask_retry", salt="resume", why=f"resume returned {res.get('result')!r} for {label!r}")
            src, labels = self._ask_exits(dirs)
            by_label = dict(zip(labels, dirs))
            label = labels[dirs.index(chosen)]
            res = cc.resume_program_proto(src, fn="think", args=[self.place, "explore"],
                                          answers=[label], exe=self.exe)
            if res.get("result") != label:
                raise RuntimeError(f"resume returned {res.get('result')!r}, not the chosen label {label!r}")
        cc.run_program_proto(src, fn="act", args=[f"went {chosen} from {self.place}", "explore"],
                             exe=self.exe)
        came_from = self.place
        # The move is an ATTEMPT, not a guaranteed arrival. The WORLD resolves
        # it and may refuse; the refusal is a PERCEPT — he walked into the
        # thing — so a collision is what teaches an obstacle, and no oracle
        # ever had to list the obstacles for him (2026-09-15: the offer used
        # to be pre-filtered to the legal moves, which made a collision
        # impossible and the map a gift). Worlds with no `try_move` hook keep
        # the old always-arrives behaviour.
        resolve = getattr(self.env, "try_move", None)
        out = (resolve(self.place, chosen, exits) if resolve is not None
               else {"ok": True, "to": exits[chosen], "kind": "open"})
        self.place = out["to"]
        bumped = None if out.get("ok", True) else out.get("kind", "wall")
        new = 0
        if bumped is not None:                           # refused: he is still where he was
            fact = f"{self.NON_PLACES.get(bumped, 'a wall')} is the {chosen} neighbor of {came_from}"
            if self._learn([fact]):
                self.walls.add(fact)
            self._t("bump", place=came_from, tried=chosen, hit=bumped, learned=fact)
            obs = []
        else:
            self.visits[self.place] = self.visits.get(self.place, 0) + 1
            obs = self.look_around(self.place)
            if chosen in self.DIR_NAMES and self.place != came_from:
                # PROPRIOCEPTION: he went that way and ended up here. True
                # whether or not he can see, and the only thing that builds a
                # map for an agent with SEE_EXITS off.
                obs = [f"{self.place} is the {chosen} neighbor of {came_from}"] + list(obs)
            new = self._learn(obs)
            new += self.on_arrive(self.place)            # world-specific arrival effects (eating…)
            new += self.derive_symmetry(obs)             # join facts the moment they land
        if self.chem is not None:                        # discovery feeds curiosity -- as SURPRISE, not routine
            # habituation: what he usually learns per step is expected; only a
            # burst above that expectation reads as novelty (a steady trickle
            # used to pin dopamine high and every mood read "excited")
            ema = getattr(self, "_expect_new", None)
            expected = 1.0 if ema is None else ema
            surprise = max(0.0, new - expected) / (expected + 1.0)
            self._expect_new = new if ema is None else 0.8 * ema + 0.2 * new
            self.chem.update(novelty=min(1.0, 0.7 * surprise), valence=0.1 * min(1, new))
        rec = {"from": came_from, "place": self.place, "chosen": chosen, "new": new,
               "probed": probed, "label": label, "offered": len(labels), "bumped": bumped}
        self._t("explore", **rec)
        self.log.append(rec)
        return rec

    # ── his own programs: join known facts on the VM ────────────────────────
    def _certify_join(self, program: str, expect: str, fn: str = "solve") -> bool:
        """Run one of his own programs (function `fn`); the derivation is
        accepted only if it executes and recovers exactly the proposed object.
        With a ledger mounted (`self.ledger`), the decision — certified or
        rejected — is hashed, signed and stored; `self._last_cert` carries
        the hash for the library entry (ledger.py, 2026-09-04)."""
        from cubbyllm.bridges import cubelang_client as cc
        got, err = None, None
        try:
            out = cc.run_program_proto(program, fn=fn, exe=self.exe)
            got = None if out.get("result") is None else str(out["result"])
            ok = got is not None and got == expect
        except cc.CubelangRunError as e:
            ok, err = False, str(e)[:200]
        ledger = getattr(self, "ledger", None)
        self._last_cert = None
        if ledger is not None:
            self._last_cert = ledger.record(program, fn, "join", fn, None, expect, got, ok,
                                            "certified" if ok else f"REJECTED{(' (' + err + ')') if err else ''}")
        return ok

    def derive_symmetry(self, facts: list[str]) -> int:
        """"B is the d neighbor of A" ⇒ "A is the opp(d) neighbor of B" —
        knowledge about places not yet visited. The opposite-direction axiom
        is his; each join runs as a bind/recover certificate on the VM."""
        from serve import MemoryCortex
        n = 0
        for f in facts:
            m = self._nbr_re.match(" ".join(f.split()))
            # "a hazard"/"the edge" are not places either: deriving
            # "X is the left neighbor of a hazard" was junk (fixed 2026-09-15
            # with the bump percept, which made the same shape reachable for
            # walls that are now learned by collision)
            if not m or m.group("b") in self.NON_PLACES.values():
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
        nonplaces = set(self.NON_PLACES.values())        # a wall / a hazard / the edge are not exits
        for f in self.world.texts:
            m = self._nbr_re.match(f)
            if m and m.group("b") not in nonplaces:
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
