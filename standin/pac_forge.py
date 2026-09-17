"""pac_forge - he invents moves, the VM certifies them, he keeps the good ones.

Wired: WIRED (mixed into `pacman.CubbyGhost`).

One of five clusters cut out of a 1,950-line class body. Nothing about the
clusters is new - they were always there, separated by banner comments - and
splitting them into mixins only makes the seams enforceable: a method in here
cannot quietly start reaching into the speech machinery without the import
saying so.

This is the part that makes a capability EARNED rather than granted. Owner:
*"combos should not be granted either, even the capability of doing them should
be hidden."* So nothing here offers a power move until the agent has raised the
question himself, written the smallest program that would settle it, and had
the VM certify it.
"""
from __future__ import annotations

from pacworld import REST, _manh  # noqa: E402
from world import cross, pattern_dna, pattern_name  # noqa: E402

__wiring__ = "WIRED"


class ForgeMixin:

    # `combos` is LOCKED at birth. Nick, 2026-09-15: *"combos should not be
    # granted either, even the capability of doing them should be hidden"* — so
    # no power move is offered, nothing is composed, mutated or crossed, and
    # the HUD lists no powers. Being handed the knowledge that steps can be
    # chained is being handed the maze's legal moves by another name.
    #
    # The gate itself lives on CubbyMan, because a capability is HIS and not
    # this world's; what is here is only the question this world raises and the
    # verdict that grants it.
    UNLOCKS = {"chaining is the way of a move": "combos"}


    @property
    def powers(self) -> list[str]:
        """His MOVES (jump + patterns), not his forged tools — the HUD's list.
        Empty while `combos` is locked: not "powers he has none of" but a
        capability he does not know exists."""
        if "combos" not in self.can:
            return []
        return [n for n, e in self.library.active().items() if e["kind"] in ("jump", "pattern")]


    def _wonder_about_combos(self, why: str) -> None:
        """One step at a time is not working — can I do more than one?

        Being stuck or out of time is the only thing that raises the question,
        and the VM is what answers it: he writes the smallest possible
        two-step program and it either certifies or it does not. A confirmation
        unlocks the capability AND is a fact he holds; a refusal closes the
        question until something else raises it.

        This is why `combos` is not a flag someone sets. Nick, 2026-09-15:
        *"even the capability of doing them should be hidden"* — a capability
        you were handed is the maze's legal-move list wearing a different hat."""
        if "combos" in self.can or self.guesses is None:
            return
        from hypothesis import Hypothesis
        name = "TWOSTEP"
        program = self.library.moves_program(extra={name: ["step", "step"]})
        fn = self.library.fn_name(name)
        self.guesses.frame(Hypothesis(
            claim="maybe I can take more than one step at a time",
            test="write the smallest two-step move and see whether it holds together",
            verifier="vm",
            if_true="chaining is the way of a move",
            if_false="one at a time is the way of a move",
            payload={"program": program, "fn": fn, "expected": name},
            made_at=self.env.steps))
        self._t("wonder", about="combos", why=why)


    def on_capability(self, name: str, because: str) -> None:
        """CubbyMan grants the capability; this world only says it out loud."""
        self._think("capability", name=name)


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
        if "combos" not in self.can:                     # he does not know moves can be chained yet
            return None
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


    def _cross(self, why: str) -> str | None:
        """CROSS two moves he already owns into one neither of them was.

        Nick, 2026-09-15: *"combining combos + the DNA concept (square vs
        circle) could be the right thing there."* A pattern is a shape DNA
        (see `cross`), so the child's properties — how far it reaches, how
        many turns it makes — are readable from the parents' before the VM
        sees it. That is what makes this a proposal rather than a guess:
        `_mutate` edits one parent at random; this one predicts what the
        child will be FOR, and says so in the rationale the library keeps."""
        if "combos" not in self.can:
            return None
        act = {n: e["pattern"] for n, e in self.library.active().items()
               if e["kind"] == "pattern" and e.get("pattern")}
        if len(act) < 2:
            return None
        names = sorted(act)
        for _ in range(8):
            a, b = self.rng.sample(names, 2)
            child = cross(act[a], act[b])
            if not child or pattern_name(child) in self.library:
                continue
            da, db, dc = pattern_dna(act[a]), pattern_dna(act[b]), pattern_dna(child)
            name = pattern_name(child)
            self._last_proposal = self.env.steps
            rationale = (f"crossed {a} ({act[a]}: reach {da['run']}, {da['turns']} turns) with "
                         f"{b} ({act[b]}: reach {db['run']}, {db['turns']} turns) -> {child}: "
                         f"reach {dc['run']}, {dc['turns']} turns over {dc['length']} steps")
            made = self._compose(name, why, list(child), child, "pattern", rationale)
            if made:
                self.library.entries[made]["reasoning"].update(
                    {"parents": [a, b], "dna": dc, "parent_dna": {a: da, b: db}})
                self._t("cross", child=made, pattern=child, parents=[a, b], dna=dc)
                self._think("modify", parent=f"{a}+{b}", child=made, edit="crossed")
            return made
        return None


    def _mutate(self, why: str) -> str | None:
        """MODIFY an existing program instead of composing a fresh one: pick an
        active pattern (the most valuable, or one that was legal but never
        paid), apply ONE edit — append a slot, drop a slot, or swap one — and
        compose the child with its lineage (parent, edit) in the reasoning.
        The parent stays (retire/consolidate decide its fate)."""
        if "combos" not in self.can:
            return None
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
        """Combine, then improve, then invent — in that order.

        CROSS two moves he owns (their DNAs predict the child's properties);
        failing that MODIFY one at random; failing that sample a fresh
        composition. Lengths are sampled from what has paid off; out-of-time
        asks for longer ones. The rationale is written into the program's
        reasoning trace either way."""
        if "combos" not in self.can:                     # the capability is not his yet
            return None
        retired = self.library.retire(self.env.steps)
        if retired:
            self._t("retire", name=retired, reason=self.library.entries[retired]["retired_reason"])
        n_active = sum(1 for e in self.library.active().values() if e["kind"] == "pattern")
        if n_active >= self.MAX_ACTIVE_PATTERNS:          # full: consolidate now, then edit rather than add
            for n in self.library.consolidate(self.env.level, keep=self.MAX_ACTIVE_PATTERNS - 2):
                self._t("retire", name=n, reason=self.library.entries[n]["retired_reason"])
        made = self._cross(why) or self._mutate(why)     # combine what he has before editing it
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
        if "combos" not in self.can or not self.library.entries:
            return exits                                 # a locked capability is absent, not merely unused
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
