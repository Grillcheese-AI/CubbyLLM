"""school -- the emitter learns arithmetic the way a child does: small concepts first, tries until it gets it
right, fewer tries as it masters a level, harder problems after, and dopamine on every verified solve.

Wired: STANDALONE (2026-09-30; the curriculum's state and signals. The sampler, the VM and the weight update are
the runner's -- `validation/train_school_torch.py` -- so this module holds no model and no torch).

The owner, 2026-09-30: "like a human does not start with rocket science algebra, we mix concepts we learned at
elementary school to get there... so the model can try until it gets it right, then the amount of tryouts are
reduced over time to challenge it more and difficulty increases. Each time it solves one right dopamine gets up
as a reward." And: "arithmetic should improve over time, the model needs to learn from its mistakes."

LEVELS  ordered by what a problem needs (`import_tinygsm.difficulty`): one operation; two steps; three or four;
        numbers in words and unit conversions (the arithmetic world's $K); distractor numbers; five or six
        steps; seven to ten. A batch is mostly the current level plus a REVIEW share of mastered ones, so an
        old skill that slips is caught.
TRIES   a problem is attempted up to the level's BUDGET (sampled programs, run in the VM, stopped at the first
        correct one). The budget starts at `max_tries`; when the level's solve rate over a window clears
        `master_at`, it halves (16 -> 8 -> 4 -> 2 -> 1); at one try and still clearing it, the level is
        mastered and the next opens with the full budget again. Below `struggle_at` the budget doubles back, and
        at full budget a struggling level sends a share of the batch back to the level before (a teacher's
        review, not a demotion).
DOPAMINE the reward-prediction error of the striatum (`striatum.py`, the same honest rule): per level, an
        expected reward; a problem's reward is +1 when solved within budget, 0 when not; the dopamine is
        `reward - expected`, which then moves the expectation by `alpha`. A first solve of a hard problem is a
        large burst; a routine solve at a mastered level is ~0 -- boredom, the signal to move on; a failure
        where success was expected is a DIP. Dopamine is the error, never the reward, and never a model's
        opinion: only the VM against a known answer says "solved".
MISTAKES every wrong attempt before the solve is kept with the solve: a (wrong, right) pair for the same prompt,
        with the FIRST WRONG STEP located by the programs' step values (`arith_step_values`, whose last value
        is the VM's): the first step of the wrong program whose value the right program never computes.
        Training rows carry the dopamine as their weight: the solve is reinforced by how surprising it was,
        the wrong program is pushed down by the dip, and a refusal (a program that does not run, a copied
        literal) weighs 0 -- a "don't know" is never punished, or the loop learns to guess.
WHY     the owner, 2026-09-30: the frontier models' math came over generations, not at once; a model trained to
        be good at everything at every level at once answers past what it knows -- false positives. So a
        level opens only after the one below is mastered, and the mastery map is the serve-time competence
        boundary (`speak_policy`): above it the emitter is not asked to guess.
MEMORY  every solve is written to the arithmetic world as an attested procedure (`arith_world.remember`), so it
        is recallable on the next question before any retrain; the retrain consolidates it into the weights.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
from collections import deque
from dataclasses import dataclass, field

from ..core.protocols import Wiring

__wiring__ = Wiring.STANDALONE


@dataclass(frozen=True)
class Level:
    name: str
    min_steps: int
    max_steps: int
    needs: str = ""            # "" | "words_or_units" | "distractors"

    def admits(self, d: dict) -> bool:
        if not self.min_steps <= int(d.get("steps", 0)) <= self.max_steps:
            return False
        if self.needs == "words_or_units":
            return d.get("word_numbers", 0) > 0 or d.get("unit_constants", 0) > 0
        if self.needs == "distractors":
            return d.get("distractors", 0) > 0
        return d.get("word_numbers", 0) == 0 and d.get("unit_constants", 0) == 0 and d.get("distractors", 0) == 0


LEVELS: tuple[Level, ...] = (
    Level("one operation", 1, 1),
    Level("two steps", 2, 2),
    Level("three or four steps", 3, 4),
    Level("words and units", 1, 4, "words_or_units"),
    Level("distractors", 1, 5, "distractors"),
    Level("five or six steps", 5, 6),
    Level("seven to ten steps", 7, 10),
)


def level_of(d: dict, levels: tuple[Level, ...] = LEVELS) -> int | None:
    """The first level that admits a problem's difficulty; the long levels also take problems with words,
    units or distractors (the hard end mixes everything)."""
    for i, lv in enumerate(levels):
        if lv.admits(d):
            return i
    steps = int(d.get("steps", 0))
    for i, lv in enumerate(levels):
        if lv.min_steps <= steps <= lv.max_steps and lv.min_steps >= 5:
            return i
    return None


# an attempt's outcome, as the VM and the slot check say it
CORRECT, WRONG, REFUSED = "correct", "wrong", "refused"


@dataclass
class LevelState:
    budget: int
    expected: float = 0.0                          # the striatum's expectation for this level
    window: deque = field(default_factory=lambda: deque(maxlen=64))   # solved-within-budget, recent problems
    seen: int = 0
    solved: int = 0
    mastered: bool = False

    @property
    def rate(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0


@dataclass
class Curriculum:
    levels: tuple[Level, ...] = LEVELS
    max_tries: int = 16
    master_at: float = 0.8           # solve rate over a full window that halves the budget
    struggle_at: float = 0.3         # below this over a full window, the budget doubles back
    window: int = 64
    review: float = 0.2              # share of a batch drawn from mastered levels
    alpha: float = 0.1               # how fast the expectation follows the rewards
    seed: int = 0
    # the ordering control (2026-09-30): every level open from the start and drawn uniformly -- the same tries,
    # budgets, dopamine and mistake rows, no curriculum. The literature finds curricula add little (on
    # transformers, one pass, no tries); the owner expects the recurrent pupil in this loop to differ. The
    # difference between the two runs is the ordering's effect alone.
    open_all: bool = False
    current: int = 0
    state: list[LevelState] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)   # every level change and budget change, on the record
    patterns: dict = field(default_factory=dict)      # "level|error kind" -> count (gsm8k.cube's `patterns`)
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.state:
            self.state = [LevelState(self.max_tries, window=deque(maxlen=self.window)) for _ in self.levels]

    # -- what to practise ----------------------------------------------------------
    def plan_batch(self, n: int) -> list[int]:
        """The level of each of the batch's n problems: the current level, a review share from mastered
        levels, and when the current level struggles at full budget, extra practice on the level before.
        With `open_all`: every level, uniformly (the no-curriculum control)."""
        if self.open_all:
            return [self._rng.randrange(len(self.levels)) for _ in range(n)]
        cur = self.state[self.current]
        back = self.current > 0 and cur.budget >= self.max_tries and len(cur.window) == self.window \
            and cur.rate < self.struggle_at
        out = []
        for _ in range(n):
            r = self._rng.random()
            done = [i for i in range(self.current) if self.state[i].mastered]
            if done and r < self.review:
                out.append(self._rng.choice(done))
            elif back and r < self.review + 0.3:
                out.append(self.current - 1)
            else:
                out.append(self.current)
        return out

    def budget(self, level: int) -> int:
        return self.state[level].budget

    # -- what happened ---------------------------------------------------------------
    def record(self, level: int, outcomes: list[str]) -> dict:
        """One problem's attempts, in order, as the VM judged them (stopped at the first CORRECT, at most the
        level's budget). Returns the dopamine signal and each attempt's training weight."""
        st = self.state[level]
        solved = CORRECT in outcomes[:st.budget]
        reward = 1.0 if solved else 0.0
        dopamine = reward - st.expected
        st.expected += self.alpha * dopamine
        st.seen += 1
        st.solved += int(solved)
        st.window.append(1 if solved else 0)
        # a solve is reinforced by its surprise; a wrong attempt is pushed down by the surprise plus how sure the
        # level was (failing what is expected hurts more); a refusal weighs nothing
        weights = [max(dopamine, 0.0) if o == CORRECT else (-(abs(dopamine) + 0.5 * st.expected) if o == WRONG else 0.0)
                   for o in outcomes]
        event = self._adapt(level)
        return {"level": level, "solved": solved, "tries": (outcomes.index(CORRECT) + 1) if solved else len(outcomes),
                "reward": reward, "expected": round(st.expected, 4), "dopamine": round(dopamine, 4),
                "weights": [round(w, 4) for w in weights], "event": event}

    def _adapt(self, level: int) -> str:
        st = self.state[level]
        if len(st.window) < self.window:
            return ""
        if st.rate >= self.master_at:
            if st.budget > 1:
                st.budget = max(1, st.budget // 2)
                st.window.clear()
                return self._log(level, f"budget -> {st.budget}")
            if not st.mastered:
                st.mastered = True
                if level == self.current and self.current + 1 < len(self.levels):
                    self.current += 1
                    return self._log(level, f"mastered; next level: {self.levels[self.current].name}")
                return self._log(level, "mastered")
        elif st.rate < self.struggle_at and st.budget < self.max_tries:
            st.budget = min(self.max_tries, st.budget * 2)
            st.window.clear()
            return self._log(level, f"struggling; budget -> {st.budget}")
        elif st.mastered and st.rate < self.master_at - 0.2:
            st.mastered = False                    # a review found the skill slipping: practise it again
            return self._log(level, "slipped; back in practice")
        return ""

    def _log(self, level: int, what: str) -> str:
        self.trace.append({"level": level, "name": self.levels[level].name, "event": what,
                           "rate": round(self.state[level].rate, 3), "seen": self.state[level].seen})
        return what

    def learn(self, level: int, error: str) -> bool:
        """gsm8k.cube's `learn()`: count a mistake's kind at its level; True (and a PatternLearned trace row) the
        first time a kind shows up there, so the report says which mistakes each level teaches."""
        key = f"{level}|{error}"
        new = key not in self.patterns
        self.patterns[key] = self.patterns.get(key, 0) + 1
        if new:
            self._log(level, f"pattern learned: {error}")
        return new

    # -- the competence boundary: what the emitter may answer at serve ---------------
    # The owner, 2026-09-30: "one of the main mistakes we make with models is that we want them to be good at
    # everything at every level at once... then you get a lot of false positives / hallucinations." So the
    # school's map of what is mastered is also what the host lets the emitter answer: a mastered level speaks
    # on one VM-verified program; a level still in practice speaks only when a majority of k sampled programs
    # agree in the VM; a level not yet reached is refused (or handed to another proposer) -- never guessed.
    def competence(self, level: int | None) -> str:
        if level is None:
            return "not yet"
        st = self.state[level]
        if st.mastered:
            return "mastered"
        return "practising" if level <= self.current or self.open_all else "not yet"

    def speak_policy(self, level: int | None, k_practising: int = 8) -> dict:
        """How the host may answer a question of this level: tries to sample, the share that must agree, or refuse."""
        c = self.competence(level)
        if c == "mastered":
            return {"competence": c, "samples": 1, "agree": 1.0, "speak": True}
        if c == "practising":
            return {"competence": c, "samples": k_practising, "agree": 0.5, "speak": True}
        return {"competence": c, "samples": 0, "agree": 1.0, "speak": False}

    # -- the tonic level: is there anything left to learn here ----------------------
    def tonic(self, level: int) -> float:
        """Mean reward at the level over its window: near 1 at a mastered level (routine solves, no surprise)."""
        return self.state[level].rate

    # -- persistence -------------------------------------------------------------------
    def save(self, path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"current": self.current, "max_tries": self.max_tries, "trace": self.trace,
                                 "patterns": self.patterns,
                                 "state": [{"budget": s.budget, "expected": s.expected, "window": list(s.window),
                                            "seen": s.seen, "solved": s.solved, "mastered": s.mastered}
                                           for s in self.state]}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path, **kw) -> "Curriculum":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        c = cls(max_tries=d["max_tries"], **kw)
        c.current, c.trace, c.patterns = d["current"], d["trace"], dict(d.get("patterns", {}))
        for s, x in zip(c.state, d["state"]):
            s.budget, s.expected, s.seen, s.solved, s.mastered = x["budget"], x["expected"], x["seen"], x["solved"], x["mastered"]
            s.window.extend(x["window"])
        return c


# -- mistakes -----------------------------------------------------------------------

def first_wrong_step(wrong_values: list[float], right_values: list[float], tol: float = 1e-6) -> int | None:
    """The first step of the wrong program whose value the right program never computes -- where it went off
    the road. None when every step's value is one the right program also reaches (the error is at the end)."""
    for k, v in enumerate(wrong_values):
        if not any(abs(v - r) <= tol * max(1.0, abs(r)) for r in right_values):
            return k
    return None


def mistake_rows(prompt: str, attempts: list[dict], result: dict, reference: dict | None = None) -> list[dict]:
    """Training rows from one problem: the solve (weighted by its dopamine) and every wrong attempt before it,
    paired with the solve and its first wrong step. `attempts`: [{program, outcome, values, stated}] in order.
    Not solved within the budget and a `reference` given ({program, values}: the dataset's verified program):
    the solution is SHOWN -- a row of its own, weighted by the dip or by what the level has left to master,
    whichever is larger -- and the wrong attempts pair with it, the way a pupil who could not get there learns
    from the worked answer."""
    rows = []
    right = next((a for a in attempts if a["outcome"] == CORRECT), None)
    shown = 0.0
    if right is None and reference is not None:
        right = reference
        # as much to learn as the level has left to master: a new level (expected ~0) learns fully from the
        # worked answer even though failing it was no surprise
        shown = round(max(abs(result["dopamine"]), 1.0 - result["expected"]), 4)
        rows.append({"prompt": prompt, "program": reference["program"], "weight": shown,
                     "kind": "shown", "level": result["level"], "dopamine": result["dopamine"]})
    for a, w in zip(attempts, result["weights"]):
        if a["outcome"] == WRONG and shown:
            w = -max(abs(w), shown)
        if a["outcome"] == CORRECT:
            rows.append({"prompt": prompt, "program": a["program"], "weight": w, "kind": "solve",
                         "level": result["level"], "dopamine": result["dopamine"]})
        elif a["outcome"] == WRONG and right is not None:
            wv, rv = a.get("values") or [], right.get("values") or []
            rows.append({"prompt": prompt, "rejected": a["program"], "chosen": right["program"], "weight": w,
                         "kind": "mistake", "level": result["level"], "dopamine": result["dopamine"],
                         "first_wrong_step": first_wrong_step(wv, rv),
                         "error": classify_error(wv[-1], rv[-1], a.get("stated"), rv, wv) if wv and rv else "other"})
    return rows


# -- what KIND of mistake: gsm8k.cube's ISolverLearn.learn(), made to run -------------
# `cubelang/examples/gsm8k.cube` (the owner's earlier GSM8K solver) declares the loop this module runs: `verify`
# checks the answer, and on a failure `learn(input, expected, actual)` computes `classify_error(input, diff)`,
# keeps a handler per error key in the program's `patterns`, and emits PatternLearned. That file only parses
# (cubelang docs/DRIFT.md, "Bugs found" #1); the classification lives here, on the values the VM computed.

UNIT_FACTORS = (60.0, 24.0, 7.0, 12.0, 100.0, 1000.0, 52.0, 365.0, 30.0, 16.0, 3600.0)


def classify_error(wrong: float, gold: float, stated: list[float] | None = None,
                   right_values: list[float] | None = None, wrong_values: list[float] | None = None,
                   tol: float = 1e-6) -> str:
    """The error key of a wrong answer, from values alone:
      stopped early     the answer is one of the right program's intermediate values
      reversed          the answer (or its first wrong step) is a right value negated or inverted
      unit conversion   off by a factor the arithmetic world holds (60, 7, 100, ...)
      one number off    off by a number the question states (added / multiplied one too many or too few)
      other             none of these"""
    def eq(a, b):
        return abs(a - b) <= tol * max(1.0, abs(b))
    rv, wv, st = right_values or [], wrong_values or [], stated or []
    if any(eq(wrong, r) for r in rv[:-1]):
        return "stopped early"
    for v in [wrong] + wv:
        if any(eq(v, -r) for r in rv if r) or any(v and eq(1.0 / v, r) for r in rv if r):
            return "reversed"
    if gold and wrong:
        ratio = wrong / gold
        if any(eq(ratio, f) or eq(ratio, 1.0 / f) for f in UNIT_FACTORS):
            return "unit conversion"
        if any(n and (eq(abs(wrong - gold), abs(n)) or eq(ratio, n) or eq(ratio, 1.0 / n)) for n in st):
            return "one number off"
    return "other"


# -- the step values of an arithmetic program (for the first wrong step) ---------------
_ARITH_LINE = re.compile(r"^\s*(create|assign|add|sub|mul|div)\s+(s\d+)\s*(?::\s*\w+|=\s*([^;]+)|,\s*([^;]+))\s*;",
                         re.M)


def arith_step_values(program: str) -> list[float] | None:
    """Each step register's value, in creation order, for a program of the arithmetic dialect (create / assign /
    add / sub / mul / div); None when it is not that dialect or divides by zero. The final value is the one
    the VM returns (pinned against it); the earlier ones are what `first_wrong_step` compares."""
    from .slots import num_value
    body = program.split("public function solve", 1)[-1]
    regs: dict[str, float] = {}
    order: list[str] = []
    for m in _ARITH_LINE.finditer(body):
        op, reg = m.group(1), m.group(2)
        if op == "create":
            regs[reg] = 0.0
            order.append(reg)
            continue
        rhs = (m.group(3) or m.group(4) or "").strip()
        if reg not in regs:
            return None
        v = regs[rhs] if rhs in regs else num_value(rhs)
        if v is None:
            return None
        if op == "assign":
            regs[reg] = v
        elif op == "add":
            regs[reg] += v
        elif op == "sub":
            regs[reg] -= v
        elif op == "mul":
            regs[reg] *= v
        else:
            if v == 0:
                return None
            regs[reg] /= v
    return [regs[r] for r in order] or None


# -- reward hacking: an answer written in is not an answer computed ----------------------
# "Agentic World Modeling" (arXiv 2604.22748, §5, L3 evolvers) names the failure modes of self-revision:
# confirmation bias, catastrophic forgetting, reward hacking, blame assignment. Here the reward is the VM against
# the gold, and the one cheap hack is a program that writes the answer as a literal (a pool problem the SFT data
# already showed). The judge refuses it, so it earns nothing -- and a refusal is never punished either.

def answer_written_in(program: str, gold, stated=(), sources=("$N",)) -> str:
    """Why this slotted program does not count as solving, or "" when it may: it uses no number from the
    question, or it writes the gold value itself as a literal (and the question did not state it).
    `sources`: the slot kinds that count as "from the question". A step of the write-back loop passes
    ("$N", "$S"): a later step that combines only earlier steps' VM values ($S) is chaining, not a hack."""
    from .slots import NUM_LIT_RX, num_value
    if not any(s in program for s in sources):
        return "no number from the question"
    g = num_value(str(gold)) if gold is not None else None
    if g is None or any(s is not None and abs(s - g) <= 1e-9 * max(1.0, abs(g)) for s in stated):
        return ""
    for m in NUM_LIT_RX.finditer(program):
        v = num_value(m.group("num"))
        if v is not None and abs(v - g) <= 1e-9 * max(1.0, abs(g)) and v != 0:
            return "the answer written in, not computed"
    return ""