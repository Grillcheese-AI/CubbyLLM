"""striatum -- proposer arbitration by reward prediction error (the basal ganglia's job, with an honest reward).

Wired: WIRED (2026-09-12; GrillCheese alpha's basal ganglia / dopamine brought over with the VM as the reward).

Four proposers can put a plan in front of the gate -- the grammar, the hippocampus, the emitter,
a frontier model in probes -- and until now they were tried in a fixed order. The striatum keeps,
per (proposer, question shape), an EXPECTED value of trying that proposer, and orders the
proposers best-first. After each try it receives the gate's outcome and computes the dopamine
signal: the prediction error delta = reward - expected, which moves the expectation by alpha *
delta. Dopamine is the error, not the reward; the reward is the gate's.

The reward, and the two constraints that keep the loop honest:
    certified (VM-verified, spoken)     +1.0
    refused / no plan / walk failed      0.0   -- NEVER negative: a "don't know" that costs
                                                 anything teaches the loop to guess
    wrong (a spoken answer audited false)  -5.0 -- far below -1, so the expected value of
                                                 guessing stays below the value of refusing
The reward comes from the gate (or an audit against gold, or a user's explicit confirmation);
never from a model's opinion of its own answer (no self-play, no judge). Every delta is on the
record (`trace`).

Exploration: the TONIC level is the running mean of recent rewards; when it is low the striatum
explores -- with probability eps it tries the second-best proposer first -- so a shape whose
best proposer stopped certifying gets the others a hearing. Seeded, so runs reproduce.

What arbitration buys is not more certified answers -- every proposer is still tried until one
certifies -- but fewer wasted proposals: an emitter call is ~1 s on the GPU, a hippocampus
recall ~50 ms, the grammar ~0; the striatum learns which to ask first, per shape.
"""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .plan_verify import ask_type
from .planner import normalize

__wiring__ = Wiring.WIRED

REWARD = {"certified": 1.0, "refused": 0.0, "wrong": -5.0}


def question_shape(question: str) -> str:
    """A coarse, deterministic shape: the ask's kind, the lead word, and the possessive /
    inversion frames -- enough for the expectation to differ where the proposers differ."""
    q = normalize(question)
    words = q.split()
    lead = words[0] if words else ""
    kind = ask_type(question) or "any"
    frames = []
    if " s " in f" {q} " or "--" in question or " its " in f" {q} ":
        frames.append("possessive")
    if lead in ("what", "which") and len(words) > 2 and words[2] in ("does", "do", "did", "is", "was"):
        frames.append("inversion")
    if " of the " in q:
        frames.append("chain")
    return f"{kind}|{lead}|{'+'.join(frames) or 'flat'}"


@dataclass
class Striatum:
    alpha: float = 0.2                   # learning rate on the expectation
    eps: float = 0.1                     # exploration when the tonic level is low
    tonic_floor: float = 0.2             # below this mean reward, explore
    seed: int = 0
    expected: dict[str, float] = field(default_factory=dict)      # "proposer|shape" -> expected reward
    n: dict[str, int] = field(default_factory=dict)
    recent: list[float] = field(default_factory=list)             # the last rewards seen (tonic)
    trace: list[dict] = field(default_factory=list)               # every dopamine signal, on the record
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @staticmethod
    def key(proposer: str, shape: str) -> str:
        return f"{proposer}|{shape}"

    def value(self, proposer: str, shape: str) -> float:
        return self.expected.get(self.key(proposer, shape), 0.0)

    @property
    def tonic(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def order(self, question: str, proposers: list[str]) -> list[str]:
        """The proposers best-first for this question's shape; a proposer never tried on the
        shape ranks by its value on all shapes (its prior); explore when the tonic level is low."""
        shape = question_shape(question)
        def score(p: str) -> float:
            k = self.key(p, shape)
            if k in self.expected:
                return self.expected[k]
            seen = [v for kk, v in self.expected.items() if kk.startswith(p + "|")]
            return sum(seen) / len(seen) if seen else 0.0
        ranked = sorted(proposers, key=lambda p: (-score(p), proposers.index(p)))
        if len(ranked) > 1 and self.tonic < self.tonic_floor and self._rng.random() < self.eps:
            ranked[0], ranked[1] = ranked[1], ranked[0]
        return ranked

    def reward(self, question: str, proposer: str, outcome: str, *, source: str = "gate") -> float:
        """Deliver the gate's outcome for one try; returns the dopamine signal (delta).
        `outcome` is one of REWARD's keys; `source` names who said so (gate | audit | user)."""
        r = REWARD[outcome]
        shape = question_shape(question); k = self.key(proposer, shape)
        v = self.expected.get(k, 0.0)
        delta = r - v
        self.expected[k] = v + self.alpha * delta
        self.n[k] = self.n.get(k, 0) + 1
        self.recent.append(r); del self.recent[:-50]
        self.trace.append({"proposer": proposer, "shape": shape, "outcome": outcome, "source": source,
                           "reward": r, "expected": round(v, 4), "delta": round(delta, 4), "tonic": round(self.tonic, 4)})
        return delta

    # -- persistence ----------------------------------------------------------------
    def save(self, path: pathlib.Path | str) -> None:
        p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"alpha": self.alpha, "eps": self.eps, "tonic_floor": self.tonic_floor, "seed": self.seed,
                                 "expected": self.expected, "n": self.n, "recent": self.recent}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Striatum":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        s = cls(alpha=d["alpha"], eps=d["eps"], tonic_floor=d["tonic_floor"], seed=d["seed"])
        s.expected = dict(d["expected"]); s.n = dict(d["n"]); s.recent = list(d["recent"])
        return s
