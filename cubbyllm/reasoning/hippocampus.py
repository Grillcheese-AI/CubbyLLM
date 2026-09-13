"""hippocampus -- the episodic side cortex: certified chains, recalled as plan candidates.

Wired: WIRED (2026-09-12; the GrillCheese hippocampal design brought over as a SIDE cortex).

What it is. An explicit episode store with one sparse binary code per episode. An
episode is what the reasoning loop already produces and the harvest already records:
the question, the plan that covered it, the chain the VM certified, the answer, and
the provenance (which store, which source, when). Recall is pattern completion from a
partial cue -- a new question, in any surface form -- to the episodes whose content
words it shares, and what comes back is a PLAN CANDIDATE: it goes through the
disposer (`covers()`), the walk and the VM exactly as an emitter's plan does. Memory
proposes; the host disposes; the VM is the only truth gate. Nothing is spoken because
memory remembered it.

The parts, named after what they do in the hippocampus:
  DG   `encode`: the content words of the question, the plan's relation labels and its
       seed (frame words out), each hashed to a 256-bit bipolar vector, bundled by
       majority and binarised -- SimHash over a bag of words, so two questions that
       differ only in their frame land at Hamming distance ~0 and two chains about
       different entities land apart (pattern separation by the words that differ).
  CA3  `recall`: the cue's code against every live episode's code, nearest first
       (a Hamming scan; 552k codes as Python ints is ~50 ms). `propose` completes the
       pattern into plans: the episode's own plan, and the episode's relation shape
       REBOUND to the entity the new question names (the chain shape remembered, the
       entity the question's) -- analogical transfer, and the disposer decides.
  consolidation  `consolidate(keep)`: the episode's FACTS live in the world store (they
       went through the gate); the hippocampus keeps the code, the plan, the
       provenance and a utility count, and retires the low-utility episodes beyond
       `keep` to the cold list. Retire, never delete.

Why an explicit store and not a weight memory: exp_a_forgetting, exp_a5_sparse and
exp_g2_nyt (2026-07-23) measured Hebbian, NLMS, SDM and DG-sparse storage at
D=1024..2048 -- every one loses recall between 512 and 1,024 associations, and the
world holds 552,297 facts. Codes are cheap and do not forget: 256 bits per episode,
18 MB for the whole world, inside the serve budget.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .plan_verify import _FRAME_AND_JOINT
from .planner import QuestionPlan, normalize

__wiring__ = Wiring.WIRED

N_BITS = 256


@dataclass
class Episode:
    question: str
    relations: list[str]              # hop 0 first, as the plan named them
    seed: str
    chain: list[str]                  # the facts the VM certified, in walk order
    answer: str
    provenance: dict = field(default_factory=dict)
    code: int = 0                     # the whole episode: question words + relations + seed
    shape: int = 0                    # the relation labels alone -- the chain's shape, entity-free
    utility: int = 0                  # recalls that ended in a verified answer
    retired: bool = False
    written_at: float = 0.0

    @property
    def plan(self) -> QuestionPlan:
        return QuestionPlan(relations=[None] + list(self.relations[1:]), tail=f"{self.relations[0]} of {self.seed}",
                            n_hop=len(self.relations))


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in _FRAME_AND_JOINT]


def _token_bits(token: str) -> int:
    """The surface code: a word's bits from its hash -- every word quasi-orthogonal to every other."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=N_BITS // 8).digest(), "big")


class WordBits:
    """The semantic code: a word's bits as the SimHash of its FastWordEncoder block vector
    (standin/data/build_word_table.py, from the MoWM axiom codebook -- the worlds' own block
    space), so 'born' and 'birth' share more bits than chance (163/256 measured) while names
    stay pattern-separated by the signal half of the blend; a word the table does not know
    falls back to its surface code. Loaded from a {word: hex} json; stdlib only."""

    def __init__(self, bits: dict[str, int], meta: dict | None = None) -> None:
        self.bits, self.meta = bits, dict(meta or {})

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "WordBits":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if int(d.get("n_bits", N_BITS)) != N_BITS:
            raise ValueError(f"word bits are {d.get('n_bits')} wide; the hippocampus uses {N_BITS}")
        return cls({w: int(h, 16) for w, h in d["bits"].items()}, {k: v for k, v in d.items() if k != "bits"})

    def __call__(self, token: str) -> int:
        b = self.bits.get(token)
        return _token_bits(token) if b is None else b

    def __len__(self) -> int:
        return len(self.bits)


def encode(words: list[str], word_bits=None) -> int:
    """DG: bundle the words' bipolar vectors by majority, binarise (ties -> 0). `word_bits`:
    the per-word code (default: the surface hash; a `WordBits` table for the semantic code)."""
    if not words:
        return 0
    wb = word_bits or _token_bits
    counts = [0] * N_BITS
    for w in dict.fromkeys(words):                 # a bag: each word once
        bits = wb(w)
        for j in range(N_BITS):
            counts[j] += 1 if (bits >> j) & 1 else -1
    code = 0
    for j in range(N_BITS):
        if counts[j] > 0:
            code |= 1 << j
    return code


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class Hippocampus:
    def __init__(self, word_bits=None) -> None:
        self.episodes: list[Episode] = []
        self.word_bits = word_bits            # None: the surface hash; a WordBits table: the semantic code

    # -- write ------------------------------------------------------------------
    def write(self, question: str, relations: list[str], seed: str, chain: list[str], answer: str,
              provenance: dict | None = None) -> Episode:
        words = content_words(question) + [w for r in relations for w in content_words(r)] + content_words(seed)
        ep = Episode(question=question, relations=[normalize(r) for r in relations], seed=normalize(seed),
                     chain=list(chain), answer=answer, provenance=dict(provenance or {}),
                     code=encode(words, self.word_bits), shape=encode([w for r in relations for w in content_words(r)], self.word_bits),
                     written_at=time.time())
        self.episodes.append(ep)
        return ep

    def __len__(self) -> int:
        return sum(1 for e in self.episodes if not e.retired)

    # -- recall (CA3) -----------------------------------------------------------
    def recall(self, question: str, k: int = 3, max_distance: int | None = None) -> list[tuple[Episode, int]]:
        """The k nearest live episodes to the question's content words, nearest first."""
        cue = encode(content_words(question), self.word_bits)
        scored = [(hamming(cue, e.code), i) for i, e in enumerate(self.episodes) if not e.retired]
        scored.sort()
        out = []
        for d, i in scored[:k]:
            if max_distance is not None and d > max_distance:
                break
            out.append((self.episodes[i], d))
        return out

    def recall_shape(self, question: str, k: int = 3) -> list[tuple[Episode, int]]:
        """The k nearest DISTINCT chain shapes (relation labels alone, entity-free) to the
        question -- the cue for analogical transfer: the same relations about another
        entity. One episode stands for each shape (the most useful).

        Scored word-by-word, not bundle-to-bundle: a shape is two or three words and a
        free-text question ten, so a majority bundle of the question drowns the shape
        (exp_r14 probe, 2026-09-12: `date of birth` was not in the top 8 for "in which year
        was X born"). Each shape word takes its best bit-agreement with any question word
        (the semantic DG makes born/birth 163 of 256, chance 128); the shape's score is the
        mean, and the distance reported is 256 minus it. The question's own wording of a
        relation is what `covers()` will demand; this is the same test, softened."""
        wb = self.word_bits or _token_bits
        cue_bits = [wb(w) for w in dict.fromkeys(content_words(question))]
        if not cue_bits:
            return []
        best: dict[tuple[str, ...], tuple[int, int, int]] = {}
        for i, e in enumerate(self.episodes):
            if e.retired:
                continue
            key = tuple(e.relations)
            if key in best and self.episodes[best[key][1]].utility >= e.utility:
                continue
            words = [w for r in e.relations for w in content_words(r)]
            if not words:
                continue
            agree = [max(N_BITS - hamming(wb(w), c) for c in cue_bits) for w in words]
            d = N_BITS - sum(agree) // len(agree)
            cur = best.get(key)
            if cur is None or (d, -e.utility) < (cur[0], -self.episodes[cur[1]].utility):
                best[key] = (d, i, e.utility)
        ranked = sorted(best.values())[:k]
        return [(self.episodes[i], d) for d, i, _u in ranked]

    def propose(self, question: str, k: int = 3, max_distance: int | None = None,
                rebind: bool = True) -> list[tuple[QuestionPlan, Episode, str]]:
        """Plan candidates for the question, in the order to try them: the recalled
        episodes' own plans ('recalled'), then -- when the question names a different
        entity -- the nearest chain SHAPES with the seed rebound to the question's
        residual entity ('rebound'). Every candidate is a proposal for the disposer; a
        wrong one dies at `covers()`, the walk or the VM."""
        q = normalize(question)
        out: list[tuple[QuestionPlan, Episode, str]] = []
        seen: set[tuple[tuple[str, ...], str]] = set()
        for ep, _d in self.recall(question, k=k, max_distance=max_distance):
            key = (tuple(ep.relations), ep.seed)
            if key not in seen:
                seen.add(key); out.append((ep.plan, ep, "recalled"))
        if rebind:
            for ep, _d in self.recall_shape(question, k=k):
                if ep.seed in q:
                    continue                                   # its own entity: the recalled plan above
                seed = residual_entity(question, ep.relations)
                if seed and (tuple(ep.relations), seed) not in seen:
                    seen.add((tuple(ep.relations), seed))
                    out.append((QuestionPlan(relations=[None] + list(ep.relations[1:]), tail=f"{ep.relations[0]} of {seed}",
                                             n_hop=len(ep.relations)), ep, "rebound"))
        return out

    def reinforce(self, ep: Episode) -> None:
        ep.utility += 1

    # -- consolidation ----------------------------------------------------------
    def consolidate(self, keep: int) -> list[Episode]:
        """Retire every live episode beyond the `keep` most useful (ties: most recent
        first). Retired episodes stay on the record and are never recalled; nothing is
        deleted. Returns what was retired."""
        live = [e for e in self.episodes if not e.retired]
        live.sort(key=lambda e: (e.utility, e.written_at), reverse=True)
        retired = live[keep:]
        for e in retired:
            e.retired = True
        return retired

    # -- persistence: one jsonl line per episode, the code as hex ----------------
    def save(self, path: pathlib.Path | str) -> int:
        p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.episodes:
                d = dict(e.__dict__); d["code"] = format(e.code, "x")
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return len(self.episodes)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Hippocampus":
        h = cls()
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); d["code"] = int(d["code"], 16)
                h.episodes.append(Episode(**d))
        return h


def residual_entity(question: str, relations: list[str]) -> str | None:
    """What is left of the question once the relations and the frame words are removed --
    the entity the question names, as the whole leftover span (never a fragment of it:
    'jean valjean' stays 'jean valjean', so a shorter entity's facts cannot be walked
    for it)."""
    body = normalize(question)
    for r in sorted((normalize(r) for r in relations), key=len, reverse=True):
        body = body.replace(r, " ", 1)
    words = [w for w in body.split() if w not in _FRAME_AND_JOINT]
    return " ".join(words) if words else None
