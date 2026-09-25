"""capsules -- the store's facts as capsules: where each came from, how it has been used, and a binary code
to find it by resemblance.

Wired: STANDALONE (a `FactStore` subclass; nothing on the serve path builds one yet).

A capsule is one fact plus what the host knows about it:

    world, source   which store holds it and who stated it (the store's `provenance` keeps the full string)
    time            when the fact was true (the store's `times`, unchanged)
    uses, verified  how often a walk used it, and how often that use was verified by the VM
    added, last     when it arrived and when it was last used
    code            `dim` bits: a majority vote over hashed features of its text -- the words and their
                    letter triples -- so two facts that share names share bits, and a misspelt name
                    ("Haslem") still lands next to the right one ("Haslam")

Exact lookup is untouched: `lookup` is FactStore's TripleIndex, and it is what a walk trusts. The code is
only for resemblance -- "facts about things like X", aliases, near-miss names -- and what it returns is a
fact and a score, never a vector: the symbolic boundary holds.

The code is computed with stdlib BLAKE2b under a fixed key, not with grilly2's hashing, which switches
digests when `blake3` is missing; a store written in one environment must read the same in every other.
grilly2 does the arithmetic: with it (`grilly.vsa.packed`), Hamming distance and top-k run on the GPU;
without it, numpy computes the same integers. Tested equal.

Kept from grilly v1's CapsuleMemory: the record and its use counters (its consolidation priority and
stability, here `uses`, `verified` and `priority`). Not kept: its 384->32 projection (a Linear, grilly2's
migration found) and its injection into transformer layers, which would put memory into the model's
hidden state without passing the gate.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import time
from functools import lru_cache

import numpy as np

from worlds import FactStore

__wiring__ = "STANDALONE"

KEY = b"cubby.capsule.v1"                 # changing this, or the features, makes stored codes unreadable
_TOKEN = re.compile(r"[\w']+")
_STOP = frozenset("the a an of in on at to for by with from as is was were are be been it its this that and or".split())
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def features(text: str) -> list[str]:
    """Words (minus a few function words) and each word's letter triples, marked so they cannot collide."""
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP] or _TOKEN.findall(text.lower())
    out = []
    for w in dict.fromkeys(words):
        out.append("w:" + w)
        padded = f"#{w}#"
        out += ["t:" + padded[i:i + 3] for i in range(len(padded) - 2)]
    return list(dict.fromkeys(out))


@lru_cache(maxsize=200_000)
def _bits(feature: str, dim: int) -> np.ndarray:
    out, c = bytearray(), 0
    while len(out) < dim // 8:
        out += hashlib.blake2b(f"{feature}\x00{c}".encode("utf-8"), digest_size=64, key=KEY).digest()
        c += 1
    return np.unpackbits(np.frombuffer(bytes(out[:dim // 8]), dtype=np.uint8), bitorder="little")


def encode(text: str, dim: int = 1024) -> np.ndarray:
    """`dim` bits as `dim // 32` int32 words (grilly2's packed layout): each bit the majority of the
    text's features, a tie going to 0 as in grilly2's bundle."""
    if dim <= 0 or dim % 32:
        raise ValueError(f"dim must be a positive multiple of 32, got {dim}")
    fs = features(text)
    if not fs:
        return np.zeros(dim // 32, dtype=np.int32)
    acc = np.zeros(dim, dtype=np.int32)
    for f in fs:
        acc += _bits(f, dim)
    bits = (2 * acc > len(fs)).astype(np.uint8)
    return np.packbits(bits, bitorder="little").view(np.int32).copy()


def hamming_numpy(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(codebook, query[None, :])
    return _POP[x.view(np.uint8)].sum(axis=1).astype(np.int32)


def _grilly_packed():
    try:
        from grilly.vsa import packed
        return packed
    except Exception:                                    # grilly v1, or no grilly: numpy computes the same
        return None


class CapsuleStore(FactStore):
    """A FactStore whose facts are capsules (see the module). `world` names the store the capsules live in;
    one world, one store, so a plugin's capsules can never surface in the host's recall."""

    def __init__(self, texts: list[str] | None = None, enc=None, name: str = "facts", dim: int = 1024,
                 gpu: bool | None = None) -> None:
        self.dim = dim
        self.meta: dict[str, dict] = {}
        self._codes: list[np.ndarray] = []
        self._book: np.ndarray | None = None
        self._gbook = None
        self._packed = _grilly_packed() if gpu is not False else None
        if gpu and self._packed is None:
            raise RuntimeError("gpu=True needs grilly2 (grilly.vsa.packed)")
        super().__init__(texts=texts, enc=enc, name=name)

    # -- growth ---------------------------------------------------------------------------------------
    def add(self, text: str, source: str | None = None, when: float | None = None) -> bool:
        if not super().add(text):
            return False
        key = self._key(text)
        now = time.time() if when is None else when
        self.meta[key] = {"world": self.name, "source": source, "added": now, "last": None, "uses": 0, "verified": 0}
        self._codes.append(encode(key, self.dim))
        self._book = self._gbook = None
        return True

    def codebook(self) -> np.ndarray:
        if self._book is None:
            self._book = (np.stack(self._codes) if self._codes
                          else np.zeros((0, self.dim // 32), dtype=np.int32))
        return self._book

    # -- resemblance ----------------------------------------------------------------------------------
    def similar(self, query: str, k: int = 5, min_similarity: float = 0.0) -> list[tuple[float, str]]:
        """The `k` capsules whose codes are nearest the query's: (similarity, fact), nearest first.
        1.0 is the same features; 0.5 is unrelated."""
        if not self.texts:
            return []
        q = encode(query, self.dim)
        k = min(k, len(self.texts))
        if self._packed is not None:
            import grilly
            if self._gbook is None:
                self._gbook = grilly.from_numpy(self.codebook())
            idx, dist, _ = self._packed.topk(grilly.from_numpy(q), self._gbook, k=k, dim=self.dim)
            idx, dist = idx.numpy().tolist(), dist.numpy().tolist()
        else:
            d = hamming_numpy(q, self.codebook())
            idx = np.argsort(d, kind="stable")[:k].tolist()
            dist = d[idx].tolist()
        out = [(1.0 - di / self.dim, self.texts[i]) for i, di in zip(idx, dist)]
        return [(s, t) for s, t in out if s >= min_similarity]

    # -- use --------------------------------------------------------------------------------------------
    def touch(self, text: str, verified: bool = False, when: float | None = None) -> None:
        """A walk used this fact; `verified` when the VM verified that use."""
        m = self.meta.get(self._key(text))
        if m is None:
            return
        m["uses"] += 1
        m["verified"] += bool(verified)
        m["last"] = time.time() if when is None else when

    def priority(self, text: str, now: float | None = None, half_life_days: float = 30.0) -> float:
        """How much the sleep cycle should want to keep it: verified uses count three, any use one, and
        the whole halves every `half_life_days` since it was last used (or added)."""
        m = self.meta[self._key(text)]
        now = time.time() if now is None else now
        age = max(0.0, now - (m["last"] or m["added"])) / 86400.0
        return (1 + m["uses"] + 2 * m["verified"]) * 0.5 ** (age / half_life_days)

    def candidates(self, k: int, now: float | None = None) -> list[str]:
        """The `k` highest-priority capsules that have at least one VERIFIED use: what the sleep cycle may
        consider for its durable log. Its own rules (a trusted source, no contradiction) still decide."""
        ok = [t for t in self.texts if self.meta[t]["verified"] > 0]
        return sorted(ok, key=lambda t: -self.priority(t, now))[:k]

    # -- persistence ------------------------------------------------------------------------------------
    def save(self, folder: str | pathlib.Path) -> None:
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "capsules.jsonl").open("w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(json.dumps({"fact": t, **self.meta[t], "time": self.times.get(t),
                                    "provenance": self.provenance.get(t)}, ensure_ascii=False) + "\n")
        np.save(folder / "codes.npy", self.codebook())
        (folder / "capsules.meta.json").write_text(json.dumps(
            {"world": self.name, "dim": self.dim, "n": len(self.texts), "key": KEY.decode(),
             "features": "words + letter triples, blake2b-64, majority"}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | pathlib.Path, enc=None, gpu: bool | None = None) -> "CapsuleStore":
        folder = pathlib.Path(folder)
        head = json.loads((folder / "capsules.meta.json").read_text(encoding="utf-8"))
        store = cls(name=head["world"], dim=head["dim"], enc=enc, gpu=gpu)
        for line in (folder / "capsules.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            store.add(r["fact"], source=r.get("source"), when=r.get("added"))
            key = store._key(r["fact"])
            store.meta[key].update({k: r[k] for k in ("last", "uses", "verified")})
            if r.get("time"):
                store.times[key] = r["time"]
            if r.get("provenance"):
                store.provenance[key] = r["provenance"]
        codes = np.load(folder / "codes.npy")
        if codes.shape != store.codebook().shape or not np.array_equal(codes, store.codebook()):
            raise ValueError(f"{folder}: stored codes differ from recomputed ones -- the key or the features changed")
        return store
