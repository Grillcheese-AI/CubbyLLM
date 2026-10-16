"""ear — what a heard sentence is worth, good or bad, learned by contrast.

Wired: NOT YET in the serve path; BlockWorld's Cubby hears through it (`--hearing`).
Nothing in cubbyllm/ imports this.

ONE DIRECTION IN A HYPERVECTOR SPACE. A sentence is coded as the bundle of
its words, its word pairs in order, and its letter trigrams: bipolar codes
from a hash, so no table and no model are needed and French, Québécois and
English share the space. Pairs carry what words alone cannot ("not bad" is not
"bad", "pas mal" is praise), and a word that follows a negation is coded as
its negated self; trigrams let "chu tanné", "tannée" and a misspelling land
near each other.

CONTRAST IS THE TRAINING, AND IT IS ADDITION. One direction W in the space;
a sentence's valence is tanh(code . W). Teaching a sentence adds its code to
W by how wrong the ear was about it (the delta rule: a surprise-sized step,
nothing added for what it already hears right), so a hundred "good job"s do
not drown a rare "pas pire pantoute". A minimal pair, "good job, Cubby" /
"bad job, Cubby", is taught as both sentences plus their difference until the
good one hears better by a margin: what the two share cancels in the
difference, and only what differs points anywhere. A corpus of labelled
sentences is one pass kept as the mean of W over it (an averaged learner: its
last few hundred sentences do not get the last word); the pairs go on top. No gradient run, no epochs:
the no-retraining rule. Teaching more is adding more.

EXPERIENCE ADDS TO THE SAME DIRECTION. `learn(text, outcome)`: a sentence that
came before something good is added toward good, before a hurt toward bad,
by how much it mattered. A body that hears "careful!" before every fall comes
to hear "careful!" as bad news, and keeps that outside the world it learned it
in, because the ear is the same object everywhere.

WHAT IT MAY DO. This reads the sentence, not the hearer. What a valence is
allowed to do to Cubby is `afferent`'s call (valence does not transfer; warmth
aimed at him lands on `social`), and in a body it is a fact about the speaker
weighed by trust, never a reward: reward stays efferent.

WHAT THIS BRINGS TO CUBBYLLM. A sentence-level, bilingual good/bad read with
negation, that is inspectable (`why` names the words that made it good or
bad) and that keeps learning from consequences after it ships.

THE ADVANTAGE OVER A CONVENTIONAL MODEL. A sentiment classifier is a trained
artifact: fixed at its last run, opaque, and blind to what words came to mean
for its user. This is one vector you can read, grown by adding examples, and
changed by what actually happened after the words were said.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

__wiring__ = "BlockWorld --hearing"

DIM = 8192
TRI_W = 0.15          # letter trigrams count less than words: spelling, not meaning
PAIR_W = 0.8          # ... and word pairs a little less than words
SEM_W = 1.0           # the meaning part (a lexicon's word vectors), beside the hashed part
NEGATORS = {"not", "no", "never", "dont", "don't", "doesnt", "doesn't", "isnt", "isn't", "wasnt", "wasn't",
            "cant", "can't", "wont", "won't", "aint", "nothing", "nobody", "without",
            "pas", "jamais", "rien", "personne", "sans", "aucun", "aucune", "pantoute"}
YOU = {"you", "your", "you're", "youre", "yours", "u", "ur", "yourself", "cubby",
       "tu", "toi", "te", "ton", "ta", "tes", "vous", "votre", "vos", "t'es", "t'as", "t'", "té"}
_WORD = re.compile(r"[\w']+")


def _fold(s: str) -> str:
    """Lowercase; keep accents in words (é is not e in "tanné"), drop them only for trigrams."""
    return s.lower().replace("’", "'")


def _plain(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def features(text: str) -> list[tuple[str, float]]:
    """The sentence as weighted features: words (negated after a negator), word pairs, trigrams."""
    words = _WORD.findall(_fold(text))
    out, neg = [], 0
    for i, w in enumerate(words):
        base = w.split("'")[-1] if "'" in w and len(w.split("'")[-1]) > 2 else w    # "j'aime" -> "aime"
        out.append((("~" if neg else "") + base, 1.0))
        neg = 3 if base in NEGATORS or w.startswith("n'") else max(0, neg - 1)
        if i:
            out.append((words[i - 1] + " " + w, PAIR_W))
        p = f"<{_plain(base)}>"
        out += [("#" + p[k:k + 3], TRI_W) for k in range(len(p) - 2)]
    return out


def aimed(text: str) -> bool:
    """Said TO him (you, tu, vous, Cubby), not about somebody else."""
    return bool(set(_WORD.findall(_fold(text))) & YOU)


class Ear:
    LR = 0.3              # a step is this share of the surprise
    MARGIN = 0.6          # a pair is learned once the good one hears this much better

    def __init__(self, dim: int = DIM, lexicon=None, sem: int | None = None) -> None:
        self.dim = dim
        self.lexicon = lexicon                       # ear_lexicon.Lexicon, or None: hashed codes only
        self.sem = lexicon.dim if lexicon is not None else (sem or 0)
        self.w = np.zeros(dim + self.sem, np.float32)   # the good-minus-bad direction
        self.b = 0.0                                 # how good anything sounds before its words count
        self.taught = 0
        self._avg: list | None = None                # building: the running sum of W (an averaged learner)
        self.lived = 0
        self._codes: dict[str, np.ndarray] = {}

    # ── codes ───────────────────────────────────────────────────────────────
    def _fcode(self, f: str) -> np.ndarray:
        c = self._codes.get(f)
        if c is None:
            seed = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            c = np.random.default_rng(seed).integers(0, 2, self.dim, dtype=np.int8) * 2 - 1
            if len(self._codes) < 300_000:
                self._codes[f] = c
        return c

    def _parts(self, fs) -> tuple[np.ndarray, np.ndarray | None]:
        v = np.zeros(self.dim, np.float32)
        for f, wt in fs:
            v += wt * self._fcode(f)
        if not self.sem:
            return v, None
        s = np.zeros(self.sem, np.float32)
        for f, _ in fs:
            if self.lexicon is not None and not f.startswith("#") and " " not in f:
                vec = self.lexicon.word(f.lstrip("~"))
                if vec is not None:
                    s += -vec if f.startswith("~") else vec    # a negated word means its opposite
        return v, s

    def code(self, text: str) -> np.ndarray:
        v, s = self._parts(features(text))
        n = float(np.linalg.norm(v))
        v = v / n if n else v
        if s is not None:
            m = float(np.linalg.norm(s))
            v = np.concatenate([v, SEM_W * s / m if m else s]) / (np.sqrt(1 + SEM_W ** 2) if m else 1.0)
        return v.astype(np.float32)

    # ── teaching: everything is addition, sized by surprise ─────────────────
    def teach(self, text: str, valence: float, w: float = 1.0) -> float:
        """One labelled sentence, valence in [-1, 1]. Returns the surprise it caused."""
        c = self.code(text)
        err = float(np.clip(valence, -1, 1)) - float(np.tanh(c @ self.w + self.b))
        self.w += self.LR * w * err * c
        self.b += 0.05 * self.LR * w * err
        self.taught += 1
        self._sum()
        return err

    def _sum(self) -> None:
        if self._avg is not None:
            self._avg[0] += self.w
            self._avg[1] += self.b
            self._avg[2] += 1

    def averaging(self, on: bool) -> None:
        """While building from a corpus, keep the mean of W over every step (one pass, low variance);
        turning it off makes the mean the ear's W."""
        if on:
            self._avg = [np.zeros_like(self.w), 0.0, 0]
        elif self._avg is not None and self._avg[2]:
            self.w, self.b = (self._avg[0] / self._avg[2]).astype(np.float32), self._avg[1] / self._avg[2]
            self._avg = None

    def teach_pair(self, good: str, bad: str, w: float = 1.0) -> float:
        """A minimal pair: each to its side, then their difference until good hears better by MARGIN."""
        cg, cb = self.code(good), self.code(bad)
        self.teach(good, 1.0, w)
        self.teach(bad, -1.0, w)
        d = cg - cb
        gap = float(d @ self.w)
        if gap < self.MARGIN:
            self.w += self.LR * w * (self.MARGIN - gap) * d / max(float(d @ d), 1e-6)
            self._sum()
        self.taught -= 1
        return self.MARGIN - gap

    def learn(self, text: str, outcome: float, w: float = 0.5) -> float:
        """What came after the words, in the hearer's own body or world: + good, - bad."""
        if not outcome:
            return 0.0
        self.lived += 1
        self.taught -= 1
        return self.teach(text, float(np.clip(outcome, -1, 1)), w)

    # ── hearing ─────────────────────────────────────────────────────────────
    def valence(self, text: str) -> float:
        return float(np.tanh(float(self.code(text) @ self.w) + self.b))

    def hear(self, text: str) -> dict:
        """A heard sentence -> its valence, whether it was said to him, and the words behind it."""
        v = self.valence(text)
        return {"valence": round(v, 3), "aimed": aimed(text),
                "warmth": round(max(0.0, v), 3) if aimed(text) else 0.0, "why": self.why(text, 3)}

    def why(self, text: str, k: int = 4) -> list[tuple[str, float]]:
        """The words and pairs that pushed it most, signed: inspectable, not a verdict from nowhere."""
        fs = features(text)
        v, s = self._parts(fs)
        n = float(np.linalg.norm(v)) or 1.0
        m = float(np.linalg.norm(s)) if s is not None else 0.0
        k_ = np.sqrt(1 + SEM_W ** 2) if m else 1.0
        got: dict[str, float] = {}
        for f, wt in fs:
            if not f.startswith("#"):
                c = wt * float(self._fcode(f) @ self.w[:self.dim]) / n
                vec = self.lexicon.word(f.lstrip("~")) if (m and " " not in f) else None
                if vec is not None:
                    c += (-1 if f.startswith("~") else 1) * SEM_W * float(vec @ self.w[self.dim:]) / m
                got[f] = got.get(f, 0.0) + c / k_
        return sorted(((f, round(v, 3)) for f, v in got.items()), key=lambda x: -abs(x[1]))[:k]

    # ── keeping ─────────────────────────────────────────────────────────────
    def clone(self) -> "Ear":
        """A hearer's own copy: its own W to learn into, the lexicon shared (it never changes)."""
        e = Ear(self.dim, self.lexicon, self.sem)
        e.w, e.b, e.taught, e.lived, e._codes = self.w.copy(), self.b, self.taught, self.lived, self._codes
        return e

    # ── keeping ─────────────────────────────────────────────────────────────
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w, meta=np.array(json.dumps(
            {"dim": self.dim, "sem": self.sem, "lexicon": getattr(self.lexicon, "name", None),
             "b": self.b, "taught": self.taught, "lived": self.lived})))
        return path

    @classmethod
    def load(cls, path, lexicon=None) -> "Ear":
        """The ear as saved; without its lexicon the meaning part hears nothing, the rest still hears."""
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        sem = meta.get("sem", 0)
        ear = cls(meta["dim"], lexicon if lexicon is not None and lexicon.dim == sem else None, sem)
        ear.w = z["w"].astype(np.float32)
        ear.b, ear.taught, ear.lived = meta.get("b", 0.0), meta["taught"], meta.get("lived", 0)
        return ear


def from_pairs(pairs: list[dict], sentences: list[dict] | None = None, dim: int = DIM, seed: int = 0,
               lexicon=None) -> Ear:
    """An ear taught in one pass: labelled sentences ({"text", "valence"}) first, kept as the mean of
    W over them (a broad, low-variance start), then the minimal pairs ({"good", "bad"}) on top, each
    group in a fixed shuffled order. Pairs come last and are not averaged away: they say most."""
    rng = np.random.default_rng(seed)
    ear = Ear(dim, lexicon)
    ear.averaging(True)
    for i in rng.permutation(len(sentences or [])):
        s = sentences[i]
        ear.teach(s["text"], float(s["valence"]), float(s.get("w", 1.0)))
    ear.averaging(False)
    for i in rng.permutation(len(pairs)):
        p = pairs[i]
        ear.teach_pair(p["good"], p["bad"], float(p.get("w", 1.0)))
    return ear


__all__ = ["Ear", "features", "aimed", "from_pairs"]
