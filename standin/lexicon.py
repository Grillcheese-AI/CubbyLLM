"""lexicon — words whose affect is LEARNED from what happened next.

Wired: WIRED (stand-in game path, via `coach.Coach`). Nothing in cubbyllm/
imports this.

Ported from GrillCheese's `brain/amygdala.py` per the port-don't-link rule,
and deliberately only PART of it. Owner: *"it was working well with the
amygdala and snn in grillcheese maybe lets just reuse that?"* — so here is
what was taken, and what was left, and why.

TAKEN, because it is the mechanism this needed and nothing here had:

    _token_affect_ema      every token carries a running (valence, arousal,
                           threat) that moves toward whatever actually
                           happened, with momentum 0.20 while the word is new
                           and 0.08 once it has been seen 16 times — fast to
                           learn, slow to be talked out of it.
    the MIGRATING sets     a token that settles strongly positive joins the
                           positive set; strongly negative, the negative one;
                           negative-and-activating, the threat set. The
                           vocabulary grows itself.
    the stemmer            cheap suffix trimming so "ghosts" and "ghost" are
                           one word.

LEFT BEHIND, and this is not a shortcut:

    AmygdalaAffectNet      a 3-layer MLP over token vectors that are
                           `blake2b`-seeded RANDOM unit vectors. Random
                           vectors carry no semantics, so the network cannot
                           generalise to a word it has not seen — it can only
                           memorise, which is what the EMA above already does,
                           more cheaply and legibly. Its two outputs then meet
                           the four lexical features in a 2x6 linear head, and
                           that head is where the work actually happens.
    grilly / numpy         a GPU backend for a 2x6 matrix multiply. The
                           stand-in stays dependency-free.
    the SNN / STDP         NOT REJECTED — deferred, and the reason I first
                           gave for skipping it was wrong. `stdp` appears in
                           `pipelines/train_pipeline.py`, so I read it as a
                           training path colliding with *"the whole concept of
                           cubbyllm is no-retraining"*. Owner: *"stdp was live
                           not at train time."* That makes it the opposite of
                           a collision: live synaptic plasticity is host-side
                           state changing during a session, which is exactly
                           this project's thesis and exactly the missing
                           middle timescale — faster than a corpus, slower
                           than an EMA over one conversation. It wants its own
                           read of the live path rather than a guess, and it
                           is the obvious next port after this one.
    the word lists         `sev1`, `outage`, `incident` — an ops assistant's
                           threat vocabulary. Wrong world. The mechanism ports;
                           the tables do not.

WHY THIS MATTERS MORE HERE THAN IT DID THERE, and it is the whole reason to
bother: GrillCheese had to be TOLD the target (`train_step(text,
target_emotion)`). In the maze nobody has to tell it anything. A warning is
right or wrong within six steps. A death happens or it does not. The level
clears or it does not. THE WORLD SUPPLIES THE LABEL, which is exactly what
every affect corpus in this project has lacked — `docs/cube_vs_human_vad.md`
ended by saying the arousal axis has to be validated against behaviour rather
than text, and this is the same point one level down: the words get their
affect from behaviour too.

So the player teaches Cubby their own vocabulary without either of them
trying. Say "lâche pas la patate" while he is in trouble and he gets out of
it, and the phrase moves warm — for THIS player, in host state, with no
gradient anywhere near the trunk. Someone else's encouragement words are
different words and are learned separately. That is the shaping mechanism with
a learning rule under it instead of a regular expression.

THE ADVANTAGE OVER A CONVENTIONAL MODEL. A conventional model's sense of which
words are warm was fixed when its weights were frozen; it cannot learn that
in your house "lâche pas la patate" is what encouragement sounds like, and the
only way to teach it is a fine-tune. Here the vocabulary is host state that
updates from outcomes during play, per player, and can be shown to them,
edited, or thrown away.
"""
from __future__ import annotations

import re

__wiring__ = "WIRED"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# The stemmer, ported. Extended past ASCII because half of what gets said to
# this agent is French and `[a-zA-Z0-9']+` silently cuts "lâche" into "che".
_WORD = re.compile(r"[^\W\d_]+['’]?[^\W\d_]*|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall((text or "").lower()):
        tok = raw.strip("'’")
        if not tok:
            continue
        if tok.endswith("ing") and len(tok) > 5:
            tok = tok[:-3]
        elif tok.endswith("ed") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("es") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 3:
            tok = tok[:-1]
        out.append(tok)
    return out


# Seeds, not a fixed vocabulary. Every one of these can be moved by evidence,
# and words not listed here are learned from nothing at all. They exist so the
# first conversation is not blank, which is the cold-start problem and not a
# reason to hard-code a personality.
#                     (valence, arousal, threat)
SEED = {
    # a claim about the world — the only feature that may raise threat
    "careful": (-0.2, 0.7, 0.8), "behind": (-0.2, 0.6, 0.6),
    # "watch" sits DELIBERATELY below `coach.WARN_AT`. The stemmer turns
    # "watching" into it, so "ok I'm watching" came out as a warning and put
    # threat into a body for no reason. The word is genuinely ambiguous on its
    # own — "watch out" and "I'm watching you play" are opposite messages — and
    # an ambiguous word must not fire alone. If this player uses it as a
    # warning and ghosts follow, `learn` raises it; that is the mechanism
    # working rather than a table being right in advance.
    "watch": (-0.1, 0.5, 0.45),
    "ghost": (-0.3, 0.6, 0.8), "danger": (-0.4, 0.8, 0.9), "run": (-0.2, 0.8, 0.7),
    "attention": (-0.2, 0.7, 0.8), "gaffe": (-0.2, 0.7, 0.8), "derriere": (-0.2, 0.6, 0.6),
    "fantome": (-0.3, 0.6, 0.8), "sauve": (-0.3, 0.8, 0.8), "degage": (-0.3, 0.8, 0.7),
    # encouragement, EN
    "can": (0.4, 0.5, 0.0), "go": (0.5, 0.7, 0.0), "nice": (0.7, 0.4, 0.0),
    "great": (0.8, 0.5, 0.0), "bravo": (0.8, 0.6, 0.0), "proud": (0.8, 0.4, 0.0),
    "almost": (0.4, 0.6, 0.0), "believe": (0.7, 0.4, 0.0), "keep": (0.5, 0.5, 0.0),
    # encouragement, QC — none of these is a translation of anything above
    "lache": (0.7, 0.6, 0.0), "patate": (0.5, 0.5, 0.0), "capable": (0.8, 0.4, 0.0),
    "continue": (0.5, 0.4, 0.0), "vas": (0.5, 0.6, 0.0), "courage": (0.7, 0.5, 0.0),
}


class AffectLexicon:
    """Per-player word affect, learned online from outcomes.

    `read` is what a message is worth now; `learn` is the world telling it
    what that message was actually worth. Nothing here touches model weights.
    """

    NEW_MOMENTUM, OLD_MOMENTUM, SETTLED_AT = 0.20, 0.08, 16
    STRONG = 0.45                 # a token this far out joins a set

    def __init__(self, seed: dict | None = None) -> None:
        self.ema: dict[str, list] = {}
        self.count: dict[str, int] = {}
        for tok, (v, a, t) in (SEED if seed is None else seed).items():
            self.ema[tok] = [v, a, t]
            self.count[tok] = 1               # a seed is one observation, not a law

    # ── reading ─────────────────────────────────────────────────────────────
    def read(self, text: str) -> dict:
        """-> {valence, arousal, threat, known}. Averaged over KNOWN tokens
        only: a sentence of unknown words is not neutral evidence, it is no
        evidence, and dividing by the full token count would quietly drag
        every reading toward zero as sentences get longer."""
        toks = tokenize(text)
        hits = [self.ema[t] for t in toks if t in self.ema]
        if not hits:
            return {"valence": 0.0, "arousal": 0.0, "threat": 0.0, "known": 0}
        n = float(len(hits))
        return {"valence": _clip(sum(h[0] for h in hits) / n, -1, 1),
                "arousal": _clip(sum(h[1] for h in hits) / n, 0, 1),
                "threat": _clip(max(h[2] for h in hits), 0, 1),   # one real warning is enough
                "known": len(hits)}

    # ── learning, from what actually happened ───────────────────────────────
    def learn(self, text: str, *, valence: float = 0.0, arousal: float = 0.0,
              threat: float = 0.0, weight: float = 1.0) -> int:
        """The world's verdict on a message -> its words move toward it.

        Momentum falls once a word has been seen `SETTLED_AT` times: quick to
        pick up a new word, slow to be argued out of a settled one. `weight`
        scales it for evidence that is real but weak — a level cleared two
        hundred steps after somebody said something is not that somebody's
        doing.
        """
        toks = set(tokenize(text))
        target = (_clip(valence, -1, 1), _clip(arousal, 0, 1), _clip(threat, 0, 1))
        for tok in toks:
            seen = self.count.get(tok, 0)
            m = (self.NEW_MOMENTUM if seen < self.SETTLED_AT else self.OLD_MOMENTUM) * _clip(weight, 0, 1)
            prev = self.ema.get(tok)
            self.ema[tok] = list(target) if prev is None else \
                [(1 - m) * p + m * t for p, t in zip(prev, target)]
            self.count[tok] = seen + 1
        return len(toks)

    # ── the sets, which are a view rather than a second store ───────────────
    #
    # GrillCheese kept `_positive_words` / `_negative_words` / `_threat_words`
    # as separate mutable sets alongside the EMA, which is one fact in two
    # places and the failure mode this project keeps hitting (the corner table
    # lived twice and both copies were wrong identically). They are derived
    # here, so they cannot disagree with the numbers they summarise.
    def words(self, kind: str) -> list[str]:
        if kind == "positive":
            pick = lambda v, a, t: v >= self.STRONG            # noqa: E731
        elif kind == "negative":
            pick = lambda v, a, t: v <= -self.STRONG           # noqa: E731
        elif kind == "threat":
            pick = lambda v, a, t: t >= 0.60 or (v <= -0.15 and a >= 0.55)   # noqa: E731
        else:
            raise ValueError(f"unknown set {kind!r}")
        return sorted(w for w, (v, a, t) in self.ema.items() if pick(v, a, t))

    # ── persistence: plain JSON, so a player's vocabulary is portable ───────
    def export_state(self) -> dict:
        return {"ema": {k: [round(x, 4) for x in v] for k, v in self.ema.items()},
                "count": dict(self.count)}

    def load_state(self, state: dict | None) -> None:
        if not isinstance(state, dict):
            return
        ema, count = state.get("ema"), state.get("count")
        if isinstance(ema, dict):
            self.ema = {str(k): [float(x) for x in v][:3] for k, v in ema.items()
                        if isinstance(v, (list, tuple)) and len(v) >= 3}
        if isinstance(count, dict):
            self.count = {str(k): int(v) for k, v in count.items()}

    def __len__(self) -> int:
        return len(self.ema)
