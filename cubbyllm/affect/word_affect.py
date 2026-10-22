"""lexicon — words whose affect is LEARNED from what happened next.

Wired: WIRED (stand-in game path, via `coach.Coach`). Nothing in cubbyllm/
imports this.

Ported from GrillCheese's `brain/amygdala.py` per the port-don't-link rule,
and deliberately only PART of it. Owner: *"it was working well with the
amygdala and snn in grillcheese maybe lets just reuse that?"* — so here is
what was taken, and what was left, and why.

READ THE RIGHT REPO. The notes below were first written against an OLDER
checkout, and the owner corrected me: the live one is `grillcheese-alpha-
0.0.1`. Its amygdala is a different and better thing, and the bits I said I
was rejecting are not in it:

  * its embeddings are REAL — `SentenceTransformer('all-MiniLM-L6-v2')`, not
    hash-seeded random vectors. Its own `train_amygdala.py` says why, in the
    author's words: *"The Vulkan embedder uses random weights and doesn't
    capture emotion. Sentence-transformers gives proper semantic similarity."*
    That is the exact objection I raised, already found and already fixed
    there. My criticism was of a superseded version.
  * there is no 2x6 linear head. The blend is a fixed scalar mix of a static
    60-word lexicon and a 3-layer MLP with a residual connection, trained with
    Adam.
  * there is no per-token EMA and no growing word sets. Those are MINE. The
    older checkout is where the idea came from; nothing was lifted.

WHAT SURVIVES THE CORRECTION, and it is the load-bearing half: the alpha's
amygdala needs a TRAINING RUN. `is_calibrated` gates the neural path, the
weights come from `train_amygdala.py` over labelled JSONL, and until that has
happened the learned path does not run at all. *"The whole concept of cubbyllm
is no-retraining"*, so the EMA below is still the right shape here for reasons
that have nothing to do with which repo I read.

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

LEFT BEHIND, and this is not a shortcut. Two of these describe the OLDER
checkout, which is where the port started; the alpha's version of each is
recorded beside it:

    AmygdalaAffectNet      OLD checkout: a 3-layer MLP over token vectors that
                           are `blake2b`-seeded RANDOM unit vectors, whose two
                           outputs then meet four lexical features in a 2x6
                           linear head. Random vectors carry no semantics, so
                           it could not generalise to a word it had not seen —
                           only memorise, which the EMA above already does more
                           cheaply and more legibly.
                           ALPHA: none of that holds any more — real MiniLM
                           embeddings, no 2x6 head, a residual MLP with Adam.
                           Still left behind, for the reason at the top: it is
                           inert until `train_amygdala.py` has been run over
                           labelled JSONL, and this project does not get a
                           training run.
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
                           than an EMA over one conversation.
                           That read has now happened, against the alpha, and
                           the loop is NOT closed there either:
                           `learning_state/stdp_state.json` is 811 KB, saved
                           and restored every run, and read by nothing.
                           `get_associated_tokens` has no caller in the repo;
                           `get_modulations` has one, in a test. `snn.process()`
                           runs AFTER generation at both entry points, and
                           `SNNCompute` holds no weight matrix at all. The
                           plasticity is real and write-only — it never reaches
                           an answer. What IS closed there, and is worth a port
                           on its own merits, is three loops that all route
                           through prompt text: basal-ganglia `strategy_biases`,
                           experience-buffer familiarity → arousal, and
                           endocrine/CNS state. So live STDP that actually
                           modulates output is unbuilt in both repos. Building
                           it IS the port; there is nothing to lift.
    the word lists         `sev1`, `outage`, `incident` — an ops assistant's
                           threat vocabulary. Wrong world. The mechanism ports;
                           the tables do not.

WHY THIS MATTERS MORE HERE THAN IT DID THERE, and it is the whole reason to
bother: GrillCheese has to be TOLD the target. In the alpha there IS an online
path, `_online_amygdala_learning`, but its target comes from a `quality` score
that both call sites pass as the literal 0.7, which falls through to
`target_valence = predicted * 0.9` — so it trains the head toward a shrunken
copy of its own last prediction, once per turn. No world, no label, no signal.
In the maze nobody has to tell it anything. A warning is
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
            # A NEW WORD STARTS AT NEUTRAL AND IS MOVED, rather than being set
            # to the target outright. The old version assigned `list(target)`
            # on first sight, which threw the weight away exactly when it
            # matters most: under the three-factor rule the weight IS the
            # evidence — how recently it was said times how hard the world
            # reacted — so a first-time word learned from a graze and one
            # learned from a catastrophe came out identical. Caught by
            # `test_credit_is_graded_by_how_much_the_body_moved`.
            #
            # It also fixes something that was wrong on its own terms: one
            # observation is not certainty, and a single confirmed warning
            # should not be enough to make a word a warning word.
            prev = self.ema.get(tok) or [0.0, 0.0, 0.0]
            self.ema[tok] = [(1 - m) * p + m * t for p, t in zip(prev, target)]
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


# ── eligibility: the middle term of a three-factor rule ─────────────────────
#
# Ported in SPIRIT from GrillCheese's `_apply_online_plasticity`, which keeps
# `_stdp_pre_trace` and `_stdp_post_trace` at `trace_decay=0.95` and updates
# them live on every chat turn. The owner was right that it runs live and I was
# wrong to call it a training path.
#
# But reading the whole of it: nothing ever READS those weights. Across that
# repo they are initialised, updated, grown, snapshotted, restored and asserted
# on in a round-trip test, and no forward pass consumes them. The SNN's one
# forward use, `spike_activity`, returns a scalar that lands in a telemetry
# struct next to `routing_entropy`. The loop is open, which is why the traces
# and the hormones are both there and never multiplied together.
#
# THAT PRODUCT IS THE WHOLE MECHANISM. Reward-modulated STDP is three factors:
#
#   PRE    the word was used            -> `mark`
#   POST   it is still eligible         -> the decaying trace
#   MOD    a neuromodulator moved       -> supplied by the world, at the moment
#                                          it moves, as a signed magnitude
#
# Two of the three were already here and the third is what this project has
# that GrillCheese did not: a world that delivers real reward and real pain at
# known instants. `docs/cube_vs_human_vad.md` ended by saying the arousal axis
# has to be validated against behaviour rather than text. This is the same
# point one level further down — the credit for a WORD comes from behaviour too.
#
# WHAT IT REPLACES, and why the old version was a stand-in rather than a
# mechanism: `Coach.outcome` taught every message inside a fixed 12-step window
# at a flat weight of 0.35. So a sentence one step before a death and one
# twelve steps before got identical credit, a sentence thirteen steps before
# got none, and the size of what happened did not matter at all. A cliff, and a
# constant. A trace is graded, has no edge, and is scaled by how much the
# chemistry actually moved.


class Eligibility:
    """What was said recently, and how much of it still counts.

    One trace per message rather than per token, because the message is the
    unit a person actually chose. `strength` starts at 1.0 and decays each
    step; below `FLOOR` it is dropped, which is what stops a lexicon filling up
    with superstition about something said a minute ago.
    """

    DECAY = 0.85                  # per step; ~0.05 after 18 steps
    FLOOR = 0.05
    MAX_HELD = 40                 # a bound, so a chatty player cannot grow it forever

    def __init__(self) -> None:
        self.held: list[dict] = []
        self._step = 0

    def mark(self, text: str, step: int, strength: float = 1.0) -> None:
        """Something was said. It is eligible from now until it decays out."""
        if not (text or "").strip():
            return
        self.decay_to(step)
        self.held.append({"text": text, "trace": _clip(strength, 0, 1), "at": step})
        if len(self.held) > self.MAX_HELD:
            self.held = sorted(self.held, key=lambda h: -h["trace"])[:self.MAX_HELD]

    def decay_to(self, step: int) -> None:
        """Advance the clock. Idempotent for a step already applied, so a
        caller may tick and mark in either order without double-decaying."""
        gap = step - self._step
        if gap <= 0:
            return
        self._step = step
        f = self.DECAY ** gap
        for h in self.held:
            h["trace"] *= f
        self.held = [h for h in self.held if h["trace"] >= self.FLOOR]

    def active(self) -> list[tuple]:
        """[(text, trace)], strongest first."""
        return [(h["text"], h["trace"]) for h in sorted(self.held, key=lambda h: -h["trace"])]

    def clear(self) -> None:
        self.held.clear()

    def __len__(self) -> int:
        return len(self.held)
