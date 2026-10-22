"""cubbyllm/affect/neurochem_body.py -- the chemistry as a felt body: the somatic names and their influence, the
emotion cube and its corners, what a state could be, the modulation it applies and the manner it speaks in. Split out
of neurochem.py (order 0031) as a mixin of Neurochemistry, verbatim, with the constants it reads.

Wired: WIRED (Neurochemistry).
"""
from __future__ import annotations

from .neurochem_appraise import _clip

__wiring__ = "WIRED"


class NeuroBody:
    """Neurochemistry's felt body: somatic, cube, corners, manner (a mixin; order 0031)."""

    # A region of hormone space -> names that FIT it. Opposed readings live in
    # the same list on purpose: the chemistry says how the body is, never what
    # to call it. Each list is a real fork — despair and stubbornness are the
    # same body read two ways, and so are elation and recklessness.
    #
    # THE ORDER IS LOAD-BEARING: `could_be` takes the first hot region, so this
    # is a priority list, not a lookup table. Pain first, because it is an
    # explicit signal rather than an inference and a body in pain is not also
    # having a nice time. Then the TWO-HORMONE conjunctions, then the
    # single-signal readings — a conjunction is more diagnostic than one
    # hormone moving, so it should win when both fit. Ordering by pleasantness
    # instead put "rawness or irritation" ahead of elation on a serotonin dip
    # that a noradrenaline surge had caused, and eating a power star came out
    # as irritation.
    _NAMES: list[tuple[str, list[str]]] = [
        ("hurt",        ["shock", "fear", "anger"]),
        ("low_da_hi_c", ["despair", "stubbornness", "grim determination", "being fed up"]),
        ("hi_ne_lo_da", ["dread", "being rattled", "nerve"]),
        # High dopamine with high noradrenaline is TWO different states and
        # serotonin is what separates them — Lövheim puts anger at (low 5HT,
        # high DA, high NE) and joy at (high 5HT, high DA, high NE). Collapsing
        # them into one region let the compass say *furious* while the names on
        # offer said *elation*, and the model duly reported having a great time
        # in a rage. Two classifiers over one hormone space have to agree about
        # the space.
        ("hi_da_hi_ne_lo_5ht", ["anger", "fury", "being pushed too far"]),
        ("hi_da_hi_ne", ["elation", "recklessness", "being on a roll"]),
        ("hi_da_lo_ne", ["contentment", "ease", "quiet satisfaction"]),
        ("craving",     ["wanting", "restlessness", "an itch"]),
        ("lo_5ht",      ["rawness", "irritation"]),
        ("hi_ot",       ["warmth", "trust"]),
    ]

    # The Lövheim corner this state is nearest -> the honest readings OF THAT
    # CORNER. Keyed off `dominant_emotion` rather than off a second set of
    # thresholds, because two classifiers over one space disagree in the band
    # between them however carefully each is tuned: the corner classifier is
    # nearest-corner and the thresholds were cutoffs, so the compass said
    # `annoyance` while the names said `elation` on the same step. One
    # classifier decides WHERE he is; this table only decides what that place
    # can honestly be called, and the fork inside each list is the part worth
    # handing over.
    #
    # NOTHING IS EMPTY ANY MORE. The two entries that used to be blank were
    # `contempt` and `shame`, excluded because they need somebody else to feel
    # them about — correct reasoning applied to the wrong corners. Those
    # coordinates are Lövheim's FEAR and DISGUST, so a solitary agent in a maze
    # full of ghosts has every reason to reach both, and the fear corner had
    # simply been silent. Trust is the genuinely social one, and it has no
    # corner at all: it is oxytocin, which is not an axis of this cube.
    #
    # `spent` is the corner the panel renamed from shame: no drive, no reward
    # signal, no arousal. It is what four failed attempts at one level look like.
    _BY_CORNER: dict[str, list[str]] = {
        "spent":    ["being spent", "running on empty", "past caring"],
        "distress": ["distress", "being wrung out", "at a loss"],
        "fear":     ["dread", "being rattled", "nerve"],
        "anger":    ["anger", "fury", "being pushed too far"],
        "disgust":  ["being fed up", "sick of it", "revulsion"],
        "surprise": ["surprise", "being caught out", "confusion"],
        "joy":      ["quiet satisfaction", "ease", "warmth"],
        "interest": ["interest", "being on a roll", "elation"],
        "neutral":  [],
    }

    # ── Layer 1: the somatic drives the cube has no axis for ───────────────
    #
    # Pain and craving are not cognitive states sitting somewhere in monoamine
    # space; they are homeostatic imperatives on their own circuits, and the
    # Lövheim cube reads 5-HT, DA and NE only. Forced onto those three axes
    # they get swallowed — a hurting body lands wherever high NE and low 5-HT
    # put it, which is the anger corner. The corner is doing its job on the
    # evidence it has, and the evidence does not contain the thing that matters.
    #
    # So they do two things, not one. They SKEW the monoamines (below, through
    # the drives, so the skew integrates and decays like everything else rather
    # than being pasted onto the vector at read time — that is what makes pain
    # linger instead of ending with the blow). And above a threshold they
    # INTERCEPT the naming, because past a point "he is in pain" is the true
    # thing to say about him whatever corner the three amines imply.
    #
    # The influence vectors are named rather than scattered through the drive
    # expressions so they can be read, argued with and tested as a unit.
    #                       5-HT    DA     NE
    PAIN_INFLUENCE =      (-0.25, -0.30, +0.60)   # vulnerability, anti-reward, hyper-arousal

    CRAVING_INFLUENCE =   (-0.10,  0.00, +0.30)   # frustration, searching alertness

    #
    # CRAVING'S DOPAMINE TERM IS DELIBERATELY ZERO, against the obvious reading
    # of incentive salience. Wanting IS mesolimbic dopamine — but the phasic
    # burst to a cue, not the tonic level, and `craving` here is DERIVED from
    # the tonic shortfall (`tolerance` x how far dopamine sits below quiescent).
    # A positive term would therefore close the very gap that produced it: DA
    # rises, the shortfall shrinks, craving collapses, DA falls, craving
    # returns — an oscillator that erases the withdrawal this models. The
    # phasic half is real and belongs on a CUE (a frightened ghost actually in
    # sight), not on the baseline; see `pacman.CubbyGhost.chasing`, which is
    # where a cue exists. Wiring it here would be a feedback loop wearing the
    # costume of neuroscience.
    PAIN_INTERCEPT, AGONY = 0.35, 0.70

    CRAVING_INTERCEPT, FRANTIC = 0.30, 0.65

    SOMATIC_NAMES = {
        "pain":    ["it hurts", "shock", "fear"],
        "agony":   ["agony", "panic", "rage"],
        "craving": ["wanting", "restlessness", "an itch"],
        "frantic": ["need", "a craving I can't sit with", "desperation"],
    }

    def somatic(self) -> list[str]:
        """Layer 1. The readings that outrank the cube, or [] when neither fires.

        Graded, because a scraped knee and a broken leg are not the same word:
        past the second threshold the name changes rather than just getting
        louder. Pain wins ties — a body that is both hurting and wanting has a
        more urgent problem than the wanting."""
        if self.pain >= self.PAIN_INTERCEPT and self.pain >= self.craving:
            return list(self.SOMATIC_NAMES["agony" if self.pain > self.AGONY else "pain"])
        if self.craving >= self.CRAVING_INTERCEPT:
            return list(self.SOMATIC_NAMES["frantic" if self.craving > self.FRANTIC else "craving"])
        return []

    def cube(self) -> tuple:
        """(5-HT, DA, NE) scaled so QUIESCENT IS THE CENTRE of the cube.

        Raw concentrations never approach 0 or 1 — noradrenaline lives near
        0.15 in a band topping out at 0.90 — so the raw point sits permanently
        in one half of the cube and the corners on the other side are
        unreachable by construction. That is why he could not be anxious."""
        base, out = self.quiescent(), []
        for key, now in (("5HT", self.serotonin), ("DA", self.dopamine), ("NE", self.noradrenaline)):
            rest, (lo, hi) = base[key], self._BAND[key]
            span = (hi - rest) if now >= rest else (rest - lo)
            out.append(_clip(0.5 + 0.5 * (now - rest) / max(span, 1e-6), 0, 1))
        return tuple(out)

    # Lövheim's PUBLISHED corners, (5-HT, DA, NE). Verified against the source
    # table 2026-09-17, after the previous table turned out to have only TWO of
    # its eight corners in the right place:
    #
    #   coord    was          Lövheim actually says
    #   0,0,0    sad          shame/humiliation
    #   0,0,1    anxious      distress/anguish
    #   0,1,0    contempt     FEAR/terror
    #   0,1,1    angry        anger/rage              (correct)
    #   1,0,0    shame        CONTEMPT/DISGUST
    #   1,0,1    surprise     surprise                (correct)
    #   1,1,0    warm         enjoyment/joy
    #   1,1,1    joy          INTEREST/EXCITEMENT
    #
    # Three consequences we had been debugging as separate bugs. Anticipation
    # had "no corner" because JOY was sitting on the interest vertex - which is
    # why curiosity needed the novelty override that this commit deletes. And
    # the two corners excluded as social, `contempt` and `shame`, were in fact
    # Lövheim's FEAR and DISGUST vertices, so the fear corner has been muted
    # this whole time. That is most of "he never feels anxiety when he fails".
    #
    # NAMES ARE MONOAMINE SYNDROMES, NOT PLUTCHIK PETALS. Forcing petal names
    # onto vertices is what produced the mismatch; the petal mapping lives one
    # layer up in `pacman._PETAL` where it can be many-to-one without lying.
    #
    # `spent` rather than `shame` at all-low: five of six panel models argued
    # independently that no-serotonin/no-dopamine/no-noradrenaline is depletion,
    # not shame - shame is a high-arousal state with a social appraisal attached,
    # and all-low is what the ODE looks like after sustained unrewarded effort.
    # That trades an unreachable social corner for a reachable, useful one.
    _CORNERS = {"spent":    (0, 0, 0),   # Lövheim shame/humiliation -> depletion
                "distress": (0, 0, 1),   # distress/anguish
                "fear":     (0, 1, 0),   # fear/terror
                "anger":    (0, 1, 1),   # anger/rage
                "disgust":  (1, 0, 0),   # contempt/disgust
                "surprise": (1, 0, 1),   # surprise
                "joy":      (1, 1, 0),   # enjoyment/joy - consummatory, low NE
                "interest": (1, 1, 1)}   # interest/excitement - the explorer's vertex

    # ── Layer 2: where in the cube, how far out, and what he is near ────────
    def corner_position(self) -> dict:
        """The geometry, with no names attached — naming is Layer 3's job.

        `intensity` is how far out toward the corner he actually is, which is
        what separates annoyance from rage inside one octant: the corner says
        WHICH family, the distance says how much of it. `runner_up` and
        `margin` are the boundary case — a point sitting between two corners
        should not be described as squarely in either, and Plutchik already has
        a name for the pair.

        Nearest-corner, so the eight cells meet at the centre and every corner
        is reachable. A threshold classifier over the same space carves unequal
        boxes and disagrees with this one in the band between them, which is
        exactly the bug that put `annoyance` on the compass and `elation` in
        the names on the same step."""
        pos = self.cube()
        def dist(c):
            return sum((p - q) ** 2 for p, q in zip(pos, c)) ** 0.5
        ranked = sorted(self._CORNERS.items(), key=lambda kv: dist(kv[1]))
        best, second = ranked[0][0], ranked[1][0]
        reach = sum((p - 0.5) ** 2 for p in pos) ** 0.5 / (0.75 ** 0.5)
        return {"corner": best, "intensity": _clip(reach, 0, 1), "runner_up": second,
                "margin": dist(self._CORNERS[second]) - dist(self._CORNERS[best]),
                "pos": pos}

    def could_be(self, limit: int = 3) -> list[str]:
        """Layer 1, then a default Layer 3 for callers with no naming table.

        The game overrides the naming with the Plutchik one (`pacman.felt_names`),
        which reads the intensity tier and the boundary dyad. This keeps a
        sensible answer for ToyVerse and the tests, and it is the same corner
        either way — only the vocabulary differs."""
        somatic = self.somatic()
        if somatic:
            return somatic[:limit]
        return list(self._BY_CORNER.get(self.dominant_emotion, [])[:limit])

    # ── the dials: hormones -> HOW he speaks, with no feeling word anywhere ──
    #
    # Ported from GrillCheese's `EndocrineSystem.get_modulation_factors`
    # (backend/brain/endocrine.py), which is the piece this port was missing.
    # That system never hands the generator a label either; it hands it
    # continuous knobs — creativity, caution, warmth, energy, stability, focus
    # — and lets the voice fall out of them. Two additions here, for the two
    # signals GrillCheese's version has no input for: pain and craving.
    #
    # This is the answer to *"it still sounds a bit too artificial"*. A model
    # told "you feel anxious" says the word anxious. A model whose sentence is
    # generated short, clipped and wary SOUNDS anxious without the word
    # appearing anywhere — which is what feeling something looks like from
    # outside, and it cannot be faked by copying a label out of a record.
    def modulation(self) -> dict:
        """Continuous 0..1 knobs on the voice. No emotion words, by design."""
        da, ne = self.dopamine, self.noradrenaline
        sht, ot, c = self.serotonin, self.oxytocin, self.cortisol
        hurt, want = self.pain, self.craving
        return {
            "creativity": _clip(da - c * 0.5 + 0.3 - hurt * 0.3, 0, 1),
            # CAUTION IS FAST AND SLOW, and it used to be only slow. It read
            # `c * 1.5 + hurt * 0.4` — cortisol, which is a deliberately
            # sluggish EMA of arousal, so a startle did not move it at all. A
            # body that only gets careful after minutes of sustained stress is
            # missing the part that matters: you widen your guard the instant
            # something moves, and that is noradrenaline, not cortisol.
            #
            # Found by moving `pac_nav.wariness` off a raw hormone read and
            # onto this knob — the knob is where hormones become world-agnostic
            # gains, and a second place reading a raw hormone is a second
            # opinion about what alarm means. The berth stopped responding to a
            # warning, and the right fix was not to put the hormone back in the
            # world: the global knob was wrong for everybody, the voice
            # included.
            "caution": _clip(c * 1.2 + ne * 0.5 + hurt * 0.4, 0, 1),
            "warmth": _clip(ot * 1.2 + sht * 0.3 - hurt * 0.5, 0, 1),
            "energy": _clip(ne * 1.3 + da * 0.2, 0, 1),
            "stability": _clip(sht * 1.2 - c * 0.3 - hurt * 0.4, 0, 1),
            "focus": _clip(ne * 0.8 + c * 0.2 + want * 0.3, 0, 1),
            "urgency": _clip(ne * 0.6 + hurt * 0.6 + want * 0.4, 0, 1),
        }

    # each dial, at its extremes, in terms of MANNER — how a person talks, not
    # how they feel. The speaker is never told what is happening to it.
    _MANNER = {
        "caution":   (0.62, "measured, and not committing to much", 0.18, None),
        "energy":    (0.55, "quick, a bit clipped", 0.20, "slow, not much left in it"),
        "warmth":    (0.55, "open", 0.15, "flat, keeping its distance"),
        "stability": (0.18, None, 0.62, None),
        "urgency":   (0.60, "short — no time to round it off", 0.15, None),
    }

    def manner(self, limit: int = 2) -> str:
        """The dials as a short instruction about DELIVERY, for the prompt.

        Only the extremes speak: a dial sitting mid-range says nothing, because
        a voice note for every knob on every turn is a style sheet, and a style
        sheet read aloud is exactly the artificial thing this replaces."""
        m, out = self.modulation(), []
        for key, (hi, hi_word, lo, lo_word) in self._MANNER.items():
            v = m[key]
            if hi_word and v >= hi:
                out.append((v - hi, hi_word))
            elif lo_word and v <= lo:
                out.append((lo - v, lo_word))
        out.sort(key=lambda t: -t[0])
        return ", ".join(w for _, w in out[:limit])
