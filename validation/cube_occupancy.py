"""Which corners does the ODE actually visit?

The test the panel asked for, and it now has a reason to exist. Remapping `_CORNERS`
to Lövheim's published table made `test_being_caught_raises_fear` fail: a ghost catch
produces DISTRESS/ANGUISH (low 5-HT, low DA, high NE) where the test expected FEAR.

That is not obviously a regression. Lövheim's fear/terror sits at low 5-HT, HIGH DA,
LOW NE - anticipatory dread, oriented toward a threat that has not landed - while
distress/anguish is the aftermath: alarm up, reward prospect gone. Being eaten is the
aftermath. Arguably the new reading is the more honest one.

But it exposes a real tension. This ODE has a `- threat * 0.45` term on dopamine, added
so a hunted agent stops feeling rewarded. Lövheim's fear corner needs dopamine HIGH. So
under threat the ODE moves AWAY from the corner named fear and toward the one named
distress, and fear may be unreachable by construction - the same class of bug as the
pre-scaling cube, where noradrenaline's band made half the corners impossible.

Two ways out, and they are not equivalent:
  (a) the threat->DA coupling is wrong, and fear should keep some drive in it;
  (b) the coupling is right and this ODE's threat response simply is distress.

Guessing between them is how a wrong constant survives. So: run the world, log every
frame's corner, and report the histogram. A corner at 0% occupancy after a full run is
unreachable, and unreachable is a fact rather than an opinion.
"""
from __future__ import annotations

import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402

OUT = ROOT / "docs" / "cube_occupancy.md"


def drive(frames: int, **signals) -> Counter:
    """Hold one situation steady and see where the body ends up."""
    chem = Neurochemistry()
    seen = Counter()
    for _ in range(frames):
        chem.update(**signals)
        seen[chem.dominant_emotion] += 1
    return seen


# Situations this agent actually meets, described in the ODE's own signal vocabulary
# rather than in emotion words - the point is to find out what they PRODUCE.
SITUATIONS = {
    "idle, nothing happening":      dict(novelty=0.0, reward=0.0, threat=0.0, social=0.0),
    "exploring, new ground":        dict(novelty=0.9, reward=0.1, threat=0.0, social=0.0),
    "pellet eaten":                 dict(novelty=0.2, reward=0.9, threat=0.0, social=0.0),
    "ghost closing in":             dict(novelty=0.3, reward=0.0, threat=0.9, social=0.0),
    "ghost near, still exploring":  dict(novelty=0.7, reward=0.1, threat=0.6, social=0.0),
    "caught - hurt, no prospect":   dict(novelty=0.1, reward=0.0, threat=0.8, social=0.0),
    "stuck, nothing works":         dict(novelty=0.0, reward=0.0, threat=0.1, social=0.0),
    "level cleared":                dict(novelty=0.4, reward=1.0, threat=0.0, social=0.0),
}


def main() -> int:
    corners = list(Neurochemistry._CORNERS)
    rows, union = [], Counter()
    for label, sig in SITUATIONS.items():
        seen = drive(120, **sig)
        union.update(seen)
        rows.append((label, seen))

    # The harder question: with pain on top, does anything change?
    hurt = Neurochemistry()
    hurt_seen = Counter()
    for i in range(120):
        hurt.update(novelty=0.1, reward=0.0, threat=0.8, social=0.0)
        if hasattr(hurt, "hurt"):
            hurt.hurt(0.6)
        hurt_seen[hurt.dominant_emotion] += 1
    union.update(hurt_seen)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Cube occupancy after the corner remap\n\n")
        f.write("120 frames per situation, each held steady from a fresh body. "
                "Signals are the ODE's own inputs, not emotion words - the question is "
                "what each situation PRODUCES.\n\n")
        f.write("| situation | corners visited |\n|---|---|\n")
        for label, seen in rows:
            f.write(f"| {label} | " +
                    ", ".join(f"**{k}** {v}" for k, v in seen.most_common()) + " |\n")
        f.write(f"| caught + pain injected | " +
                ", ".join(f"**{k}** {v}" for k, v in hurt_seen.most_common()) + " |\n")

        f.write("\n## Reachability\n\n")
        f.write("| corner | coord | frames | reachable |\n|---|---|---:|---|\n")
        for c in corners:
            n = union.get(c, 0)
            f.write(f"| {c} | {Neurochemistry._CORNERS[c]} | {n} | "
                    f"{'yes' if n else '**NO**'} |\n")
        f.write(f"| neutral | centre | {union.get('neutral', 0)} | - |\n")

        dead = [c for c in corners if not union.get(c)]
        f.write("\n## Reading this\n\n")
        if dead:
            f.write(f"**Unreachable: {', '.join(dead)}.** A corner no situation can "
                    f"produce is a corner the agent can never be in - the same class of "
                    f"bug as the pre-scaling cube, where noradrenaline's band made half "
                    f"the vertices impossible by construction.\n\n")
            if "fear" in dead:
                f.write("`fear` specifically is the one to argue about. Lövheim puts it "
                        "at low 5-HT / HIGH DA / low NE, and this ODE subtracts "
                        "`threat * 0.45` from dopamine - so threat pushes the body away "
                        "from the fear corner and toward distress. Either the coupling "
                        "is too strong, or this ODE's threat response genuinely is "
                        "distress rather than fear. That is a modelling decision, not a "
                        "bug to patch quietly.\n")
        else:
            f.write("Every corner is reachable from at least one ordinary situation. "
                    "That is the first time this has been true.\n")

    print(f"wrote {OUT}")
    for c in corners:
        print(f"  {c:<9} {Neurochemistry._CORNERS[c]}  {union.get(c, 0):>4} frames")
    print(f"  neutral             {union.get('neutral', 0):>4} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
