"""Where does each axis actually go? The occupancy result needs a cause.

After the corner remap, four of eight corners were never visited in 960 frames across
eight situations: `spent`, `fear`, `joy`, `interest`. The two that fire - distress
(0,0,1) and anger (0,1,1) - share low serotonin and high noradrenaline.

That is the signature of two pinned axes, not of eight situations happening to miss
four corners. If 5-HT never rises above its quiescent point, the whole high-5-HT face
(disgust, surprise, joy, interest) is out of reach; if NE never falls below it, the
low-NE corners (spent, fear, joy) go too. Together those two account for exactly the
four that are missing.

Same class of bug as the one already fixed once here: before the quiescent-centred
scaling, noradrenaline lived near 0.15 in a band topping out at 0.90, so the raw point
sat permanently in one half of the cube and no amount of threat could reach the corners
on the other side. The scaling fixed the READOUT. This asks whether the DYNAMICS have
the same problem one layer down - whether the drives can move each hormone to both
sides of its own resting value at all.

Prints the cube coordinate per situation, so the answer is a number rather than an
inference.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from neurochem import Neurochemistry  # noqa: E402

# THE GAME'S OWN CALL SITES, copied from `pacman.py` rather than invented.
#
# The first version of this probe passed `reward=` and no `valence=`, concluded that
# serotonin was pinned at its floor in every situation, and was wrong: `joy` is
# `max(0, valence)`, and `reward` feeds dopamine only. Inventing the signals meant
# measuring a body nobody ever builds. These are the arguments the live game actually
# sends, line numbers included so they can be checked against the source.
SITUATIONS = {
    "idle (nothing sent)":        dict(),
    "discovery (l.1473)":         dict(novelty=0.7, valence=0.8),
    "caught (l.1721)":            dict(threat=1.0, valence=-0.8, pain=0.6),
    "star / power pellet (2511)": dict(valence=0.8, surge=0.9, focus=0.6, novelty=0.3),
    "small good thing (2513)":    dict(valence=0.3),
    "escaped (2759)":             dict(valence=0.6, novelty=0.4, social=0.2),
    "failed a level (2801)":      dict(valence=-0.7, threat=0.14, focus=0.3),
    "ghost near (2918)":          dict(threat=0.6, novelty=0.5),
    "exploring, calm (2923)":     dict(novelty=0.5),
    "pellet (2938)":              dict(valence=0.9, novelty=0.3),
    "level cleared (2946)":       dict(valence=1.0, reward=1.0, novelty=0.3),
    "ambient threat (2980)":      dict(threat=0.3),
}


def main() -> int:
    base = Neurochemistry.quiescent()
    print("quiescent:", {k: round(v, 3) for k, v in base.items()
                         if k in ("5HT", "DA", "NE")})
    print()
    print(f"{'situation':<28} {'5-HT':>18} {'DA':>18} {'NE':>18}   cube          corner")
    print(f"{'':<28} {'raw (cube)':>18} {'raw (cube)':>18} {'raw (cube)':>18}")
    lo = {"5HT": 1.0, "DA": 1.0, "NE": 1.0}
    hi = {"5HT": 0.0, "DA": 0.0, "NE": 0.0}
    for label, sig in SITUATIONS.items():
        chem = Neurochemistry()
        for _ in range(150):
            chem.update(**sig)
        c = chem.cube()
        raw = {"5HT": chem.serotonin, "DA": chem.dopamine, "NE": chem.noradrenaline}
        for i, k in enumerate(("5HT", "DA", "NE")):
            lo[k] = min(lo[k], c[i])
            hi[k] = max(hi[k], c[i])
        cells = "  ".join(f"{raw[k]:>6.3f} ({c[i]:.2f})"
                          for i, k in enumerate(("5HT", "DA", "NE")))
        print(f"{label:<28} {cells}   {tuple(round(x, 2) for x in c)}  {chem.dominant_emotion}")

    print()
    print("cube-coordinate range reached across every situation:")
    for k in ("5HT", "DA", "NE"):
        span = hi[k] - lo[k]
        verdict = "PINNED" if span < 0.25 else "moves"
        side = ("never below centre" if lo[k] >= 0.5 else
                "never above centre" if hi[k] <= 0.5 else "both sides")
        print(f"  {k:<4} {lo[k]:.2f} .. {hi[k]:.2f}   span {span:.2f}  {verdict}, {side}")
    print()
    print("An axis that never crosses 0.5 halves the cube. Two of them leave two corners.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

