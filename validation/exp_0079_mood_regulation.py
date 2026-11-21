"""exp_0079_mood_regulation -- order 0079's checks of the mood on the hormone model alone (no language model).

Wired: STANDALONE (an experiment).

    python3 validation/exp_0079_mood_regulation.py --topical <Topical-Chat's train.jsonl> [--conversations 10000]

What it measures, written as measured (the order: "the numbers written as measured, not tuned to pass"):
1. **Recovery:** from each extreme (a corner of the cube at its band edges, oxytocin at either edge), with no input,
   the ODE frames, and the empty messages of two frames each, until all eight scales are within one of 0.
2. **Pace:** over Topical-Chat conversations, each person's turns sent in order through the host's own `nudge`
   (`ChatState`: appraisal, the transfer, the ODE), the largest change of any scale in one turn.
3. **Tolerance:** the same upsetting message five times in a row from rest: how much the fifth moves the scales
   against the first (the sum of absolute changes).
4. **Reach:** which names and GoEmotions shades the conversations reach, and the largest value of each scale.
The time between turns is not simulated: `nudge`'s `gap_s` acts only once a user grants the timing channel.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cubbyllm.affect import mood  # noqa: E402
from cubbyllm.affect.afferent import Afferent  # noqa: E402
from cubbyllm.affect.neurochem import Neurochemistry  # noqa: E402
from cubbyllm.speech.chat_state import ChatState  # noqa: E402

__wiring__ = "STANDALONE"

UPSET = {"anger": "I'm so angry, everything is ruined and I hate this.",
         "sad": "I feel awful, my dog died and I can't stop crying."}
AXES = (("5HT", "serotonin"), ("DA", "dopamine"), ("NE", "noradrenaline"))


class Body(ChatState):
    """The host's chat state with nothing else: the ODE, the transfer, the lexical appraiser (the default)."""

    def __init__(self) -> None:
        self.chem, self.afferent, self.appraiser = Neurochemistry(), Afferent(), None
        self._seen_vocab, self.signals = set(), {}
        self.state = self._hormones()


def quiet(chem: Neurochemistry, cap: int = 4000) -> tuple[int, int]:
    """(frames, empty two-frame messages) until every scale is within one of 0; (cap, cap // 2) if never."""
    for f in range(cap):
        if all(abs(v) <= 1 for v in mood.scales(chem).values()):
            return f, (f + 1) // 2
        chem.update()
    return cap, cap // 2


def recovery() -> dict:
    out = {}
    q = Neurochemistry.quiescent()
    for corner, coords in Neurochemistry._CORNERS.items():
        b = Body()
        for (key, attr), bit in zip(AXES, coords):
            lo, hi = Neurochemistry._BAND[key]
            setattr(b.chem, attr, hi if bit else lo)
        start = mood.scales(b.chem)
        frames, msgs = quiet(b.chem)
        out[corner] = {"start": start, "frames_to_within_1": frames, "empty_messages": msgs}
    for name, ot in (("oxytocin high", Neurochemistry._BAND["OT"][1]), ("oxytocin low", Neurochemistry._BAND["OT"][0])):
        b = Body()
        b.chem.oxytocin = ot
        start = mood.scales(b.chem)
        frames, msgs = quiet(b.chem)
        out[name] = {"start": start, "frames_to_within_1": frames, "empty_messages": msgs}
    out["_quiescent"] = {k: round(v, 4) for k, v in q.items()}
    return out


def conversations(path: pathlib.Path, limit: int) -> dict:
    jumps, per_petal = [], collections.Counter()
    top = dict.fromkeys(mood.ORDER, 0)
    low = dict.fromkeys(mood.ORDER, 0)
    names, shades, turns, n = collections.Counter(), collections.Counter(), 0, 0
    for line in open(path, encoding="utf-8"):
        if n >= limit:
            break
        _, conv = json.loads(line)
        b, n = Body(), n + 1
        before = mood.scales(b.chem)
        for t in conv.get("content") or []:
            if t.get("agent") != "agent_1":
                continue
            b.nudge(t.get("message") or "")
            after = mood.scales(b.chem)
            d = {p: after[p] - before[p] for p in mood.ORDER}
            m = max(abs(x) for x in d.values())
            jumps.append(m)
            for p in mood.ORDER:
                per_petal[p] = max(per_petal[p], abs(d[p]))
                top[p], low[p] = max(top[p], after[p]), min(low[p], after[p])
            for x in mood.named(after):
                names[x] += 1
            for x in mood.shade(after):
                shades[x] += 1
            turns += 1
            before = after
    jumps.sort()
    pct = lambda q: jumps[min(len(jumps) - 1, int(q * len(jumps)))] if jumps else None  # noqa: E731
    return {"conversations": n, "user_turns": turns,
            "largest_one_turn_change": jumps[-1] if jumps else None,
            "one_turn_change_p50_p90_p99": [pct(0.5), pct(0.9), pct(0.99)],
            "largest_one_turn_change_by_petal": dict(per_petal),
            "highest_by_petal": top, "lowest_by_petal": low,
            "names_reached": dict(names.most_common()), "shades_reached": dict(shades.most_common()),
            "shades_never_reached": sorted(set(mood.LABEL_PETAL) - set(shades) - {"neutral"})}


def tolerance() -> dict:
    out = {}
    for kind, text in UPSET.items():
        b = Body()
        moves, before = [], mood.scales(b.chem)
        for _ in range(5):
            b.nudge(text)
            after = mood.scales(b.chem)
            moves.append(sum(abs(after[p] - before[p]) for p in mood.ORDER))
            before = after
        out[kind] = {"message": text, "moves": moves, "after_five": before,
                     "fifth_over_first": round(moves[4] / moves[0], 3) if moves[0] else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topical", required=True, help="Topical-Chat's train.jsonl (off-repo)")
    ap.add_argument("--conversations", type=int, default=10000)
    ap.add_argument("--regulated", action="store_true",
                    help="CB_MOOD_REGULATION=1: novelty habituates, a friendly turn raises serotonin (Nick's yes)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.regulated:
        import os
        os.environ["CB_MOOD_REGULATION"] = "1"
    a.out = a.out or str(ROOT / "validation" / "logs" / ("exp_0079_mood_regulation" + ("_regulated" if a.regulated else "")
                                                         + ".json"))
    res = {"what": "order 0079: the mood's regulation on the hormone model alone", "regulated": a.regulated,
           "validate": mood.validate(),
           "recovery": recovery(), "tolerance": tolerance(),
           "pace_and_reach": conversations(pathlib.Path(a.topical), a.conversations)}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
