"""exp_0079_mood_read -- why the pilot's mood was flat: the read, not the hormones (order 0079, Cowork, 2026-11-21).

Two checks, counts only, written to `validation/logs/exp_0079_mood_read.json`:

1. **The read against labels** (`--affect`, Nick's amygdala_affect.jsonl: text, valence, arousal; off the repo). The
   rows at exactly (0.0, 0.5) are the file's default and are left out. For each labelled text: the lexical
   `appraise` valence (what the pilot used: `ChatState` with no appraiser), and the correlation with the label.
2. **The pilot's moods replayed** (`--pilot`, the pilot's conversations, off the repo) two ways over the same user
   turns: the lexical read alone (as drafted), and the read the host has at serving (`perception.ModelAppraiser`:
   the trunk names the user's emotion, its Plutchik petal goes through `PETAL_AFFECT`, blended with the lexicon by
   magnitude), with the drafting agent's GoEmotions label standing in for the trunk's read (`mood.LABEL_PETAL`).
   Either way the read goes through `afferent` before the ODE, so Cubby is not made to mirror the user. Each is
   read with the two fixes for the oddities (`mood.SIDE_MARGIN`, `chat_state.CONCERN_SAFE`) and without them.

    python validation/exp_0079_mood_read.py --affect <amygdala_affect.jsonl> --pilot <chat_0079_pilot_conversations.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CB_MOOD_REGULATION"] = "1"

from cubbyllm.affect import mood  # noqa: E402
from cubbyllm.affect.neurochem import appraise  # noqa: E402
from cubbyllm.affect.perception import PETAL_AFFECT  # noqa: E402
from cubbyllm.speech import chat_state  # noqa: E402
from validation.sft.build_chat_0079 import Body, LabelAppraiser  # noqa: E402


def corr(x: list[float], y: list[float]) -> float | None:
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 3) if sx and sy else None


def read_check(path: str) -> dict:
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    lab = [r for r in rows if not (r["valence"] == 0 and r["arousal"] == 0.5)]
    reads = [appraise(r["text"], set())["valence"] for r in lab]
    v = [r["valence"] for r in lab]
    pos = [i for i, r in enumerate(lab) if r["valence"] >= 0.6]
    neg = [i for i, r in enumerate(lab) if r["valence"] <= -0.6]
    return {"rows": len(rows), "rows_at_the_default_0_0.5": len(rows) - len(lab), "labelled": len(lab),
            "lexical_read_is_0": sum(x == 0 for x in reads), "corr_label_vs_lexical_read": corr(v, reads),
            "label_at_least_+0.6": len(pos), "of_them_read_above_+0.2": sum(reads[i] > 0.2 for i in pos),
            "label_at_most_-0.6": len(neg), "of_them_read_below_-0.2": sum(reads[i] < -0.2 for i in neg)}


def summary(scales: list[dict]) -> dict:
    return {"turns": len(scales),
            "at_4_or_more": {e: sum(s[e] >= 4 for s in scales) for e in mood.ORDER},
            "at_1_or_more": {e: sum(s[e] >= 1 for s in scales) for e in mood.ORDER},
            "turns_with_a_shade": sum(bool(mood.shade(s)) for s in scales),
            "named": dict(collections.Counter(n for s in scales for n in mood.named(s)).most_common())}


def replay(path: str) -> dict:
    convs = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    lexical, labelled, seems = [], [], []
    for c in convs:
        a, b = Body(), Body()
        b.appraiser = LabelAppraiser()
        for t in c["turns"]:
            a.nudge(t["user"])
            b.appraiser.label = t["seems"]
            b.nudge(t["user"])
            lexical.append(mood.scales(a.chem))
            labelled.append(mood.scales(b.chem))
            seems.append(t["seems"])
    groups = {"sad or grieving": ("sadness", "grief"), "joyful, excited or proud": ("joy", "excitement", "pride"),
              "nervous or afraid": ("nervousness", "fear"),
              "curious, nervous or afraid": ("curiosity", "nervousness", "fear")}
    by = {}
    for name, labels in groups.items():
        idx = [i for i, s in enumerate(seems) if s in labels]
        by[name] = {"turns": len(idx), **{f"mean_{e}": round(statistics.mean(labelled[i][e] for i in idx), 2)
                                          for e in ("joy", "trust", "sadness", "fear", "anger")},
                    "anger_at_1_or_more": sum(labelled[i]["anger"] >= 1 for i in idx),
                    "shades": dict(collections.Counter(x for i in idx for x in mood.shade(labelled[i])).most_common())}
    return {"conversations": len(convs), "lexical_read": summary(lexical), "label_read": summary(labelled),
            "label_read_by_how_the_user_seems": by}


def replay_both(path: str) -> dict:
    """The replay with the two fixes for the oddities (`mood.SIDE_MARGIN`, `chat_state.CONCERN_SAFE`) and without them
    (an infinite margin, i.e. closeness alone, and no concern in safety: as the pilot drafted)."""
    keep = (mood.SIDE_MARGIN, chat_state.CONCERN_SAFE)
    try:
        mood.SIDE_MARGIN, chat_state.CONCERN_SAFE = float("inf"), 0.0
        before = replay(path)
    finally:
        mood.SIDE_MARGIN, chat_state.CONCERN_SAFE = keep
    return {"side_margin": keep[0], "concern_safe": keep[1], "with_the_fixes": replay(path), "without_them": before}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--affect", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--log", default=str(ROOT / "validation" / "logs" / "exp_0079_mood_read.json"))
    a = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[0], "read_check": read_check(a.affect), "pilot_replay": replay_both(a.pilot)}
    pathlib.Path(a.log).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
