"""Build Cubby's ear (standin/ear.py) from contrastive sentences, and say how well it hears.

    python standin/data/build_ear.py [--corpora DIR] [--bulk standin/data/out/ear_bulk.jsonl]
                                     [--out standin/data/out/ear.npz]

Sources, each optional but the first:
  ear_seed.jsonl         hand-written minimal pairs, EN / FR / Québécois (committed)
  --bulk                 generated pairs (build_ear_bulk.py), dataset-time only
  --corpora DIR          a folder of labelled affect sets, if present: amygdala_affect.jsonl and
                         affect_from_convos.jsonl ({"text", "valence"}), ei_11.jsonl (a query with
                         its sentiment score), taught at half weight: they are people describing
                         their own state, not talk to Cubby (ei_11's warm/plain reply pairs were
                         tried and dropped: they cost 4 points on the probes).
                         Defaults to $CUBBY_DATASETS/domains/conversation.
  the lexicon            ear_lexicon.Lexicon.find() (potion word vectors), unless --no-lexicon.

Held out: one pair in five of every source (by a hash of its text, so the split never moves).
Reported: on held-out pairs, how often the good one hears better than the bad one, and how often
each lands on its own side of zero; on held-out labelled sentences, sign agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standin"))
from cubbyllm.affect.ear import from_pairs  # noqa: E402
from cubbyllm.affect.ear_lexicon import Lexicon  # noqa: E402

SEED = ROOT / "standin" / "data" / "ear_seed.jsonl"


def held(text: str) -> bool:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 5 == 0


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def corpora(folder: Path) -> tuple[list[dict], list[dict]]:
    pairs, sents = [], []
    for name in ("amygdala_affect.jsonl", "affect_from_convos.jsonl"):
        for r in jsonl(folder / name):
            if isinstance(r.get("text"), str) and isinstance(r.get("valence"), (int, float)):
                sents.append({"text": r["text"], "valence": float(r["valence"]), "w": 0.5, "src": name})
    for r in jsonl(folder / "ei_11.jsonl"):
        q = (r.get("context_input") or {}).get("query")
        s = (r.get("sentiment") or {}).get("score")
        if isinstance(q, str) and isinstance(s, (int, float)):
            sents.append({"text": q, "valence": max(-1.0, min(1.0, float(s))), "w": 0.5, "src": "ei_11"})
    return pairs, sents


def score(ear, pairs: list[dict], sents: list[dict]) -> dict:
    order = side = 0
    for p in pairs:
        g, b = ear.valence(p["good"]), ear.valence(p["bad"])
        order += g > b
        side += (g > 0) + (b < 0)
    sign = sum((ear.valence(s["text"]) > 0) == (s["valence"] > 0) for s in sents if abs(s["valence"]) >= 0.3)
    n_s = sum(1 for s in sents if abs(s["valence"]) >= 0.3)
    return {"pairs": len(pairs), "order": round(order / max(1, len(pairs)), 3),
            "side": round(side / max(1, 2 * len(pairs)), 3),
            "sentences": n_s, "sign": round(sign / max(1, n_s), 3)}


PROBES = [("Great work, buddy!", 1), ("You're hopeless.", -1), ("Careful, there's lava!", -1), ("We're safe now.", 1),
          ("Je suis content de toi.", 1), ("T'es poche.", -1), ("Bien joué!", 1), ("Quelle catastrophe.", -1),
          ("Not bad, Cubby.", 1), ("Pas mal du tout!", 1), ("That's not good.", -1), ("Ce n'est pas bien.", -1),
          ("Thank you so much!", 1), ("Get lost.", -1), ("C'est malade, bravo!", 1), ("Ayoye, ça fait mal.", -1),
          ("Nice house!", 1), ("Ugly house.", -1), ("I'm proud of you.", 1), ("You disappoint me.", -1),
          ("Good night, sleep well.", 1), ("The zombies broke the door!", -1), ("Lâche pas, t'es bon!", 1),
          ("T'es un nul.", -1)]


def probes(ear) -> dict:
    """Sentences in none of the sources, labelled by hand: how many land on their side."""
    right = [t for t, y in PROBES if (ear.valence(t) > 0) == (y > 0)]
    return {"n": len(PROBES), "right": round(len(right) / len(PROBES), 3),
            "wrong": [t for t, _ in PROBES if t not in right]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default=str(Path(os.environ.get("CUBBY_DATASETS", "")) / "domains" / "conversation")
                    if os.environ.get("CUBBY_DATASETS") else "")
    ap.add_argument("--bulk", default=str(ROOT / "standin" / "data" / "out" / "ear_bulk.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "ear.npz"))
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "validation" / "logs" / f"ear_build_{time.strftime('%Y-%m-%d')}.json"))
    a = ap.parse_args()
    seed = jsonl(SEED)
    for p in seed:
        p["src"], p["w"] = "seed:" + p.get("lang", "?"), 2.0     # hand-written: the clearest contrasts
    bulk = [dict(p, src="bulk:" + p.get("lang", "?")) for p in jsonl(Path(a.bulk)) if p.get("good") and p.get("bad")]
    cp, cs = corpora(Path(a.corpora)) if a.corpora else ([], [])
    pairs, sents = seed + bulk + cp, cs
    train_p = [p for p in pairs if not held(p["good"])]
    test_p = [p for p in pairs if held(p["good"])]
    train_s = [s for s in sents if not held(s["text"])]
    test_s = [s for s in sents if held(s["text"])]
    lex = None if a.no_lexicon else Lexicon.find()
    t0 = time.perf_counter()
    ear = from_pairs(train_p, train_s, lexicon=lex)
    report = {"built": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.perf_counter() - t0, 1),
              "taught": {"pairs": len(train_p), "sentences": len(train_s)}, "lexicon": lex and lex.name,
              "held_out": {}, "examples": {}}
    groups = sorted({p["src"].split(":")[0] + (":" + p["src"].split(":")[1] if p["src"].startswith(("seed", "bulk")) else "")
                     for p in test_p} | {s["src"] for s in test_s})
    for g in groups:
        report["held_out"][g] = score(ear, [p for p in test_p if p["src"] == g], [s for s in test_s if s["src"] == g])
    for t in ("Good job, Cubby!", "That's not bad at all.", "Watch out, the zombies are coming!",
              "Bravo, t'es capable!", "C'est pas pire pantoute!", "You're useless.", "Je suis fier de toi."):
        report["examples"][t] = ear.hear(t)
    report["probes"] = probes(ear)
    ear = from_pairs(pairs, sents, lexicon=lex)       # the shipped ear: everything, held-out included
    ear.save(a.out)
    report["out"] = os.path.relpath(a.out, ROOT)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
