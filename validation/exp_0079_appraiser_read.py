"""exp_0079_appraiser_read -- does the trunk read the user's emotion as well as 0079's training data will assume?
(order 0079, Cowork, 2026-11-21; set up for Nick's next GPU session.)

The final run drafts with the read the host serves with (`perception.ModelAppraiser`), the user model's label standing in
for the trunk. This measures the trunk itself, through the same `ModelAppraiser.read`, on two of Nick's files (off the
repo); counts only, written to `validation/logs/exp_0079_appraiser_read{--tag}.json`:

1. **`emotions.jsonl`** (short English messages, each with its Plutchik primary and an intensity): the petal the
   trunk names against the label, overall, by petal, and the confusions, on the distinct texts whose primary is one
   petal (1,714; the rows with two primaries or none of the eight are left out). A miss with no petal named is counted
   apart.
2. **`mental_health_svc.jsonl`** (posts labelled Normal, Depression, Suicidal, Anxiety): a sample per status (cut to
   its first `--words` words), and how often the trunk names a petal of distress there (sadness or fear), against
   how often it does on Normal. This is the read Cubby's care in a hard moment depends on; it is a measure, not a
   detector, and no post leaves the machine or enters the repo.

    python validation/exp_0079_appraiser_read.py --emotions <emotions.jsonl> --mental-health <mental_health_svc.jsonl> \\
        (--gguf PATH | --server URL | --cubby ADAPTER --export DIR --tokenizer JSON | --dry) [--per-status 300] [--tag _x]

`--dry` runs the whole pipeline with an emitter that names nothing (every read a miss), to check the files and the
script before a GPU session.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubbyllm.affect.perception import PETALS, ModelAppraiser  # noqa: E402

DISTRESS = {"sadness", "fear"}
EXPECTED = {"Depression": {"sadness"}, "Suicidal": {"sadness", "fear"}, "Anxiety": {"fear"}}


class DryEmitter:
    def emit(self, prompt: str, **_kw) -> str:
        return ""


def emitter_from(a: argparse.Namespace):
    if a.dry:
        return DryEmitter()
    sys.path.insert(0, str(ROOT / "standin" / "data"))
    if a.cubby:
        # the TALK seat, as the host serves the perception read (Cubby450mEmitter is the programs seat: it wraps
        # the prompt as "Question:/Program:" and decodes a program, so no petal could ever come back; 2026-11-21)
        from cubbyllm.emit.seats_450m import Cubby450mTrunk
        from cubbyllm.emit.talk import Cubby450mTalk
        return Cubby450mTalk(Cubby450mTrunk(a.export, a.tokenizer, {"talk": a.cubby}))
    from standin.emitter import LlamaCppEmitter, LlamaServerEmitter
    return LlamaCppEmitter(a.gguf) if a.gguf else LlamaServerEmitter(a.server)


def emotions(path: str, app: ModelAppraiser, limit: int) -> dict:
    seen, rows = set(), []
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        p = r["plutchik"]["primary"]
        if isinstance(p, str) and p.lower() in PETALS and r["text"] not in seen:
            seen.add(r["text"])
            rows.append((r["text"], p.lower()))
    rows = rows[:limit] if limit else rows
    hit, none, by, conf = 0, 0, collections.defaultdict(lambda: [0, 0]), collections.Counter()
    for text, gold in rows:
        petal = app.read(text, "en").get("petal")
        by[gold][1] += 1
        if petal is None:
            none += 1
        elif petal == gold:
            hit += 1
            by[gold][0] += 1
        else:
            conf[f"{gold} -> {petal}"] += 1
    return {"texts": len(rows), "petal_right": hit, "no_petal_named": none,
            "by_petal": {g: {"right": v[0], "of": v[1]} for g, v in sorted(by.items())},
            "confusions": dict(conf.most_common(15))}


def mental_health(path: str, app: ModelAppraiser, per_status: int, words: int, seed: int) -> dict:
    by: dict[str, list[str]] = collections.defaultdict(list)
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        by[r["status"]].append(" ".join(str(r["text"]).split()[:words]))
    out = {}
    for status, texts in sorted(by.items()):
        rng = random.Random(seed)
        sample = rng.sample(texts, min(per_status, len(texts)))
        petals = collections.Counter(app.read(t, "en").get("petal") or "none" for t in sample)
        distress = sum(n for p, n in petals.items() if p in DISTRESS)
        out[status] = {"sampled": len(sample), "of": len(texts), "petals": dict(petals.most_common()),
                       "distress_petal": distress,
                       "expected_petal": sum(n for p, n in petals.items() if p in EXPECTED.get(status, set()))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emotions", required=True)
    ap.add_argument("--mental-health", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--server")
    ap.add_argument("--cubby", help="the 450M talk adapter dir, with --export and --tokenizer")
    ap.add_argument("--export")
    ap.add_argument("--tokenizer")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="emotions.jsonl texts to read (0: all)")
    ap.add_argument("--per-status", type=int, default=300)
    ap.add_argument("--words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=79)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not (a.dry or a.gguf or a.server or a.cubby):
        raise SystemExit("give --gguf PATH, --server URL, --cubby ADAPTER (with --export, --tokenizer) or --dry")
    app = ModelAppraiser(emitter_from(a))
    out = {"what": __doc__.split("\n\n")[0], "emitter": "dry" if a.dry else (a.gguf or a.server or a.cubby),
           "emotions": emotions(a.emotions, app, a.limit),
           "mental_health": mental_health(a.mental_health, app, a.per_status, a.words, a.seed)}
    log = ROOT / "validation" / "logs" / f"exp_0079_appraiser_read{a.tag}.json"
    log.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
