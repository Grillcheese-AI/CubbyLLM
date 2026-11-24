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

**After training (Cowork's addendum, 2026-11-24): `--held FILE`** (`build_emotion_read_0079.py`'s held-out test:
emotions.jsonl's held fifth, GoEmotions' test, a machine-translated French sample) scores, per source, the answers in
the asked form (the strict parse found a petal), the petal right by the strict parse (as served: the gate) and by the
loose one; with the gate's lines. `--torch-ckpt` reads with the talk seat on torch (Colab; `talk_read_torch`), in
batches: the same prompt, the same parse, the same answer as `ModelAppraiser.read`.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
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
    if a.torch_ckpt:                                     # the talk seat on torch (Colab); --cubby is its adapter
        sys.path.insert(0, str(ROOT / "validation"))
        from talk_read_torch import TorchTalk
        return TorchTalk(a.torch_ckpt, a.cubby, a.tokenizer)
    if a.cubby:
        # the TALK seat, as the host serves the perception read (Cubby450mEmitter is the programs seat: it wraps
        # the prompt as "Question:/Program:" and decodes a program, so no petal could ever come back; 2026-11-21)
        from cubbyllm.emit.seats_450m import Cubby450mTrunk
        from cubbyllm.emit.talk import Cubby450mTalk
        return Cubby450mTalk(Cubby450mTrunk(a.export, a.tokenizer, {"talk": a.cubby}))
    from standin.emitter import LlamaCppEmitter, LlamaServerEmitter
    return LlamaCppEmitter(a.gguf) if a.gguf else LlamaServerEmitter(a.server)


class Capture:
    """The emitter, its whole last answer kept (ModelAppraiser keeps 80 characters of it)."""

    def __init__(self, emitter):
        self.emitter, self.text = emitter, ""

    def emit(self, prompt: str, **kw) -> str:
        self.text = self.emitter.emit(prompt, **kw)
        return self.text


def loose_petal(text: str) -> str | None:
    """--loose (Cowork, 2026-11-23): the first Plutchik petal named anywhere in the answer, else the petal of the first
    GoEmotions label named (`mood.LABEL_PETAL`); None when it names neither."""
    from cubbyllm.affect.mood import LABEL_PETAL
    t = text.lower()
    hits = [(m.start(), p) for p in PETALS for m in [re.search(rf"\b{p}\b", t)] if m]
    if hits:
        return min(hits)[1]
    hits = [(m.start(), LABEL_PETAL[w]) for w in LABEL_PETAL for m in [re.search(rf"\b{w}\b", t)] if m]
    return min(hits)[1] if hits else None


class Reader:
    """One read: the petal (strict: `parse_emotion`, as served; --loose: `loose_petal`), and whether the answer kept
    the asked "emotion -- petal" form (the strict parse found a petal), counted apart."""

    def __init__(self, app: ModelAppraiser, cap: Capture, loose: bool):
        self.app, self.cap, self.loose = app, cap, loose
        self.in_form = self.reads = 0

    def petal(self, text: str) -> str | None:
        strict = self.app.read(text, "en").get("petal")
        self.reads += 1
        self.in_form += strict is not None
        return loose_petal(self.cap.text) if self.loose and strict is None else strict


def emotions(path: str, app: Reader, limit: int) -> dict:
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
        petal = app.petal(text)
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


def mental_health(path: str, app: Reader, per_status: int, words: int, seed: int) -> dict:
    by: dict[str, list[str]] = collections.defaultdict(list)
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        by[r["status"]].append(" ".join(str(r["text"]).split()[:words]))
    out = {}
    for status, texts in sorted(by.items()):
        rng = random.Random(seed)
        sample = rng.sample(texts, min(per_status, len(texts)))
        petals = collections.Counter(app.petal(t) or "none" for t in sample)
        distress = sum(n for p, n in petals.items() if p in DISTRESS)
        out[status] = {"sampled": len(sample), "of": len(texts), "petals": dict(petals.most_common()),
                       "distress_petal": distress,
                       "expected_petal": sum(n for p, n in petals.items() if p in EXPECTED.get(status, set()))}
    return out


def held_read(path: str, emitter, app: ModelAppraiser) -> dict:
    """The held-out test: per source, in the asked form, right by the strict parse (the gate) and by the loose one."""
    from cubbyllm.affect.perception import EMOTION_PROMPT, EMOTION_PROMPT_FR, parse_emotion
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    answers: list = [None] * len(rows)
    inner = getattr(emitter, "emitter", emitter)
    for lang in ("en", "fr"):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        prompts = [(EMOTION_PROMPT_FR if lang == "fr" else EMOTION_PROMPT).format(t=" ".join(rows[i]["text"].split()))
                   for i in idx]
        if hasattr(inner, "emit_batch"):
            for s in range(0, len(idx), 512):
                got = inner.emit_batch(prompts[s:s + 512], context="talk", max_new_tokens=app.max_new_tokens)
                for i, g in zip(idx[s:s + 512], got):
                    answers[i] = g
        else:
            for i in idx:
                app.read(rows[i]["text"], lang)
                answers[i] = emitter.text
    by = collections.defaultdict(collections.Counter)
    for r, ans in zip(rows, answers):
        strict = parse_emotion(ans or "")[1]
        loose = strict or loose_petal(ans or "")
        c = by[r["source"]]
        c["texts"] += 1
        c["in_form"] += strict is not None
        c["right_strict"] += strict == r["petal"]
        c["right_loose"] += loose == r["petal"]
    out = {s: {**c, **{f"{k}_share": round(c[k] / c["texts"], 4) for k in ("in_form", "right_strict", "right_loose")}}
           for s, c in sorted(by.items())}
    every = sum(c["texts"] for c in by.values())
    form = sum(c["in_form"] for c in by.values()) / max(1, every)
    e, g = out.get("emotions.jsonl", {}), out.get("goemotions_test", {})
    out["gate"] = {"asked form >= 0.95 (all answers)": round(form, 4) >= 0.95,
                   "emotions.jsonl petal >= 0.60": e.get("right_strict_share", 0) >= 0.60,
                   "goemotions test petal >= 0.50": g.get("right_strict_share", 0) >= 0.50,
                   "kill: emotions.jsonl petal < 0.35": e.get("right_strict_share", 0) < 0.35,
                   "asked form share": round(form, 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emotions", default="")
    ap.add_argument("--mental-health", default="")
    ap.add_argument("--held", default="", help="build_emotion_read_0079's held-out test (after training)")
    ap.add_argument("--torch-ckpt", default="", help="the torch base: the talk seat on torch (Colab), --cubby the "
                    "adapter dir, --tokenizer")
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
    ap.add_argument("--loose", action="store_true", help="a petal named anywhere, else a GoEmotions label's (Cowork, "
                    "2026-11-23); the asked form counted apart. perception.parse_emotion is unchanged")
    a = ap.parse_args()
    if not (a.dry or a.gguf or a.server or a.cubby or a.torch_ckpt):
        raise SystemExit("give --gguf PATH, --server URL, --cubby ADAPTER (with --export, --tokenizer) or --dry")
    cap = Capture(emitter_from(a))
    app = Reader(ModelAppraiser(cap), cap, a.loose)
    name = "dry" if a.dry else (a.gguf or a.server or a.cubby)
    out = {"what": __doc__.split("\n\n")[0], "emitter": pathlib.Path(name).name if a.cubby else name,
           "parse": "loose (a petal anywhere, else a GoEmotions label)" if a.loose else "strict (parse_emotion)"}
    if a.held:
        out["held"] = held_read(a.held, cap, app.app)
    if a.emotions:
        out["emotions"] = emotions(a.emotions, app, a.limit)
    if a.mental_health:
        out["mental_health"] = mental_health(a.mental_health, app, a.per_status, a.words, a.seed)
    out["answers_in_the_asked_form"] = {"in_form": app.in_form, "reads": app.reads}
    log = ROOT / "validation" / "logs" / f"exp_0079_appraiser_read{a.tag}.json"
    log.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
