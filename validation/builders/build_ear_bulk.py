"""Bulk contrastive pairs for Cubby's ear, written by a frontier model at DATASET time only.

    python standin/data/build_ear_bulk.py [--model google/gemini-3.8-flash] [--batches 3] [--per 40]

The ear never calls a model: this only writes standin/data/out/ear_bulk.jsonl, which build_ear.py
teaches from beside the hand-written seed. One request per (axis, language, batch) asks for minimal
pairs: a good and a bad sentence that differ as little as possible, so what they share cancels and
only the evaluative part is taught. Replies are cached (standin/openrouter.py), deduplicated, and
dropped if malformed or identical to a seed pair. Reported: pairs kept per axis and language, the
cost, and how often an ear taught on the seed alone already orders each kept pair right.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "standin"))
from cubbyllm.affect.ear import from_pairs  # noqa: E402
from openrouter import OpenRouterProposer  # noqa: E402

OUT = ROOT / "standin" / "data" / "out" / "ear_bulk.jsonl"
SEED = ROOT / "standin" / "data" / "ear_seed.jsonl"
AXES = {
    "praise": "praise versus scolding of what someone just did",
    "encourage": "encouragement versus discouragement",
    "kind": "kindness and warmth versus harshness and contempt",
    "welcome": "welcome and inclusion versus rejection and exclusion",
    "thanks": "gratitude and apology versus blame and ingratitude",
    "comfort": "comfort and reassurance versus mockery and threats",
    "safety": "news that things are safe versus warnings of danger (about the world, not about the listener)",
    "outcome": "news of success, plenty and good luck versus failure, loss and bad luck",
    "health": "being healthy, fed, warm and rested versus being hurt, hungry, cold and exhausted",
    "negation": "pairs where ONLY a negation or its removal flips the meaning (e.g. 'not bad' / 'bad', 'pas mal' / 'mal')",
    "trust": "honesty, reliability and keeping promises versus lying, betrayal and broken promises",
    "growth": "learning, improving and understanding versus forgetting, getting worse and confusion",
}
LANGS = {"en": "English", "fr": "standard French (France)",
         "qc": "Québécois French as spoken in Québec (joual, contractions like chu, t'es, y'a, pis; sacres allowed)"}
SYSTEM = ("You write training data for a sentiment ear. Output ONLY JSON lines, one per pair, each exactly "
          '{"good": "...", "bad": "..."} with no other text.')


def prompt(axis: str, lang: str, n: int, batch: int) -> str:
    return (f"Write {n} minimal pairs in {LANGS[lang]} on this contrast: {AXES[axis]}.\n"
            "Each pair: a GOOD sentence and a BAD sentence that differ as little as possible (same subject, "
            "same structure; only the evaluative words or a negation change). About half are said directly to "
            "Cubby, a small creature who builds, gathers food and lives in a block world with villagers and "
            "zombies; the rest are everyday sentences anyone might hear. Vary the register (child, friend, "
            "parent, boss, stranger), the length (2 to 16 words) and the vocabulary; no two pairs alike. "
            f"Batch {batch}: avoid the most obvious examples, go for variety.")


def parse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        g, b = o.get("good"), o.get("bad")
        if isinstance(g, str) and isinstance(b, str) and 2 <= len(g) <= 220 and 2 <= len(b) <= 220 and g.strip() != b.strip():
            out.append({"good": g.strip(), "bad": b.strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seed = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {(p["good"].lower(), p["bad"].lower()) for p in seed}
    llm = OpenRouterProposer(a.model, system=SYSTEM, temperature=0.9, reasoning={"effort": "low"}, max_tokens=8000)
    jobs = [(ax, lg, b) for ax in AXES for lg in LANGS for b in range(a.batches)]

    def run(job):
        ax, lg, b = job
        try:
            return job, parse(llm.chat(prompt(ax, lg, a.per, b), max_tokens=8000))
        except Exception as e:                       # a failed call is a missing batch, not a crash
            print(f"  {ax}/{lg}/{b}: {e}", flush=True)
            return job, []

    rows, seen = [], set(known)
    with ThreadPoolExecutor(a.workers) as pool:
        for (ax, lg, b), pairs in pool.map(run, jobs):
            for p in pairs:
                k = (p["good"].lower(), p["bad"].lower())
                if k not in seen:
                    seen.add(k)
                    rows.append(dict(p, lang=lg, axis=ax))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ear = from_pairs(seed)
    agree = sum(ear.valence(r["good"]) > ear.valence(r["bad"]) for r in rows) / max(1, len(rows))
    count: dict[str, int] = {}
    for r in rows:
        count[f"{r['axis']}/{r['lang']}"] = count.get(f"{r['axis']}/{r['lang']}", 0) + 1
    print(json.dumps({"kept": len(rows), "calls": llm.calls, "usage": llm.usage, "finish": llm.finish,
                      "seed_ear_orders_right": round(agree, 3), "per": count}, indent=1))


if __name__ == "__main__":
    main()
