"""Quote records for the talk adapter: a facts block of 3-4 quotes by different people, "who said ...?",
and the speaker -- or, with the asked quote's line removed, "The facts don't say."

    python standin/data/build_quotes_sft.py --src <english_historical_quotes.json> [--items 24000]

Source: the English historical quotes set (quote, author, category; 24,022 rows; check its licence
before release). Every record is grounded by construction: the answer is the name on the asked line, and
the other lines are other people's quotes, so saying the wrong speaker is a binding error the gate can
see. Split by speaker hash, so a held speaker was never trained on.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, random, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_ground_sft import DASH, norm  # noqa: E402
from build_passage_sft import ABSENT  # noqa: E402

SOURCE = "English historical quotes (quote, author, category)"
ANSWERS = ("{a} said it.", "That was {a}.", "{a}.", "It's a quote from {a}.")


def split_of(author: str) -> str:
    return "held" if int(hashlib.sha256(norm(author).encode()).hexdigest(), 16) % 10 == 0 else "train"


def cue(quote: str, rng) -> str:
    """The asker's half-remembered version: the opening words, cut at a word boundary."""
    words = quote.split()
    k = min(len(words), rng.randint(6, 10))
    return " ".join(words[:k]).rstrip(",;:") + ("..." if k < len(words) else "")


def build(rows, n_items: int, seed: int):
    from ask import grounded_prose
    rng = random.Random(seed)
    rows = [r for r in rows if r.get("quote") and r.get("author") and 4 <= len(r["quote"].split()) <= 60]
    by_author = collections.defaultdict(list)
    for r in rows:
        by_author[" ".join(r["author"].split()).strip(", ")].append(" ".join(r["quote"].split()))
    authors = sorted(by_author)
    funnel, out = collections.Counter(), []
    for i in range(n_items):
        k = rng.choice([3, 3, 4])
        split = rng.choice(["train"] * 9 + ["held"])
        pool = [a for a in authors if split_of(a) == split]
        if len(pool) < k:
            continue
        who = rng.sample(pool, k)
        lines = [(a, rng.choice(by_author[a])) for a in who]
        ask = rng.randrange(k)
        a, qt = lines[ask]
        q = f'Who said "{cue(qt, rng)}"?'
        facts = [f'{x}{DASH}said: "{y}"' for x, y in lines]
        prompt = "Facts:\n" + "\n".join(f"- {f}" for f in facts) + f"\nQuestion: {q}\nAnswer:"
        ans = rng.choice(ANSWERS).format(a=a)
        if not grounded_prose(ans, facts, "")[0]:
            funnel["ungrounded"] += 1
            continue
        out.append({"id": f"quote:{i}", "family": "quote", "split": split, "prompt": prompt, "answer": ans,
                    "gold": [a], "forbid": [x for x in who if x != a], "source": SOURCE})
        funnel["answer"] += 1
        if rng.random() < 0.35:                          # the asked line removed: nobody here said it
            rest = [f for j, f in enumerate(facts) if j != ask]
            p2 = "Facts:\n" + "\n".join(f"- {f}" for f in rest) + f"\nQuestion: {q}\nAnswer:"
            out.append({"id": f"quote:{i}:absent", "family": "quote_absent", "split": split, "prompt": p2,
                        "answer": ABSENT, "gold": [], "forbid": [x for x in who if x != a], "source": SOURCE})
            funnel["absent"] += 1
    return out, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--items", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "quotes_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "quotes_sft.manifest.json"))
    args = ap.parse_args()
    rows = json.load(open(args.src, encoding="utf-8"))
    out, funnel = build(rows, args.items, args.seed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"source": SOURCE, "rows": len(rows), "items": args.items, "seed": args.seed,
                "funnel": dict(funnel.most_common()),
                "families": dict(collections.Counter(r["family"] for r in out)),
                "splits": dict(collections.Counter(r["split"] for r in out))}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in funnel.most_common():
        print(f"{k:12s} {v:7d}")


if __name__ == "__main__":
    main()
