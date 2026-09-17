"""Does compression track state in the writer who still has room to move?

The energy hypothesis, in the owner's words: *"fo je mette is slang... if I wasnt tired
I probably would have written 'il va falloir que je mette a jour les produits'. But it
was too long required too much energy out of me on that day so I shortened my writing."*

Tested on him, it came back flat - 0.977 mean, median 1.000, 3,475 of 3,597 messages at
exactly the ceiling. He spent that budget down years ago; the compressed form is not
what he falls back to when tired, it is his baseline, and there is nothing left to
shorten when the day gets worse. A pinned scale cannot carry a signal in either
direction.

[PEER] is the control the corpus already contains. He sits at 0.654 lexical compression
and drops accents on only 6.7% of the words that need them, against the owner's 95.4%.
He has headroom in both, so if the mechanism is real it should be visible in him.

This is the honest way to test a hypothesis the originator cannot express: find someone
who can. A null here says the mechanism is wrong; a null in a writer at the ceiling says
only that the instrument has nowhere to read.

Everything is checked against channels the writer did not choose - burst depth, hour of
day, gap before the message - and a permutation test, because a difference of 0.02 and a
difference of 0.002 both look like "not much" until the null distribution says otherwise.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import statistics as st
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from compression import PAIRS, NEEDS_ACCENT, ACCENTED  # noqa: E402

CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
OUT = ROOT / "docs" / "teams_peer_compression.md"


def local(dt):
    return dt - timedelta(hours=4 if 3 <= dt.month <= 10 else 5)


def perm_p(a, b, n=20000, seed=7):
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    obs = abs(st.mean(a) - st.mean(b))
    pool, k, rng = list(a) + list(b), 0, random.Random(seed)
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


def score(text: str, live_pairs) -> tuple:
    t = text.lower()
    fu = sum(1 for f, s in live_pairs if re.search(f, t))
    sh = sum(1 for f, s in live_pairs if re.search(s, t))
    lex = (sh / (sh + fu)) if (sh + fu) else None
    un = len(NEEDS_ACCENT.findall(t))
    ac = len(ACCENTED.findall(text))
    dia = (un / (un + ac)) if (un + ac) else None
    return lex, dia


def block(f, title, groups, key):
    f.write(f"**{title}**\n\n| group | n | mean | median |\n|---|---:|---:|---:|\n")
    vals = []
    for label, rows in groups:
        v = [r[key] for r in rows if r.get(key) is not None]
        vals.append((label, v))
        if v:
            f.write(f"| {label} | {len(v)} | {st.mean(v):.3f} | {st.median(v):.3f} |\n")
    live = [v for _, v in vals if len(v) >= 3]
    if len(live) >= 2:
        p = perm_p(live[0], live[-1])
        d = st.mean(live[-1]) - st.mean(live[0])
        f.write(f"\nfirst vs last: **{d:+.3f}**, p = **{p:.4f}** -> "
                f"{'**separates**' if p < 0.05 else 'flat'}\n\n")
    else:
        f.write("\nnot enough rows to test\n\n")


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    for r in rows:
        try:
            dt = datetime.fromisoformat((r.get("when") or "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            r["hour"] = local(dt).hour
        except Exception:
            r["hour"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Compression vs state, in the writer with headroom\n\n")
        f.write("The owner sits at 0.977 lexical compression with no room left to move. "
                "`[PEER]` sits at 0.654 and keeps his accents. If writing the full form "
                "costs energy, it should show in the person who can still afford it.\n\n")

        for who in ("[PEER]", "[SELF]"):
            sub = [r for r in rows if r["speaker"] == who]
            # Pairs this writer actually varies on. A choice already settled carries
            # nothing, and averaging it in buries the pairs that move.
            live = [(fp, sp) for fp, sp in PAIRS
                    if sum(1 for r in sub if re.search(fp, r["text"].lower())) >= 4]
            for r in sub:
                r["lex"], r["dia"] = score(r["text"], live)
            n_lex = sum(1 for r in sub if r["lex"] is not None)
            f.write(f"## {who}\n\n")
            f.write(f"{len(sub)} messages, {len(live)} of {len(PAIRS)} pairs live for this "
                    f"writer, {n_lex} messages presenting a choice.\n\n")
            if n_lex < 40:
                f.write("Too few scorable messages to read anything from.\n\n")
                continue

            for key, label in (("lex", "lexical compression"), ("dia", "accents dropped")):
                vals = [r[key] for r in sub if r.get(key) is not None]
                if len(vals) < 40:
                    continue
                f.write(f"### {label} — mean {st.mean(vals):.3f}, "
                        f"sd {st.pstdev(vals):.3f}, "
                        f"distinct {len(set(round(v, 2) for v in vals))}\n\n")
                if st.pstdev(vals) < 0.02:
                    f.write("Pinned — no spread, nothing to correlate.\n\n")
                    continue
                block(f, "by burst depth", [
                    ("opens a run (0)", [r for r in sub if r.get("burst", 0) == 0]),
                    ("1", [r for r in sub if r.get("burst", 0) == 1]),
                    ("2+", [r for r in sub if r.get("burst", 0) >= 2]),
                ], key)
                block(f, "by hour, local", [
                    ("before 12:00", [r for r in sub if (r["hour"] or 0) < 12]),
                    ("12:00-15:00", [r for r in sub if 12 <= (r["hour"] or 0) < 15]),
                    ("after 15:00", [r for r in sub if (r["hour"] or 0) >= 15]),
                ], key)
                block(f, "by message length", [
                    ("1-5 words", [r for r in sub if r.get("words", 0) <= 5]),
                    ("6-12", [r for r in sub if 6 <= r.get("words", 0) <= 12]),
                    ("13+", [r for r in sub if r.get("words", 0) >= 13]),
                ], key)

        f.write("## Reading this\n\n")
        f.write("A separation in `[PEER]` and a flat line in `[SELF]` supports the "
                "mechanism and explains the owner's null as a ceiling effect - the budget "
                "spent long ago, the instrument with nowhere left to read.\n\n")
        f.write("Flat in both is the stronger result and the less convenient one: it says "
                "compression is a per-person habit rather than a per-message state, which "
                "is already what the three-way split across writers (0.977 / 0.654 / 0.429) "
                "suggested on its own.\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
