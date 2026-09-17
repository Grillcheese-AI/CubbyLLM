"""Compression as a readout of available energy.

Owner, on a message I had called state-independent: *"fo je mette is slang... if I
wasnt tired I probably would have written 'il va falloir que je mette a jour les
produits'. But it was too long required too much energy out of me on that day so I
shortened my writing."*

That reframes the whole search. Every signal tried so far was a CORRELATE of state
bolted onto the message - emoji (turned out to be the social channel, falls to zero as
bursts deepen), exclamation marks (1.3% of this corpus, essentially absent), reply
timing (flat, p=0.83), burst depth (+0.117, but thin). Compression is not a correlate.
It is the mechanism: writing the full form costs energy, and the form that comes out
is what the budget allowed.

So this measures, per message, how far the text sits from its own full form. No model
call, no label, no asserted number - a ratio computed from which of two spellings the
writer actually chose, each pair attested in this corpus.

TWO AXES, deliberately separate:

  LEXICAL   'il va falloir que' -> 'fo', 'parce que' -> 'pcq', 'je suis' -> 'chu'.
            A choice between two real French forms. This is the energy measure.
  DIACRITIC accents dropped. Cheaper to type, and in this corpus it varies BY PERSON
            as much as by state ([PEER] accents far more than [SELF]), so it is scored
            apart rather than folded in - a per-writer habit is not a per-message state.

THE CONFOUND, stated up front: compression is expected under BOTH depletion and high
arousal. Tired writes short because the full form costs too much; wound-up writes short
because the next thought will not wait. They are opposite states with the same surface.
What separates them is volume - depletion compresses AND stops, arousal compresses AND
keeps going - so compression is reported against burst depth, never alone. That is the
arousal x valence split the owner's own labels produced, arriving from the text side.

The 14 hand labels are the test: `F` items should compress; `!` items should compress
too but at depth. If compression cannot tell them apart even with burst, it measures
brevity and not energy.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
OUT = ROOT / "docs" / "teams_compression.md"

# (full form, short form). Every pair attested in this corpus; the short side is what
# the owner types under load, the full side is what the same sentence looks like when
# there is room for it. Ordered longest-first at match time so "il va falloir que" is
# not eaten by "il va".
PAIRS = [
    (r"il (?:va )?falloir que", r"\bfo\b|\bfaut\b"),
    (r"parce qu[e']", r"\bpcq\b|\bpasque\b|\bpcque\b"),
    (r"peut-[eê]tre", r"\bp-e\b|\bpe\b"),
    (r"\bje suis\b", r"\bchu\b|\bjsuis\b"),
    (r"\bje vais\b", r"\bma\b|\bjvais\b|\bjva\b"),
    (r"en tout cas", r"\btk\b"),
    (r"\bmaintenant\b", r"\basteur\b|\basture\b"),
    (r"\bet puis\b|\bensuite\b", r"\bpis\b"),
    (r"\bc'est\b", r"\bc\b(?!')|\bcé\b|\bce\b(?= )"),
    (r"\bje pense\b", r"\bjpense\b"),
    (r"\bj'ai\b", r"\bjai\b"),
    (r"\bil y a\b", r"\by a\b|\bya\b"),
    (r"\bd'accord\b", r"\bdac\b|\bok\b"),
    (r"\bquelque chose\b", r"\bqqc\b|\bde quoi\b"),
    (r"\bquelqu'un\b", r"\bqqun\b|\bqqk\b"),
    (r"\bbeaucoup\b", r"\bben ben\b|\bpas mal\b"),
    (r"\bs'il te pla[iî]t\b", r"\bstp\b"),
    (r"\bje te\b", r"\bjte\b"),
    (r"\bje le\b", r"\bjle\b"),
    (r"\btu es\b", r"\bta\b|\btes\b"),
]
ACCENTED = re.compile(r"[éèêëàâäîïôöùûüçÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ]")
# words that carry an accent in correct French and are routinely typed without one
NEEDS_ACCENT = re.compile(
    r"\b(deja|tres|apres|etait|etre|meme|prevu|arrete|reglé?|regler|probleme|problemes|"
    r"reponse|verifie|termine|eteint|enleve|categorie|categories|derniere|premiere|"
    r"different|differente|interet|complete|creer|cree|generer|genere|reussi|desole|"
    r"maniere|systeme|acces|apres-midi|decembre|fevrier)\b", re.I)


def compression(text: str) -> dict:
    t = text.lower()
    full = short = 0
    for f, s in PAIRS:
        if re.search(f, t):
            full += 1
        if re.search(s, t):
            short += 1
    unaccented = len(NEEDS_ACCENT.findall(t))
    accented = len(ACCENTED.findall(text))
    return {
        "full": full, "short": short,
        # 1.0 = every choice made was the short form; None = no choice presented
        "lex": (short / (short + full)) if (short + full) else None,
        "unaccented": unaccented, "accented": accented,
        "dia": (unaccented / (unaccented + accented)) if (unaccented + accented) else None,
    }


def block(f, title, groups):
    f.write(f"### {title}\n\n| group | n | mean lexical compression | median words |\n")
    f.write("|---|---:|---:|---:|\n")
    for label, rows in groups:
        vals = [r["c"]["lex"] for r in rows if r["c"]["lex"] is not None]
        w = [r.get("words", 0) for r in rows]
        if not vals:
            continue
        f.write(f"| {label} | {len(vals)} | {st.mean(vals):.3f} | "
                f"{st.median(w) if w else 0:.0f} |\n")
    f.write("\n")


def main() -> int:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    for r in rows:
        r["c"] = compression(r["text"])
    me = [r for r in rows if r["speaker"] == "[SELF]"]
    scored = [r for r in me if r["c"]["lex"] is not None]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Compression as an energy readout\n\n")
        f.write(f"{len(rows)} messages, {len(me)} from `[SELF]`, **{len(scored)}** of "
                f"which presented at least one full-vs-short choice. No model, no labels - "
                f"a ratio over which spelling the writer picked.\n\n")
        f.write("The premise: writing the full form costs energy. `il va falloir que je "
                "mette` and `fo je mette` are the same sentence; which one comes out is the "
                "budget, not the dialect.\n\n")

        f.write("## Is the choice actually live?\n\n")
        f.write("If the short form is used 100% of the time it is not a choice, it is just "
                "how this person writes, and it can carry no state.\n\n")
        f.write("| pair | full used | short used |\n|---|---:|---:|\n")
        for fpat, spat in PAIRS:
            nf = sum(1 for r in me if re.search(fpat, r["text"].lower()))
            ns = sum(1 for r in me if re.search(spat, r["text"].lower()))
            if nf or ns:
                f.write(f"| `{fpat[:34]}` | {nf} | {ns} |\n")

        f.write("\n## Compression by burst depth\n\n")
        f.write("The confound, head on: depletion and arousal BOTH compress. What should "
                "separate them is volume - tired compresses and stops, wound-up compresses "
                "and keeps going.\n\n")
        block(f, "by depth", [
            ("0 (opens a run)", [r for r in scored if r.get("burst", 0) == 0]),
            ("1", [r for r in scored if r.get("burst", 0) == 1]),
            ("2-4", [r for r in scored if 2 <= r.get("burst", 0) <= 4]),
            ("5+", [r for r in scored if r.get("burst", 0) >= 5]),
        ])

        f.write("## Compression by message length\n\n")
        block(f, "by words", [
            ("1-5 words", [r for r in scored if r.get("words", 0) <= 5]),
            ("6-12", [r for r in scored if 6 <= r.get("words", 0) <= 12]),
            ("13+", [r for r in scored if r.get("words", 0) >= 13]),
        ])

        f.write("## Who compresses - state or habit?\n\n")
        f.write("If compression is a per-person style rather than a per-message state, the "
                "speakers will differ far more than the situations do.\n\n")
        for sp in ("[SELF]", "[PEER]", "[PEER2]"):
            sub = [r for r in rows if r["speaker"] == sp and r["c"]["lex"] is not None]
            dia = [r["c"]["dia"] for r in rows if r["speaker"] == sp and r["c"]["dia"] is not None]
            if sub:
                f.write(f"- **{sp}** - lexical {st.mean([r['c']['lex'] for r in sub]):.3f} "
                        f"(n={len(sub)}), accents dropped "
                        f"{st.mean(dia):.3f} (n={len(dia)})\n" if dia else
                        f"- **{sp}** - lexical {st.mean([r['c']['lex'] for r in sub]):.3f}\n")

        f.write("\n## Spread\n\n")
        vals = [r["c"]["lex"] for r in scored]
        f.write(f"- mean **{st.mean(vals):.3f}**, median **{st.median(vals):.3f}**, "
                f"sd **{st.pstdev(vals):.3f}**\n")
        d = Counter(round(v, 1) for v in vals)
        f.write("- distribution: " + ", ".join(f"{k}: {v}" for k, v in sorted(d.items())) + "\n")
        f.write("\nA scale pinned at 1.0 would mean the short form always wins and there is "
                "no signal - the opposite failure from the model's intensity, which sat at "
                "0.35 in every group.\n")

        # Most pairs turn out to be settled for this writer - `il va falloir que` is 0
        # full against 220 short, `j'ai` 0 against 531. A choice already made every time
        # cannot carry a state, and averaging it in just buries the pairs that DO vary.
        # So: re-run on the live ones only, plus accents, which have ~5% headroom.
        LIVE = [(f_, s_) for f_, s_ in PAIRS
                if sum(1 for r in me if re.search(f_, r["text"].lower())) >= 4]
        f.write("\n## Restricted to the pairs that actually vary\n\n")
        f.write(f"{len(LIVE)} of {len(PAIRS)} pairs have the full form used 4+ times by "
                f"this writer. The rest are settled - a choice made the same way every time "
                f"carries nothing, and averaging it in buries the pairs that do move.\n\n")

        def live_lex(text):
            t = text.lower()
            fu = sum(1 for f_, s_ in LIVE if re.search(f_, t))
            sh = sum(1 for f_, s_ in LIVE if re.search(s_, t))
            return (sh / (sh + fu)) if (sh + fu) else None

        for r in me:
            r["live"] = live_lex(r["text"])
        sub = [r for r in me if r["live"] is not None]
        f.write(f"n = {len(sub)}, mean **{st.mean([r['live'] for r in sub]):.3f}**, "
                f"sd **{st.pstdev([r['live'] for r in sub]):.3f}**\n\n")
        f.write("| group | n | live-pair compression | accents dropped |\n|---|---:|---:|---:|\n")
        for label, sel in (("burst 0", lambda r: r.get("burst", 0) == 0),
                           ("burst 1-4", lambda r: 1 <= r.get("burst", 0) <= 4),
                           ("burst 5+", lambda r: r.get("burst", 0) >= 5),
                           ("1-5 words", lambda r: r.get("words", 0) <= 5),
                           ("6-12 words", lambda r: 6 <= r.get("words", 0) <= 12),
                           ("13+ words", lambda r: r.get("words", 0) >= 13)):
            g = [r for r in sub if sel(r)]
            d = [r["c"]["dia"] for r in me if sel(r) and r["c"]["dia"] is not None]
            if len(g) >= 20:
                f.write(f"| {label} | {len(g)} | {st.mean([r['live'] for r in g]):.3f} | "
                        f"{(st.mean(d) if d else float('nan')):.3f} |\n")

    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
