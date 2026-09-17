"""Profile a ChatGPT data export: how much of it is the owner's own French?

Only the USER turns matter here. An assistant turn is a model's French, which is the
formal/translated register we already know fails - the whole point of harvesting is to
get text a person actually typed.

Writes a report file rather than printing, because the export is large and a Windows
console mangles accented output on the way out (which is how a clean UTF-8 file gets
mistaken for mojibake).
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = r"G:\chatgpt_mydata"
OUTDIR = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out"
REPORT = os.path.join(OUTDIR, "chatgpt_export_profile.md")
CORPUS = os.path.join(OUTDIR, "chatgpt_fr_user_turns.jsonl")

# Same two probes as scan_fr: is it French at all, and is it the SPOKEN register?
FR = re.compile(r"\b(les|des|une|dans|pour|avec|mais|nous|vous|elle|cette|comme|tout|"
                r"plus|bien|faire|être|sont|est|qui|que|aux|sur|je|tu|on|ne|pas|ça|ca)\b", re.I)
QC = re.compile(r"\b(pcq|fac|tk|asteur|pis|ouais|ouin|jpense|chu|p-e|toe|quin|dla|ste|"
                r"icitte|pantoute|ben|dac|fodrais|messemble|jai|jvais|jsuis|cest|"
                r"faut|ma|kessé|kossé|tanne|tanné|ecoeure|écoeuré|correc|correct)\b", re.I)


def walk(node, mapping, out):
    """ChatGPT stores each conversation as a message TREE keyed by id. Walk the mapping
    rather than assuming a linear list - branches (edits, regenerations) live here too."""
    for _id, rec in mapping.items():
        msg = (rec or {}).get("message") or {}
        author = ((msg.get("author") or {}).get("role")) or ""
        parts = ((msg.get("content") or {}).get("parts")) or []
        text = " ".join(p for p in parts if isinstance(p, str)).strip()
        if text:
            out.append({"role": author, "text": text, "create_time": msg.get("create_time")})


def main() -> int:
    path = os.path.join(ROOT, "conversations.json")
    data = json.load(open(path, encoding="utf-8"))

    turns = []
    for conv in data:
        walk(conv, conv.get("mapping") or {}, turns)

    roles = Counter(t["role"] for t in turns)
    user = [t for t in turns if t["role"] == "user"]

    fr_user, fr_asst = [], 0
    for t in turns:
        w = len(t["text"].split())
        if w < 6:
            continue
        rate = len(FR.findall(t["text"])) / w
        if rate > 0.10:
            if t["role"] == "user":
                fr_user.append(t)
            else:
                fr_asst += 1

    qc_hits = Counter()
    qc_turns = []
    for t in fr_user:
        hits = [h.lower() for h in QC.findall(t["text"])]
        if hits:
            qc_hits.update(hits)
            qc_turns.append((len(hits), t))
    qc_turns.sort(key=lambda x: -x[0])

    os.makedirs(OUTDIR, exist_ok=True)
    with open(CORPUS, "w", encoding="utf-8") as f:
        for t in fr_user:
            f.write(json.dumps({"role": "user", "text": t["text"],
                                "create_time": t["create_time"]}, ensure_ascii=False) + "\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# ChatGPT export - is there usable French in it?\n\n")
        f.write(f"- conversations: **{len(data):,}**\n")
        f.write(f"- messages with text: **{len(turns):,}**  ")
        f.write(", ".join(f"{r or '(none)'} {n:,}" for r, n in roles.most_common()) + "\n")
        f.write(f"- French USER turns (>10% French function words, 6+ words): "
                f"**{len(fr_user):,}**\n")
        f.write(f"- French assistant turns (not useful - a model's French): {fr_asst:,}\n\n")
        f.write("Only user turns are harvested. An assistant turn is the formal/translated "
                "register that has already failed us twice.\n\n")
        f.write(f"## Quebec markers in those user turns\n\n")
        f.write(f"{len(qc_turns):,} of the {len(fr_user):,} French user turns carry at least "
                f"one informal/Quebec marker.\n\n")
        for w, n in qc_hits.most_common(30):
            f.write(f"- `{w}` x{n}\n")
        f.write("\n## Densest examples (marker count, then the turn)\n\n")
        for n, t in qc_turns[:25]:
            s = t["text"].replace("\n", " ")[:220]
            f.write(f"- **{n}** - {s}\n")
        f.write(f"\nUser-turn French written to `{os.path.basename(CORPUS)}` "
                f"({len(fr_user):,} rows). NOT scrubbed yet - a ChatGPT export is personal "
                f"by default and the scrub has to run before anything leaves the machine.\n")

    print("wrote", REPORT)
    print("french user turns", len(fr_user), "with QC markers", len(qc_turns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
