import json
import os

P = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out\teams_corpus.jsonl"
OUT = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\data\out\teams_peek.txt"
R = [json.loads(l) for l in open(P, encoding="utf-8")]

lines = ["rows %d" % len(R), ""]

SAC = ("tabarnak", "calisse", "criss", "marde", "ostie", "batince", "tabarnouche")
shown = 0
for i in range(len(R) - 6):
    w = R[i:i + 6]
    blob = " ".join(r["text"] for r in w).lower()
    if any(r["burst"] >= 2 for r in w) and any(s in blob for s in SAC):
        lines.append("--- burst with a sacre ---")
        for r in w:
            g = r.get("gap_s")
            rx = ("   react=" + ",".join(r["reactions"])) if r.get("reactions") else ""
            lines.append("%-7s b%-2s +%-7ss  %s%s" % (r["speaker"], r["burst"], g, r["text"][:90], rx))
        lines.append("")
        shown += 1
        if shown >= 3:
            break

lines.append("--- messages carrying a reaction ---")
n = 0
for r in R:
    if r.get("reactions"):
        lines.append("%-7s react=%-22s %s" % (r["speaker"], ",".join(r["reactions"]), r["text"][:80]))
        n += 1
        if n >= 10:
            break

lines.append("")
lines.append("--- fastest bursts (gap <= 5s) ---")
n = 0
for r in R:
    if isinstance(r.get("gap_s"), (int, float)) and r["gap_s"] <= 5 and r["burst"] >= 1:
        lines.append("%-7s b%-2s +%-5ss  %s" % (r["speaker"], r["burst"], r["gap_s"], r["text"][:85]))
        n += 1
        if n >= 12:
            break

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT)
