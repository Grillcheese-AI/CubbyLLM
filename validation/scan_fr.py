"""Where, if anywhere, does French actually live in the training tree?

Two probes per file, on the first 400 KB:
  FR  generic French function words - is this French at all?
  QC  informal/Quebec markers that essentially never occur in written English or in
      formal French (pcq, fac, tk, asteur, pis, chu, icitte, pantoute) - is this the
      SPOKEN register, or is it textbook French?

A file can score high on FR and ~0 on QC: that is translated or formal French, which
is the material that has failed us twice already.
"""
import os
import re

B = r"I:\grillcheese_training_data"

QC = re.compile(r"\b(pcq|fac|tk|asteur|pis|ouais|ouin|jpense|jai|chu|p-e|cest|toe|quin|dla|ste|jvais|jsuis|icitte|pantoute|ben|dac|fodrais|messemble)\b", re.I)
FR = re.compile(r"\b(les|des|une|dans|pour|avec|mais|nous|vous|elle|cette|comme|tout|plus|bien|faire|etre|\u00eatre|sont|est|qui|que|aux|sur)\b", re.I)

rows = []
for dp, dn, fn in os.walk(B):
    for f in fn:
        p = os.path.join(dp, f)
        if os.path.splitext(f)[1].lower() not in (".jsonl", ".json", ".csv", ".txt"):
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz < 2000:
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as h:
                buf = h.read(400_000)
        except OSError:
            continue
        w = max(len(buf.split()), 1)
        fr = len(FR.findall(buf)) / w
        qc = len(QC.findall(buf)) / w
        if fr > 0.02:
            rows.append((fr, qc, sz, os.path.relpath(p, B)))

rows.sort(reverse=True)
print("%7s %7s %14s  %s" % ("FR", "QC", "bytes", "file"))
for fr, qc, sz, rp in rows[:30]:
    print("%7.4f %7.4f %14s  %s" % (fr, qc, format(sz, ","), rp))
print("total files above the French threshold:", len(rows))
