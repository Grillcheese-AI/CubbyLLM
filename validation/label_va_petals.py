"""Label the valence/arousal texts with a Plutchik petal, blind to their labels.

Wired: STANDALONE (data-building only; the final model never calls OpenRouter).

The external check on the cube had no data: `emotions.jsonl` carries petals,
`amygdala_affect.jsonl` carries valence and arousal, and the 1,726 texts they share are
exactly the placeholder rows - valence 0.0, arousal 0.5, a default rather than a
reading. The 1,259 rows with real affect values carry no petal at all.

So: give them one. Their valence and arousal already exist, were produced by something
that never saw a petal, and are not changed here. If a petal assigned blind then
predicts them, that is evidence from outside the cube's own geometry - which is the one
kind of evidence this project has never had for it.

THE ANTI-CIRCULARITY RULES, which are the whole point and are easy to lose:

  - the labeller never sees valence or arousal for these texts;
  - the words "valence" and "arousal" do not appear in the prompt, nor does any
    request for intensity, strength or degree - naming the dimension invites the model
    to reason about it and the test stops being independent;
  - monoamines, the cube and this project are not mentioned;
  - a second model labels a sample blind, so petal disagreement is measurable rather
    than assumed away.

Ask for one thing only. Intensity was already shown to be decorative when a model
supplies it - anchored at 0.35 in every group of the Teams pilot - and asking for it
here would add a channel that leaks the dimension under test.

    python validation/label_va_petals.py --limit 10   # pilot: 10 windows, then judge
    python validation/label_va_petals.py              # all of them
    python validation/label_va_petals.py --offline    # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from openrouter import API, _key  # noqa: E402

SRC = pathlib.Path(r"I:\grillcheese_training_data\pre\amygdala_affect.jsonl")
CACHE = ROOT / "standin" / "data" / "out" / "va_petal_cache"
OUT = ROOT / "standin" / "data" / "out" / "va_petals.jsonl"

PLACEHOLDER = (0.0, 0.5)
WINDOW = 25
LABELLER = "google/gemini-3.8-flash"
CHECKER = "deepseek/deepseek-v3.2"

PETALS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")

SYSTEM = ("You sort short sentences by which basic emotion they express. You answer only "
          "with the JSON lines asked for, nothing else.")

# Deliberately plain. No mention of strength, degree, intensity, activation, pleasantness
# or any other word for the dimensions this labelling is going to be tested against.
PROMPT = """\
Below are {n} short sentences. For each one, say which of Plutchik's eight basic
emotions it expresses.

The eight: joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
Use "neutral" if the sentence expresses none of them.

One JSON line per sentence, in order, nothing else:
{{"i": <number>, "petal": "<one of the eight, or neutral>"}}

Pick the single closest one. Do not explain. Do not add any other field.

Sentences:
{body}
"""


def ask(model: str, win, offline: bool, max_tokens: int, effort: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{i}. {t}" for i, t in enumerate(win))
    user = PROMPT.format(n=len(win), body=body)
    tag = hashlib.sha256(f"{model}|{SYSTEM}|{user}|{max_tokens}|{effort}".encode()).hexdigest()[:20]
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        return {"model": model, "text": "", "error": "not cached and --offline"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Grillcheese-AI/CubbyLLM",
        "X-Title": "CubbyLLM va petals"})
    t0 = time.time()
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except Exception as e:
        return {"model": model, "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    ch = (raw.get("choices") or [{}])[0]
    rec = {"model": model, "text": (ch.get("message") or {}).get("content") or "",
           "finish": ch.get("finish_reason"), "usage": raw.get("usage") or {},
           "wall_s": round(time.time() - t0, 1), "error": None}
    if rec["text"].strip():
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def parse(rec, n):
    out = {}
    for line in (rec.get("text") or "").splitlines():
        line = line.strip().strip(",").lstrip("\ufeff")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        i, p = o.get("i"), str(o.get("petal", "")).strip().lower()
        if isinstance(i, int) and 0 <= i < n and (p in PETALS or p == "neutral"):
            out[i] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--check-every", type=int, default=6)
    a = ap.parse_args()

    seen, items = set(), []
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        t = (r.get("text") or "").strip()
        v, ar = r.get("valence"), r.get("arousal")
        if not t or not isinstance(v, (int, float)):
            continue
        if (float(v), float(ar or 0)) == PLACEHOLDER:
            continue                                 # the default, not a reading
        k = " ".join(t.lower().split())
        if k in seen:
            continue
        seen.add(k)
        items.append({"text": t, "valence": float(v), "arousal": float(ar or 0)})

    wins = [items[i:i + WINDOW] for i in range(0, len(items), WINDOW)]
    if a.limit:
        wins = wins[:a.limit]

    out, agree, disagree, dist, spend = [], 0, 0, Counter(), 0
    for k, win in enumerate(wins):
        texts = [w["text"] for w in win]
        rec = ask(LABELLER, texts, a.offline, a.max_tokens, a.effort)
        spend += (rec.get("usage") or {}).get("total_tokens", 0)
        got = parse(rec, len(win))
        second = {}
        if a.check_every and k % a.check_every == 0:
            rec2 = ask(CHECKER, texts, a.offline, a.max_tokens, a.effort)
            spend += (rec2.get("usage") or {}).get("total_tokens", 0)
            second = parse(rec2, len(win))
        for i, w in enumerate(win):
            p = got.get(i)
            if not p:
                continue
            dist[p] += 1
            row = dict(w)
            row["petal"] = p
            if i in second:
                if second[i] == p:
                    agree += 1
                else:
                    disagree += 1
                    row["second_opinion"] = second[i]
            out.append(row)
        print(f"  window {k + 1}/{len(wins)}  labelled {len(out)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot2 = agree + disagree
    print(f"\nwrote {OUT}")
    print(f"  {len(out)} labelled of {len(items)} candidates, {spend:,} tokens")
    print(f"  two-model agreement {agree}/{tot2} ({agree / max(tot2, 1):.0%})")
    for p, n in dist.most_common():
        print(f"  {p:<13} {n:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
