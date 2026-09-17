"""label_affect - Plutchik petal + intensity over the scrubbed chat corpus.

Wired: STANDALONE (data-building only; the final model never calls OpenRouter).

THIS IS THE FIRST STEP IN THIS PIPELINE THAT TOUCHES THE NETWORK. It reads
`teams_corpus.jsonl`, which is the SCRUBBED output of `teams_harvest.py` - names
replaced, money/urls/emails/phones replaced, machine pastes dropped, and every
credential-bearing message quarantined before the scrub even ran. It refuses to run on
a raw export, because "I'll scrub it after" is how a password reaches a third party.

WHY LABEL RATHER THAN GENERATE. Four frontier models asked to WRITE Quebec affect put
sacres in ~1 line in 3. The real corpus runs 0.5% - a 66x over-production - and the
actual discharge channel, `lol`, appeared in none of their output. A model asked to
perform a register caricatures it. A model asked to READ one cannot, because it is not
choosing the words.

TARGET SCHEMA is `emotions.jsonl`'s, not GoEmotions': `{primary, intensity, secondary}`
over Plutchik's eight petals, with intensity CONTINUOUS. That choice dissolves the
24-points-for-28-labels problem rather than solving it - the collisions came from
quantising a foreign label set into petal x tier, and there is nothing to quantise here.

WINDOWS, NOT SINGLE MESSAGES. Median message length in this corpus is 7 words and 61%
arrive within 30s of the previous one; "ok" means nothing alone and a great deal after
a five-message burst about a crash. So the model sees a run of consecutive messages and
labels each in context. It also costs ~20x less than one call per message.

THE TIMING IS DELIBERATELY WITHHELD. `gap_s` and `burst` are NOT in the prompt, even
though they are in the corpus and they are affect signal. If the labels come back and
high-arousal petals independently correlate with short gaps and long bursts, that is
convergent evidence from a channel the labeller never saw. Feed the clock in and the
check becomes circular - the one measurement that could falsify the labels is gone.

    python validation/label_affect.py --limit 40       # pilot: 40 windows, then judge
    python validation/label_affect.py                  # everything
    python validation/label_affect.py --offline        # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from openrouter import API, _key  # noqa: E402

CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
CACHE = ROOT / "standin" / "data" / "out" / "label_cache"
OUT = ROOT / "standin" / "data" / "out" / "teams_labelled.jsonl"
REPORT = ROOT / "docs" / "teams_affect_labels.md"

WINDOW = 20

# Cheap and good enough to READ a register. The expensive models were needed to test
# whether anything could WRITE one; they could not, and this is the easier job.
LABELLER = "google/gemini-3.8-flash"
CHECKER = "deepseek/deepseek-v3.2"          # blind second opinion on a sample

PETALS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")

# Lövheim's PUBLISHED corners, (5-HT, DA, NE), verified against the source table.
#
# NOTE, AND IT IS AN OPEN DECISION: `standin/neurochem.py::_CORNERS` currently does NOT
# match this - only 2 of its 8 corners sit where Lövheim puts them, which is why
# anticipation had no corner and why the fear and disgust corners were mislabelled as
# the two "social" ones and silenced. The remap is proposed and unapproved, so this
# table lives here rather than being imported: the labeller needs a correct adjacency
# graph today, and nothing here changes the running agent.
LOVHEIM = {
    "shame":        (0, 0, 0),
    "sadness":      (0, 0, 1),     # distress / anguish
    "fear":         (0, 1, 0),     # fear / terror
    "anger":        (0, 1, 1),     # anger / rage
    "disgust":      (1, 0, 0),     # contempt / disgust
    "surprise":     (1, 0, 1),
    "joy":          (1, 1, 0),     # enjoyment / joy
    "anticipation": (1, 1, 1),     # interest / excitement
}
# Trust has no Lövheim corner (it is oxytocin, not a monoamine axis). It stays a valid
# LABEL - the corpus is two colleagues and trust has a referent here - but it cannot
# take part in the geometric dyad check, so a dyad naming it is reported, not rejected.
NO_CORNER = {"trust"}


def adjacent(a: str, b: str) -> bool:
    """Cube-adjacent = differs in exactly one axis. A dyad's midpoint is only a
    representable POINT when its two corners are adjacent; between corners that differ
    in two or three axes the midpoint sits equidistant from four or more corners and
    the classifier reads it as something else entirely. Two of the panel models raised
    this independently, and it is a fact about the geometry, not an opinion."""
    if a in NO_CORNER or b in NO_CORNER or a not in LOVHEIM or b not in LOVHEIM:
        return True                      # not checkable, so not failed
    return sum(x != y for x, y in zip(LOVHEIM[a], LOVHEIM[b])) == 1


SYSTEM = (
    "Tu lis une conversation de travail entre collègues québécois et tu annotes l'état "
    "affectif de chaque message. Tu ne juges pas, tu ne corriges pas la langue, tu ne "
    "traduis pas. Tu réponds uniquement par les lignes JSON demandées."
)

PROMPT = """\
Voici {n} messages consécutifs d'une conversation de travail. `[SELF]`, `[PEER]` et
`[PEER2]` sont les interlocuteurs; les crochets comme [MONEY] ou [URL] remplacent des
informations retirées.

Pour CHAQUE message, donne une ligne JSON:
{{"i": <numéro>, "primary": "<pétale>", "intensity": <0.0 à 1.0>, "secondary": "<pétale ou null>"}}

Pétales possibles (les huit de Plutchik, en anglais):
joy, trust, fear, surprise, sadness, disgust, anger, anticipation
Utilise "neutral" comme primary quand le message ne porte aucune charge affective.

Règles:
- L'intensité est CONTINUE, pas trois paliers. 0.2 = à peine perceptible, 0.5 = nette,
  0.9 = très forte. Un message neutre a une intensité proche de 0.
- `secondary` seulement si un DEUXIÈME pétale est réellement présent, sinon null.
- Juge d'après le contexte de la conversation, pas seulement d'après le message isolé:
  « ok » après cinq messages sur une panne n'est pas « ok » au début d'une journée.
- Le registre est familier, sans accents, avec des anglicismes. C'est normal. Ne le
  traite pas comme de la négligence ou de la colère.
- Un juron n'est pas automatiquement de la colère, et son absence n'est pas du calme.
- « lol », « haha » servent souvent à désamorcer: le message peut porter de la
  frustration ou de la fierté malgré le rire.

Une ligne JSON par message, dans l'ordre, rien d'autre.

Messages:
{body}
"""


def windows(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def ask(model: str, win: list[dict], offline: bool, max_tokens: int, effort: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'{i}. {r["speaker"]}: {r["text"]}' for i, r in enumerate(win))
    user = PROMPT.format(n=len(win), body=body)
    tag = hashlib.sha256(f"{model}|{SYSTEM}|{user}|{max_tokens}|{effort}".encode()).hexdigest()[:20]
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        return {"model": model, "text": "", "error": "not cached and --offline"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.2,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Grillcheese-AI/CubbyLLM", "X-Title": "CubbyLLM affect labels"})
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


def parse_labels(rec: dict, n: int) -> dict[int, dict]:
    out = {}
    for line in (rec.get("text") or "").splitlines():
        line = line.strip().strip(",").lstrip("﻿")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        i = o.get("i")
        if isinstance(i, int) and 0 <= i < n:
            out[i] = o
    return out


def check(o: dict) -> list[str]:
    """Model proposes, geometry disposes. Same discipline as the hypothesis verifier:
    the label is a claim, and a claim that cannot survive the cube is rejected."""
    bad = []
    p, s = o.get("primary"), o.get("secondary")
    if p not in PETALS and p != "neutral":
        bad.append(f"bad-primary:{p}")
    v = o.get("intensity")
    if not isinstance(v, (int, float)) or not 0 <= v <= 1:
        bad.append("bad-intensity")
    if s not in (None, "null", "") and s not in PETALS:
        bad.append(f"bad-secondary:{s}")
    if s in PETALS and p in PETALS:
        if s == p:
            bad.append("secondary==primary")
        elif not adjacent(p, s):
            bad.append(f"non-adjacent-dyad:{p}+{s}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="label only the first N windows (pilot)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--check-every", type=int, default=8,
                    help="send every Nth window to a SECOND model, blind, for agreement")
    a = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"no scrubbed corpus at {CORPUS} - run teams_harvest.py first")
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    wins = windows(rows, WINDOW)
    if a.limit:
        wins = wins[:a.limit]

    labelled, bad, agree, disagree, dist = [], Counter(), 0, 0, Counter()
    intens_by_gap = {"fast": [], "slow": []}
    spend = 0
    for k, win in enumerate(wins):
        rec = ask(LABELLER, win, a.offline, a.max_tokens, a.effort)
        spend += (rec.get("usage") or {}).get("total_tokens", 0)
        got = parse_labels(rec, len(win))
        second = {}
        if a.check_every and k % a.check_every == 0:
            rec2 = ask(CHECKER, win, a.offline, a.max_tokens, a.effort)
            spend += (rec2.get("usage") or {}).get("total_tokens", 0)
            second = parse_labels(rec2, len(win))
        for i, r in enumerate(win):
            o = got.get(i)
            if not o:
                bad["missing"] += 1
                continue
            flaws = check(o)
            for fl in flaws:
                bad[fl] += 1
            if flaws:
                continue
            row = dict(r)
            row["primary"] = o["primary"]
            row["intensity"] = float(o["intensity"])
            row["secondary"] = o.get("secondary") if o.get("secondary") in PETALS else None
            dist[row["primary"]] += 1
            # The held-out check: the labeller never saw gap_s.
            g = r.get("gap_s")
            if isinstance(g, (int, float)) and row["primary"] != "neutral":
                intens_by_gap["fast" if g <= 30 else "slow"].append(row["intensity"])
            if i in second and second[i].get("primary"):
                if second[i]["primary"] == row["primary"]:
                    agree += 1
                else:
                    disagree += 1
                    row["second_opinion"] = second[i]["primary"]
            labelled.append(row)
        print(f"  window {k + 1}/{len(wins)}  labelled {len(labelled)}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in labelled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Affect labels over the real Quebec work corpus\n\n")
        f.write(f"Labeller `{LABELLER}`, blind second opinion `{CHECKER}` on every "
                f"{a.check_every}th window. {len(wins)} windows of {WINDOW}, "
                f"**{len(labelled)} messages labelled**, {spend:,} tokens.\n\n")
        f.write("Labels are a model's reading, not ground truth. What makes them worth "
                "anything is that two independent checks exist: a geometric one that "
                "rejects dyads the cube cannot represent, and a timing one the labeller "
                "never saw.\n\n")
        f.write("## Petal distribution\n\n| petal | n | share |\n|---|---:|---:|\n")
        tot = sum(dist.values()) or 1
        for p, n in dist.most_common():
            f.write(f"| {p} | {n} | {n / tot:.1%} |\n")
        f.write("\n## Rejected by the verifier\n\n")
        f.write("\n".join(f"- {k} x{v}" for k, v in bad.most_common()) or "- none\n")
        f.write(f"\n\n## Second-model agreement\n\n")
        tot2 = agree + disagree
        f.write(f"On the {tot2} messages both models labelled, they chose the same petal "
                f"**{agree}** times (**{agree / max(tot2, 1):.0%}**).\n\n")
        f.write("## The held-out check: timing\n\n")
        f.write("`gap_s` was never in the prompt. If the labels are reading real affect "
                "rather than surface features, non-neutral intensity should run higher on "
                "messages fired off within 30s of the last one than on messages sent after "
                "a long pause.\n\n")
        f.write(f"- within 30s: mean intensity **{mean(intens_by_gap['fast']):.3f}** "
                f"(n={len(intens_by_gap['fast'])})\n")
        f.write(f"- after 30s:  mean intensity **{mean(intens_by_gap['slow']):.3f}** "
                f"(n={len(intens_by_gap['slow'])})\n\n")
        f.write("A gap of roughly nothing means the labels are not tracking arousal, and "
                "the intensity axis is decorative. That would be a real negative result and "
                "it is worth more than a plausible-looking table.\n")

    print(f"\nwrote {OUT}\nwrote {REPORT}\n{len(labelled)} labelled, {spend:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
