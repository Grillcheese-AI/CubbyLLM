"""panel_talk_shapes -- whole-reply templates for Cubby's voice, in the style of Nick's example replies (2026-11-10:
"example of what I want cubby to say"): a greeting that uses the person's name, what Cubby can do, a wrong
assumption corrected with the checked facts, several facts handed over as a list, a closing invitation.

Wired: STANDALONE (validation only; the final model never calls OpenRouter). Second round of
`panel_talk_lines.py`, same panel, same cache. The key is read in-process by `openrouter._key` and never printed.

Templates may be model-written; facts never are (Nick, 2026-11-10; order 0010's addendum). So every fact, every
name and every ability arrives through a slot the host fills from checked records. A template with a fact of its
own is flagged.

    python validation/panel_talk_shapes.py            # ask (EN and FR), cache, write the candidates and a report
    python validation/panel_talk_shapes.py --offline  # re-read the cache, spend nothing
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import sys

from panel_talk_lines import BOAST, BRANDS, EMOJI, LANGS, OUT_DIR, PANEL, ROOT, SYSTEM, live_ids, parse  # noqa: E402
from panel_fluent_harness import ask  # noqa: E402

# kind: (how many, the slots it must hold exactly once)
KINDS = {
    "hello_name": (10, ["{user}"]),
    "about": (6, ["{abilities}"]),
    "correct": (12, ["{subject}", "{facts}"]),
    "unconfirmed": (8, ["{subject}", "{facts}"]),
    "answer_list": (14, ["{facts}"]),
    "invite": (6, []),
    "remark": (4, []),
}
SLOTS = {"{user}", "{abilities}", "{subject}", "{facts}"}

BRIEF = """You are writing reply templates for Cubby, the voice of a question-answering system.

About Cubby: it answers only from facts it has checked. It is warm, clear and a little cheeky. It never guesses.
Every fact reaches the person through a slot that the system fills from its own checked records, so a template
never contains a fact of its own.

The tone wanted, from the owner's own examples (facts replaced by slots):
- "Hi {{user}}! It's great to meet you. I'm Cubby. I'm here to make things easier for you."
- "Here is what I can do for you: {{abilities}} What's on your mind today? I'm ready to help!"
- "Actually, that is not what my records show about {{subject}}. {{facts}}"

Write exactly 60 reply templates in {language}. {style}

The split, with the code that goes in "cat":
- hello_name (10): the person has just given their name. Greet them by name and introduce Cubby. Slot {{user}} once.
- about (6): the person asks what Cubby can do. Frame the list of abilities, then invite a question. Slot
  {{abilities}} once: the system puts the list there. Do not name any ability yourself.
- correct (12): the question assumes something the records contradict. Say so kindly and directly, then hand over
  the checked facts. Slots {{subject}} and {{facts}}, each once.
- unconfirmed (8): the question assumes something the records neither show nor contradict. Say Cubby cannot confirm
  it, then give what it does have. Slots {{subject}} and {{facts}}, each once.
- answer_list (14): several checked facts handed over together, with a short lead-in and, if you like, a short
  closing. Slot {{facts}} once: the system puts the list there.
- invite (6): a closing line that invites the next question. No slot.
- remark (4): a short warm reaction after an answer, with no fact in it. No slot.

Rules:
1. No real-world facts, names, places, dates or numbers anywhere. They arrive only through the slots.
2. Two to four sentences, at most 300 characters.
3. Cubby never claims to be always right, never says it is human, and never names an AI system or a company.
4. No emojis, no hashtags, no stage directions, no markdown.
5. Every template is different from the others. Vary how they open.

Output exactly 60 lines of JSON, one object per line and nothing else:
{{"cat": "hello_name", "text": "..."}}"""


def flags(row: dict) -> list[str]:
    t, cat, f = row["text"], row["cat"], []
    if cat not in KINDS:
        return ["unknown_cat"]
    if len(t) > 300:
        f.append("too_long")
    want = KINDS[cat][1]
    for s in SLOTS:
        n = t.count(s)
        if (s in want and n != 1) or (s not in want and n):
            f.append("slot " + s.strip("{}"))
    if re.search(r"\{(?!user\}|abilities\}|subject\}|facts\})", t):
        f.append("stray_slot")
    if re.search(r"\d", t):
        f.append("number")
    if BRANDS.search(t):
        f.append("brand")
    if BOAST.search(t):
        f.append("boast")
    if EMOJI.search(t) or "**" in t:
        f.append("format")
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-read the cache, spend nothing")
    a = ap.parse_args()
    models = dict(PANEL)
    if not a.offline:
        live = live_ids()
        for m in [m for m in models if m not in live]:
            print(f"not on the live list, skipped: {m}")
            models.pop(m)
    jobs = [(m, lang) for m in models for lang in LANGS]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {j: ex.submit(ask, j[0], a.offline, models[j[0]][0], models[j[0]][1], SYSTEM,
                             BRIEF.format(language=LANGS[j[1]][0], style=LANGS[j[1]][1])) for j in jobs}
        answers = {j: futs[j].result() for j in jobs}
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers.values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for (m, lang), r in answers.items():
        for row in parse(r.get("text") or ""):
            key = (lang, re.sub(r"\W+", " ", row["text"].lower()).strip())
            fl = flags(row)
            if key in seen:
                fl.append("duplicate")
            seen.add(key)
            rows.append({"lang": lang, "model": m, **row, "flags": fl})
    out = OUT_DIR / "shapes.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    clean = [r for r in rows if not r["flags"]]
    rep = [f"reply templates: {len(rows)}, {len(clean)} with no flag; cost reported by OpenRouter: ${cost:.2f}"]
    failed = [f"{m} [{lang}]" for (m, lang), r in answers.items() if not r.get("text")]
    rep.append("  failed: " + (", ".join(failed) or "none"))
    for lang in LANGS:
        c = collections.Counter(r["cat"] for r in clean if r["lang"] == lang)
        rep.append(f"  clean, {lang}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in KINDS))
    fl = collections.Counter(f for r in rows for f in r["flags"])
    rep.append("  flags: " + (", ".join(f"{k} {v}" for k, v in fl.most_common()) or "none"))
    (OUT_DIR / "shapes_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
