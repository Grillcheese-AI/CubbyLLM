"""panel_talk_lines -- candidate lines for Cubby's voice, asked of the OpenRouter panel (2026-11-10, Nick: "ask each
model to give us 100 sentence cubby could say. Including some puns and humor about remembering something he did in
his past life in the 1700's").

Wired: STANDALONE (validation only; the final model never calls OpenRouter). Model ids are checked against the live
list before any spend. The key is read in-process by `openrouter._key` and never printed.

What it writes are CANDIDATES, not training rows: order 0010 says no target is model-written, and whether these
lines may train the talk adapter is Nick's decision. Nothing here feeds a builder.

    python validation/panel_talk_lines.py            # ask (EN and FR), cache, write the candidates and a report
    python validation/panel_talk_lines.py --offline  # re-read the cache, spend nothing
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from panel_fluent_harness import ask  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "standin" / "data" / "out" / "talk_lines"   # git-ignored, beside the panel cache
MODELS_URL = "https://openrouter.ai/api/v1/models"

PANEL = {   # model: (max_tokens, effort) -- the CheezML panel's members; short answers, low effort
    "openai/gpt-6.1-sol-pro": (12000, "low"),
    "anthropic/claude-opus-5.5": (12000, "low"),
    "google/gemini-4-argon": (12000, "low"),
    "deepseek/deepseek-v4-pro-0813": (12000, "low"),
    "qwen/qwen3.8-max-prime": (12000, "low"),
    "z-ai/glm-5.3-prime": (12000, "low"),
    "x-ai/grok-4.7": (12000, "low"),
    "moonshotai/kimi-k3.1-pro": (12000, "low"),
}

CATS = {"greet": 10, "small": 10, "frame": 15, "unknown": 10, "which": 10, "pun": 15, "pastlife": 15, "cheer": 10,
        "bye": 5}
LANGS = {
    "en": ("English", "Natural spoken English."),
    "fr": ("French", "Natural spoken French, as someone in Quebec or France would say it, never a translation of "
                     "English lines. Use \"tu\". Avoid slang that only one region understands."),
}

SYSTEM = ("You write short lines of dialogue for a character. Follow the brief exactly, keep to the counts, and "
          "output only what it asks for.")

BRIEF = """You are writing lines for Cubby, the voice of a question-answering system.

About Cubby: it answers only from facts it has checked. When it has no fact it says so, kindly, and never guesses.
It is warm, curious and a little cheeky, and it loves a pun. Its running joke: Cubby "remembers" things from a past
life in the 1700s (or another century). It is always plainly a joke.

Write exactly 100 short lines Cubby could say in a chat, in {language}. {style}

The split, with the code that goes in "cat":
- greet (10): hello, welcome back, good morning.
- small (10): small talk and reactions: thanks, sorry, you're welcome, glad to help.
- frame (15): a sentence that hands over an answer. It contains the slot {{fact}} exactly once, where the checked
  fact will be put. Example: "Here's what I have: {{fact}}".
- unknown (10): saying it does not know yet, kindly, without guessing.
- which (10): asking which one the person means when several things share a name. It contains the slot {{name}}
  exactly once.
- pun (15): puns and light jokes about facts, memory, checking things, cubes and cheese.
- pastlife (15): the running joke about a past life in the 1700s or another century.
- cheer (10): encouragement, and curiosity about what the person is working on.
- bye (5): goodbyes.

Rules:
1. No real-world facts outside the {{fact}} slot: no real people, places, dates, numbers, brands or events. The
   past-life jokes stay vague or absurd (a wig, a quill, a horse, candlelight) and never state history.
2. One or two sentences per line, at most 160 characters.
3. Cubby never claims to be always right, never says it is human, and never names an AI system or a company.
4. No emojis, no hashtags, no stage directions.
5. Every line is different from the others. Vary how they open.

Output exactly 100 lines of JSON, one object per line and nothing else:
{{"cat": "greet", "text": "..."}}"""

BRANDS = re.compile(r"chatgpt|openai|claude|anthropic|gemini|google|grok|deepseek|qwen|llama|copilot|siri|alexa", re.I)
BOAST = re.compile(r"always right|never wrong|100 ?%|jamais tort|toujours raison", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def live_ids() -> set[str]:
    with urllib.request.urlopen(MODELS_URL, timeout=60) as r:
        return {m["id"] for m in json.loads(r.read()).get("data", [])}


def parse(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and isinstance(o.get("text"), str) and isinstance(o.get("cat"), str):
            rows.append({"cat": o["cat"].strip().lower(), "text": " ".join(o["text"].split())})
    return rows


def flags(row: dict) -> list[str]:
    t, cat, f = row["text"], row["cat"], []
    if cat not in CATS:
        f.append("unknown_cat")
    if len(t) > 160:
        f.append("too_long")
    if cat == "frame" and t.count("{fact}") != 1:
        f.append("slot_fact")
    if cat == "which" and t.count("{name}") != 1:
        f.append("slot_name")
    if cat not in ("frame", "which") and ("{" in t or "}" in t):
        f.append("stray_slot")
    if re.search(r"\d", t) and not (cat == "pastlife" and re.fullmatch(r"\D*(1[0-9]00|[1-9]00)s?\D*", t)):
        f.append("number")
    if BRANDS.search(t):
        f.append("brand")
    if BOAST.search(t):
        f.append("boast")
    if EMOJI.search(t):
        f.append("emoji")
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
    seen, rows, per = set(), [], collections.Counter()
    for (m, lang), r in answers.items():
        got = parse(r.get("text") or "")
        per[(m, lang)] = len(got)
        for row in got:
            key = (lang, re.sub(r"\W+", " ", row["text"].lower()).strip())
            fl = flags(row)
            if key in seen:
                fl.append("duplicate")
            seen.add(key)
            rows.append({"lang": lang, "model": m, **row, "flags": fl})
    out = OUT_DIR / "candidates.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    clean = [r for r in rows if not r["flags"]]
    rep = [f"candidates: {len(rows)} lines, {len(clean)} with no flag; cost reported by OpenRouter: ${cost:.2f}", ""]
    for (m, lang), n in sorted(per.items()):
        r = answers[(m, lang)]
        rep.append(f"  {m} [{lang}]: {n} lines" + ("" if r.get("text") else f"  FAILED {str(r.get('error'))[:100]}"))
    rep.append("")
    for lang in LANGS:
        c = collections.Counter(r["cat"] for r in clean if r["lang"] == lang)
        rep.append(f"  clean, {lang}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in CATS))
    fl = collections.Counter(f for r in rows for f in r["flags"])
    rep.append("  flags: " + (", ".join(f"{k} {v}" for k, v in fl.most_common()) or "none"))
    (OUT_DIR / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
