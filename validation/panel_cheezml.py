"""panel_cheezml -- the CheezML record-vocabulary brief put to the OpenRouter panel (2026-11-07, Nick: "give it to our
ai openrouter panel so they can generate one with everything we have").

Wired: STANDALONE (validation only; the final model never calls OpenRouter). Model ids are checked against the live
list before any spend. The key is read in-process by `openrouter._key` and never printed.

    python validation/panel_cheezml.py            # ask, cache, write docs/research/2026-11-07-cheezml-panel.md
    python validation/panel_cheezml.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from panel_fluent_harness import ask  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIEF = ROOT / "docs" / "research" / "2026-11-07-cheezml-brief.md"
OUT = ROOT / "docs" / "research" / "2026-11-07-cheezml-panel.md"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# the fact-store panel's members (2026-11-06), re-checked against the live list at run time
PANEL = {   # model: (max_tokens, effort) -- budgets from the fact-store panel's experience
    "openai/gpt-6.1-sol-pro": (24000, "medium"),
    "anthropic/claude-opus-5.5": (48000, "medium"),
    "google/gemini-4-argon": (24000, "medium"),
    "deepseek/deepseek-v4-pro-0813": (32000, "low"),
    "qwen/qwen3.8-max-prime": (24000, "medium"),
    "z-ai/glm-5.3-prime": (24000, "medium"),
    "x-ai/grok-4.7": (24000, "medium"),
    "moonshotai/kimi-k3.1-pro": (24000, "medium"),
}

SYSTEM = ("You are one member of an independent design panel. Answer the brief exactly, in the order of its "
          "deliverables. Use only the facts it states; where something is not stated, say you don't know. "
          "Do not invent opcodes, measurements, citations or capabilities. Plain, exact prose.")


def live_ids() -> set[str]:
    with urllib.request.urlopen(MODELS_URL, timeout=60) as r:
        return {m["id"] for m in json.loads(r.read()).get("data", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-render from cache, spend nothing")
    a = ap.parse_args()
    question = BRIEF.read_text(encoding="utf-8")
    models = dict(PANEL)
    if not a.offline:
        live = live_ids()
        missing = [m for m in models if m not in live]
        for m in missing:
            print(f"not on the live list, skipped: {m}")
            models.pop(m)
    with cf.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futs = {m: ex.submit(ask, m, a.offline, mt, eff, SYSTEM, question) for m, (mt, eff) in models.items()}
        answers = [futs[m].result() for m in models]
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers)
    ok = [r for r in answers if r.get("text", "").strip()]
    lines = [
        "# Panel: CheezML, one record vocabulary (2026-11-07)", "",
        "The brief: `2026-11-07-cheezml-brief.md` (beside this file). Asked independently, same prompt, temperature "
        "0.3; budgets per model as in `validation/panel_cheezml.py`. Model ids checked against OpenRouter's live "
        "list at run time. Cached under `standin/data/out/panel_cache/`.", "",
        "**These are model opinions, not findings.** Nothing here is adopted until the synthesis has expanded every "
        "JSON-LD, validated the records against the SHACL shapes and compiled every program.", "",
        f"Cost reported by OpenRouter: ${cost:.2f}. Answers: {len(ok)} of {len(answers)}.", ""]
    for r in answers:
        u = r.get("usage") or {}
        lines += [f"## {r['model']}", ""]
        if r.get("error") and not r.get("text"):
            lines += [f"*no answer: {r['error']}*", ""]
            continue
        lines += [f"*{u.get('completion_tokens', '?')} tokens out, {r.get('wall_s', '?')} s, finish "
                  f"`{r.get('finish')}`*", "", r["text"].strip(), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ok)}/{len(answers)} answers, ${cost:.2f}")
    for r in answers:
        print(f"  {r['model']}: {'ok' if r.get('text') else 'FAILED ' + str(r.get('error'))[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
