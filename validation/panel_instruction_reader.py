"""panel_instruction_reader -- the instruction-reader brief put to the OpenRouter panel (2026-11-13, Nick: "would it be
worth it to open a panel for our instruction problem?", then "yep launch the panel").

Wired: STANDALONE (validation only; the final model never calls OpenRouter). Model ids are checked against the live
list before any spend. The key is read in-process by `openrouter._key` and never printed.

    python validation/panel_instruction_reader.py            # ask, cache, write the panel file beside the brief
    python validation/panel_instruction_reader.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import panel_fluent_harness as harness  # noqa: E402
from panel_fluent_harness import ask  # noqa: E402

__wiring__ = "STANDALONE"

harness.TIMEOUT = 2400    # high effort and 64k budgets: a pro-mode answer can take well over 15 minutes

ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIEF = ROOT / "docs" / "research" / "2026-11-13-instruction-reader-brief.md"
OUT = ROOT / "docs" / "research" / "2026-11-13-instruction-reader-panel.md"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# The qFHRR panel's members and budgets (2026-11-12: each lab's top model on the live list, high effort, 64,000
# tokens each for thinking and answer). That run cost $2.65 as reported; its upper bound was $6.47 if every member
# spent its whole budget. Nick was told both before this run. Re-checked against the live list at run time.
PANEL = {   # model: (max_tokens, effort)
    "openai/gpt-6.1-sol-pro": (64000, "high"),
    "anthropic/claude-fable-5.2": (64000, "high"),
    "google/gemini-4-argon": (64000, "high"),
    "deepseek/deepseek-v4-pro-0813": (64000, "high"),
    "qwen/qwen3.8-max-prime": (64000, "high"),
    "z-ai/glm-5.3-prime": (64000, "high"),
    "x-ai/grok-4.7": (64000, "high"),
    "moonshotai/kimi-k3.1-pro": (64000, "high"),
}

SYSTEM = ("You are one member of an independent design panel. Answer the brief exactly, in the order of its "
          "deliverables. Use only the facts it states; where something is not stated, say you don't know. "
          "Do not invent measurements, citations or capabilities, and mark every number you estimate as an estimate. "
          "Plain, exact prose.")


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
        for m in [m for m in models if m not in live]:
            print(f"not on the live list, skipped: {m}")
            models.pop(m)
    with cf.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futs = {m: ex.submit(ask, m, a.offline, mt, eff, SYSTEM, question) for m, (mt, eff) in models.items()}
        answers = [futs[m].result() for m in models]
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers)
    ok = [r for r in answers if r.get("text", "").strip()]
    lines = [
        "# Panel: the instruction reader, and what it should write (2026-11-13)", "",
        "The brief: `2026-11-13-instruction-reader-brief.md` (beside this file). Asked independently, same prompt, "
        "temperature 0.3; budgets per model as in `validation/panel_instruction_reader.py`. Model ids checked "
        "against OpenRouter's live list at run time. Cached under `standin/data/out/panel_cache/`.", "",
        "**These are model opinions, not findings.** Nothing here is adopted until an order pre-registers it and its "
        "line is read. Cowork's own answer was sealed before this ran: "
        "`2026-11-13-instruction-reader-cowork-sealed.md`.", "",
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
