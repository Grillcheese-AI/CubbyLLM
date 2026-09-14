"""A model competition over a design question: one pack, N models, their answers kept side by side.

Wired: STANDALONE (validation script; never imported by cubbyllm/). The serving model never calls this --
a competition is a design probe, like the ceiling probe, and its output is a document Nick reads.

  python validation/exp_r21_competition.py --pack docs/research/<pack>.md --tag _neutral-scoring
  python validation/exp_r21_competition.py --pack ... --models "a,b,c" --max-tokens 3000

Each model gets the pack verbatim as the user turn and the same short system line. Answers, usage and
cost land in `validation/logs/exp_r21_competition<tag>.json` and a readable `.md` beside it. Responses are
cached by (model, body) like every other OpenRouter call, so a rerun costs nothing for the models that
already answered and only pays for the ones that failed.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SYSTEM = ("You are competing against other frontier models on a research-design question from an "
          "independent AI lab. Answer the pack on its own terms: it tells you the rules, the invariants "
          "and how you are scored. Be concrete and specific; propose mechanisms, not themes. Never "
          "invent a number, a result or a citation -- the pack says what that costs.")

DEFAULT_MODELS = ("openai/gpt-5.2", "x-ai/grok-4.6", "google/gemini-3.1-pro", "anthropic/claude-opus-4.5",
                  "deepseek/deepseek-v4-pro", "moonshotai/kimi-k3", "z-ai/glm-5.3", "qwen/qwen3.7-max")


def main() -> None:
    from openrouter import OpenRouterProposer
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="the markdown pack every model is given, verbatim")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--effort", default=None, help="reasoning effort for models that take one (low|medium|high)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pack = pathlib.Path(a.pack).read_text(encoding="utf-8")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"pack {a.pack} ({len(pack):,} chars) -> {len(models)} models, max_tokens {a.max_tokens}", flush=True)
    out: dict = {"pack": a.pack, "pack_chars": len(pack), "models": {}, "system": SYSTEM,
                 "max_tokens": a.max_tokens, "temperature": a.temperature}
    t0 = time.perf_counter()
    for m in models:
        t = time.perf_counter()
        try:
            p = OpenRouterProposer(m, system=SYSTEM, temperature=a.temperature, max_tokens=a.max_tokens,
                                   reasoning=({"effort": a.effort} if a.effort else None))
            text = p.chat(pack, max_tokens=a.max_tokens)
            u = dict(p.usage)
            out["models"][m] = {"ok": bool(text.strip()), "chars": len(text), "usage": u,
                                "wall_s": round(time.perf_counter() - t, 1), "text": text}
            print(f"  {m}: {len(text):,} chars, {u.get('completion_tokens', 0):,} completion tokens, "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
        except Exception as e:                                # noqa: BLE001 -- one model failing never stops the field
            out["models"][m] = {"ok": False, "error": str(e)[:300], "wall_s": round(time.perf_counter() - t, 1)}
            print(f"  {m}: FAILED {str(e)[:160]}", flush=True)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    ok = [m for m, r in out["models"].items() if r.get("ok")]
    (LOGS / f"exp_r21_competition{a.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Model competition — {pathlib.Path(a.pack).stem}", "",
          f"GrillCheese Research Lab · {time.strftime('%Y-%m-%d')} · {len(ok)} of {len(models)} models answered · "
          f"the pack is `{a.pack}`", ""]
    for m, r in out["models"].items():
        md += [f"## {m}", ""]
        md += [r["text"].strip() if r.get("ok") else f"*(no answer: {r.get('error', 'empty')})*", "", "---", ""]
    (LOGS / f"exp_r21_competition{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{len(ok)}/{len(models)} answered in {out['wall_s']:.0f}s -> exp_r21_competition{a.tag}.{{json,md}}")


if __name__ == "__main__":
    main()
