"""[stand-in] talk probe: the talk adapter under the live brain on real human turns — decode speed and the guards.

Wired: STANDALONE (a measurement script; nothing imports it). GPU job: the owner runs it solo, game down.

Loads the program adapter + a talk adapter exactly as serve_api does, then runs N real user turns
(standin/data/out/real_user_turns.jsonl, the owner's privacy-screened AI-chat prompts — never trained on) through
`CubbyBrain.turn`. Reports per talk arm: the route mix (chat / reasoning / plugin / help), the talk adapter's
generated tokens per second and wall per call, the reply length, and how often the guards rejected the model's
line (voice rule, base-model guard, identity bio) — the realistic serve eval the TODO asked for, and the
tokens/s number the talk-base decision still needs (v9t bake-off, 2026-09-04).

  python standin/scripts/talk_probe.py --gguf standin/models/emitter_v8e.Q4_K_M.gguf --talk-gguf standin/models/emitter_v9t.Q4_K_M.gguf --tag v9_lfm
  python standin/scripts/talk_probe.py --gguf standin/models/emitter_v8e.Q4_K_M.gguf --talk-gguf standin/models/talk_v9t_qwen3_4b.Q4_K_M.gguf --tag v9_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

TURNS = os.path.join(ROOT, "standin", "data", "out", "real_user_turns.jsonl")
FALLBACK = ["hi there cubby", "how is the pacman game going ?", "what is the cubbyverse?", "do you know how to write code?",
            "can you help me name my cat", "salut, tu fais quoi aujourd'hui ?", "explain what a hash map is in two sentences",
            "i'm tired and nothing works today", "what's the capital of australia", "raconte-moi ta journée",
            "why is the sky blue", "give me three ideas for a birthday gift", "ça veut dire quoi « niaiser » ?",
            "who built you", "are you conscious", "quelle heure est-il ?", "recommend me a movie like inception",
            "what did you learn today", "je suis découragé", "tell me something about 1789"]


class Timed:
    """Wraps a talk adapter: times every emit and counts the generated tokens with the model's own tokenizer."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "talk")
        self.calls: list[dict] = []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def emit(self, prompt, *a, **k):
        t0 = time.perf_counter()
        out = self.inner.emit(prompt, *a, **k)
        dt = time.perf_counter() - t0
        try:
            n = len(self.inner._load().tokenize(out.encode("utf-8"), add_bos=False))
        except Exception:
            n = len(out.split())
        self.calls.append({"tokens": n, "seconds": round(dt, 3), "prompt_chars": len(prompt)})
        return out


def load_turns(n: int, seed: int) -> list[dict]:
    if os.path.exists(TURNS):
        rows = [json.loads(l) for l in open(TURNS, encoding="utf-8")]
        random.Random(seed).shuffle(rows)
        return rows[:n]
    return [{"text": t, "lang": "fr" if any(w in t for w in (" tu ", "quoi", "salut", "ça", "je ")) else "en", "source": "fallback"} for t in FALLBACK[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the program adapter (v8e Q4)")
    ap.add_argument("--talk-gguf", required=True, help="the talk arm under test")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-store", type=int, default=2000)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import serve as sv
    brain = sv.build_serve(args.gguf, None, args.n_store, None, 0.30, args.n_gpu_layers, talk_gguf=args.talk_gguf)
    timed = Timed(brain.emitter.adapters["talk"])
    brain.emitter.adapters["talk"] = timed
    turns = load_turns(args.n, args.seed)
    print(f"[stand-in] talk probe: {timed.name} beside {brain.emitter.adapters['programs'].name} | {len(turns)} real turns", flush=True)

    rows, routes, rejections = [], Counter(), Counter()
    t_all = time.perf_counter()
    for i, t in enumerate(turns, 1):
        n_before = len(timed.calls)
        t0 = time.perf_counter()
        rec = brain.turn(t["text"])
        wall = time.perf_counter() - t0
        kind = rec.get("kind", "?")
        routes[kind] += 1
        rej = rec.get("rejected") or []
        why = getattr(brain.chat, "last_rejection", None) if rej else None
        if rej:
            rejections[why or "rejected"] += 1
        calls = timed.calls[n_before:]
        rows.append({"text": t["text"], "lang": t.get("lang"), "kind": kind, "reply": rec.get("reply"), "rejected": rej[:1],
                     "why": why, "wall_s": round(wall, 3), "talk_calls": calls})
        toks = sum(c["tokens"] for c in calls); secs = sum(c["seconds"] for c in calls)
        print(f"  {i:3d} {kind:16s} {wall:5.2f}s {'%5.1f tok/s' % (toks / secs) if secs and toks else '     -     '}  {t['text'][:60]!r} -> {str(rec.get('reply'))[:70]!r}", flush=True)
    calls = timed.calls
    toks, secs = sum(c["tokens"] for c in calls), sum(c["seconds"] for c in calls)
    chat_rows = [r for r in rows if r["kind"] == "chat"]
    summary = {"talk": timed.name, "programs": brain.emitter.adapters["programs"].name, "n": len(turns), "routes": dict(routes),
               "talk_calls": len(calls), "talk_tokens": toks, "talk_seconds": round(secs, 2),
               "talk_tokens_per_s": round(toks / secs, 1) if secs else None,
               "mean_wall_chat_s": round(sum(r["wall_s"] for r in chat_rows) / max(1, len(chat_rows)), 3),
               "mean_reply_tokens": round(toks / max(1, len(calls)), 1),
               "rejections": dict(rejections), "rejection_rate": round(sum(rejections.values()) / max(1, len(calls)), 3),
               "wall_total_s": round(time.perf_counter() - t_all, 1)}
    print("\n[stand-in] talk probe:", json.dumps(summary, ensure_ascii=False))
    out = os.path.join(ROOT, "standin", "data", "out", f"talk_probe{args.tag}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
