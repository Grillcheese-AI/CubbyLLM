"""eval_emitter_vm — the VERIFIED read on a stand-in emitter.

Wired: STANDALONE (stand-in tooling).

Text exact-match (the Colab notebook's number) is a format read. This is the
capability read: every generated program goes through the real Rust cubelang
VM (`cubelang.exe`, strict compile, run-proto) and is scored the way the
data was verified:

  arithmetic / kernel / chain : executes AND the result matches the record's gold
  role_binding                : executes (returns a bound symbol) — no gold exists

Three ways to get generations:
  --val-generations FILE   replay the notebook's val_generations.json (exact, no model needed)
  --gguf PATH              generate live in-process via llama-cpp-python (Vulkan on the AMD card)
  --server URL             generate live through an OpenAI-compatible server (temperature 0)

Reads the val split of standin/data/out/emitter_sft.jsonl for prompts/gold
(or the generations file's own reference/gold when replaying). Writes
standin/data/out/eval_emitter_vm{TAG}.json. Every number it prints is
[stand-in].

  python standin/eval_emitter_vm.py --val-generations "D:/My Drive/cubbyllm/standin/emitter_lfm25_2p6b/val_generations.json"
  python standin/eval_emitter_vm.py --gguf "D:/My Drive/cubbyllm/standin/emitter_lfm25_2p6b/gguf/emitter-q8_0.gguf" --limit 200
  python standin/eval_emitter_vm.py --server http://127.0.0.1:8080 --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_emitter_sft import answer_fn, gold_matches, shim_isolver  # noqa: E402
from identity import identity_ok, load_facts  # noqa: E402
from standin.emitter import LlamaCppEmitter, LlamaServerEmitter, ReplayEmitter  # noqa: E402

DATA = os.environ.get("STANDIN_SFT", os.path.join(ROOT, "standin", "data", "out", "emitter_sft.jsonl"))
OUT_DIR = os.path.join(ROOT, "standin", "data", "out")


_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)


def strip_think(s: str) -> str:
    """LFM2.5's chat template opens a <think> block in the generation prompt
    and the model fills it before the answer (seen on the first SFT run,
    2026-08-30). Everything up to and including </think> is reasoning, not
    the answer: drop it before scoring programs OR identity turns (the
    reasoning text talks about AGI/other models and would trip the check)."""
    return _THINK_RE.sub("", s, count=1) if "</think>" in s else s


def strip_fences(s: str) -> str:
    """Models sometimes wrap programs in ``` fences; the VM does not want them."""
    s = strip_think(s).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip() + "\n"


def run_vm(source: str):
    from cubbyllm.bridges import cubelang_client as cc
    try:
        out = cc.run_program_proto(source, fn=answer_fn(source))   # chains answer from their last hop
        return bool(out.get("ok")), out.get("result"), None
    except Exception as e:
        return False, None, str(e).splitlines()[0][:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-generations", help="the notebook's val_generations.json (replay mode)")
    ap.add_argument("--gguf", help="GGUF file for in-process llama-cpp-python inference (live mode, Vulkan)")
    ap.add_argument("--server", help="OpenAI-compatible server base URL (live mode)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=DATA, help="the SFT jsonl whose val split to score (v4: emitter_sft_v4.jsonl)")
    ap.add_argument("--no-shim", action="store_true", help="do not apply the ISolver parse/verify shim to generations")
    args = ap.parse_args()
    if not (args.val_generations or args.server or args.gguf):
        raise SystemExit("give --val-generations FILE, --gguf PATH or --server URL")

    if args.val_generations:
        gens = json.load(open(args.val_generations, encoding="utf-8"))
        items = gens["outputs"]
        emitter = ReplayEmitter(items, name=f"replay:{gens.get('model', '?')}")
        records = [{"id": g["id"], "task": g["task"], "subtype": g.get("subtype", ""), "prompt": g["prompt"],
                    "program": g["reference"], "gold": g.get("gold"), "system": g.get("system"),
                    "lang": g.get("lang", "en")}
                   for g in items]
    else:
        emitter = LlamaCppEmitter(args.gguf, n_gpu_layers=args.n_gpu_layers) if args.gguf else LlamaServerEmitter(args.server)
        records = [json.loads(l) for l in open(args.data, encoding="utf-8")]
        records = [r for r in records if r["split"] == "val"]
        random.Random(1).shuffle(records)
    if args.limit:
        records = records[:args.limit]
    print(f"[stand-in] emitter {emitter.name} | {len(records)} val records | shim {'off' if args.no_shim else 'on'}")

    t0 = time.perf_counter()
    stats = defaultdict(Counter)
    rows = []
    facts = load_facts()
    for i, r in enumerate(records, 1):
        task = r["task"]
        if task in ("identity", "chat", "content", "emotion"):   # conversational turns: checks, not the VM
            gen = strip_think(emitter.emit(r["prompt"], system=r.get("system"))).strip()
            if task == "identity":
                ok = identity_ok(r.get("subtype", ""), gen, facts, r.get("lang", "en"))
            elif task == "chat":                     # v5: Cubby's rules hold and it is NOT a bio
                from identity import is_identity_reply, is_model_guard, voice_ok
                ok = bool(gen) and voice_ok(gen, facts) and not is_model_guard(gen) and not is_identity_reply(gen, facts)
            elif task == "emotion":                  # any of the rater's labels, first
                first = gen.lower().replace("—", ",").split(",")[0].strip(" -:.")
                ok = first in {str(g).lower() for g in (r.get("gold_any") or [r.get("gold")])}
            else:                                    # content awareness: the label comes first
                ok = gen.lower().split(" ")[0].strip(" —-:.,") == str(r.get("gold")).lower()
            stats[task]["n"] += 1
            stats[task]["identity_ok"] += int(ok)
            stats[task][f"identity_ok:{r.get('lang', 'en')}"] += int(ok)
            stats[task][f"n:{r.get('lang', 'en')}"] += 1
            rows.append({"id": r["id"], "task": task, "intent": r.get("subtype"), "lang": r.get("lang", "en"),
                         "identity_ok": ok, "generated": gen})
            continue
        gen = strip_fences(emitter.emit(r["prompt"]))
        src = gen if args.no_shim else shim_isolver(gen)
        ok, res, err = run_vm(src)
        gm = gold_matches(res, r.get("gold")) if ok else None
        sub = str(r.get("subtype") or "")
        grp = f"{task}/{sub.split(':n_hop')[0]}" if sub.startswith("game:") else task   # v4: the game's families apart
        stats[grp]["n"] += 1
        stats[grp]["executes"] += int(ok)
        if r.get("gold") is not None:
            stats[grp]["with_gold"] += 1
            stats[grp]["gold_match"] += int(bool(gm))
        stats[grp]["text_exact"] += int(" ".join(gen.split()) == " ".join(r["program"].split()))
        rows.append({"id": r["id"], "task": task, "executes": ok, "vm_result": None if res is None else str(res)[:120],
                     "gold": r.get("gold"), "gold_match": gm, "vm_error": err, "generated": gen})
        if i % 50 == 0:
            print(f"  {i}/{len(records)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    print("\n[stand-in] VM-verified eval by task:")
    summary = {}
    for task, c in sorted(stats.items()):
        if task in ("identity", "chat", "content", "emotion"):
            per_lang = {l: (c[f"identity_ok:{l}"] / c[f"n:{l}"]) for l in ("en", "fr") if c[f"n:{l}"]}
            summary[task] = {"n": c["n"], "ok": c["identity_ok"] / c["n"], "by_lang": per_lang}
            note = {"identity": "name/builder present, no AGI/other-model/feelings claims, don't-know line verbatim",
                    "chat": "voice rules hold, no base-model guard, not an identity bio",
                    "content": "the nsfw/safe label comes first",
                    "emotion": "the first emotion named is one the raters gave"}[task]
            print(f"  {task:13s} n={c['n']:4d} ok={c['identity_ok'] / c['n']:.3f} by lang {per_lang}  ({note})")
            continue
        ex = c["executes"] / c["n"]
        gm = (c["gold_match"] / c["with_gold"]) if c["with_gold"] else None
        te = c["text_exact"] / c["n"]
        summary[task] = {"n": c["n"], "executes": ex, "gold_match": gm, "text_exact": te}
        print(f"  {task:13s} n={c['n']:4d} executes={ex:.3f} gold_match={'—' if gm is None else f'{gm:.3f}'} text_exact={te:.3f}")
    errs = Counter((row.get("vm_error") or "")[:70] for row in rows if row.get("executes") is False)
    if errs:
        print("  top VM errors:")
        for e, k in errs.most_common(5):
            print(f"    {k:4d}  {e}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"eval_emitter_vm{args.tag}.json")
    json.dump({"emitter": emitter.name, "n": len(records), "shim": not args.no_shim, "summary": summary,
               "rows": rows, "wall_s": time.perf_counter() - t0}, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
