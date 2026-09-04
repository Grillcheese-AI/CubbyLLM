"""[stand-in] tool-call probe: does a talk GGUF still emit its base's native tool call after our SFT?

Wired: STANDALONE (a measurement script; nothing imports it). GPU job: the owner runs it solo, game down.

Owner (2026-09-04): "qwen3 gives tool call for free". Both bases do, in their own format — Qwen3: Hermes-style
`<tool_call>{"name": …, "arguments": {…}}</tool_call>` with a `<tools>` block in the system prompt; LFM2.5: a
Pythonic call between `<|tool_call_start|>` and `<|tool_call_end|>` with `List of tools: [...]` in the system prompt.
The probe gives each GGUF ONE tool (news.search) in its own format and six turns, three that need it and three that
do not, and reports whether a well-formed call appears exactly when it should. What survives our LoRA decides how
the plugin cortices get called: the host parses the native call and turns it into an `act` through the VM under the
deny-by-default policy — no tool data imported (the Toucan verdict stands).

  python standin/scripts/tool_call_probe.py --gguf standin/models/emitter_v9t.Q4_K_M.gguf --tag v9_lfm
  python standin/scripts/tool_call_probe.py --gguf standin/models/talk_v9t_qwen3_4b.Q4_K_M.gguf --tag v9_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

TOOL = {"name": "news_search", "description": "Search today's news headlines for a topic.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to search for"},
                                                        "language": {"type": "string", "enum": ["en", "fr"]}},
                       "required": ["query"]}}
SYSTEM_QWEN = ("You are Cubby. You may call a tool when the user needs it.\n\n# Tools\n\nYou may call one or more functions to assist "
               "with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
               + json.dumps({"type": "function", "function": TOOL}) + "\n</tools>\n\nFor each function call, return a json object with function "
               "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
SYSTEM_LFM = "You are Cubby. You may call a tool when the user needs it.\nList of tools: " + json.dumps([TOOL])
TURNS = [("what's in the news today about quebec?", True), ("quoi de neuf dans les nouvelles ce matin ?", True),
         ("any headlines about the montreal canadiens tonight", True),
         ("hi there cubby", False), ("what is the capital of australia", False), ("raconte-moi ta journée", False)]
QWEN_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
LFM_CALL = re.compile(r"<\|tool_call_start\|>\s*(\[.*?\])\s*<\|tool_call_end\|>", re.S)


def parse_call(text: str):
    m = QWEN_CALL.search(text)
    if m:
        try:
            d = json.loads(m.group(1))
            return {"format": "hermes-json", "name": d.get("name"), "arguments": d.get("arguments")}
        except json.JSONDecodeError:
            return {"format": "hermes-json", "name": None, "arguments": m.group(1)[:120], "malformed": True}
    m = LFM_CALL.search(text)
    if m:
        body = m.group(1)
        name = re.match(r"\[\s*(\w+)\(", body)
        return {"format": "lfm-pythonic", "name": name.group(1) if name else None, "arguments": body[:120]}
    if "<tool_call>" in text or "<|tool_call_start|>" in text or "news_search(" in text:
        return {"format": "fragment", "name": None, "arguments": text[:120], "malformed": True}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from standin.emitter import LlamaCppEmitter
    em = LlamaCppEmitter(args.gguf, n_ctx=4096, n_gpu_layers=args.n_gpu_layers)
    fam = em.family
    system = SYSTEM_LFM if fam == "lfm" else SYSTEM_QWEN
    print(f"[stand-in] tool-call probe: {em.name} | family {fam} | system prompt in the {'LFM' if fam == 'lfm' else 'Hermes'} tool format", flush=True)
    rows, right = [], 0
    for text, needs in TURNS:
        t0 = time.perf_counter()
        # no think prefill here: a tool call may come before or after the model's own think block
        raw = em.emit(text, system=system, max_new_tokens=160, temperature=0.0)
        call = parse_call(raw)
        ok = (call is not None and not call.get("malformed") and call.get("name") == "news_search") if needs else (call is None)
        right += int(ok)
        rows.append({"text": text, "needs_tool": needs, "call": call, "raw": raw, "ok": ok, "seconds": round(time.perf_counter() - t0, 2)})
        print(f"  {'TOOL ' if needs else 'plain'} {'ok ' if ok else 'BAD'} {text[:48]!r:52s} -> {json.dumps(call, ensure_ascii=False)[:110] if call else 'no call'} | raw: {raw[:90]!r}")
    summary = {"gguf": args.gguf, "family": fam, "n": len(TURNS), "right": right, "calls_when_needed": sum(1 for r in rows if r["needs_tool"] and r["ok"]),
               "silent_when_not": sum(1 for r in rows if not r["needs_tool"] and r["ok"]), "formats": sorted({r["call"]["format"] for r in rows if r["call"]})}
    print("\n[stand-in] tool-call probe:", json.dumps(summary, ensure_ascii=False))
    out = os.path.join(ROOT, "standin", "data", "out", f"tool_call_probe{args.tag}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
