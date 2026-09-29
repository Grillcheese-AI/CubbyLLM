"""panel_fluent_harness — put three open design problems to the panel of frontier models.

Wired: STANDALONE (validation only; the final model never calls OpenRouter).

The problems (Nick, 2026-09-28): (A) fluent conversation without losing the grounded behaviours, at
450M; (B) a bootstrap and an orchestrator, so the whole system acts as an agent harness; (C) an
HDC-based FFN whose units react like linked neurons, anchored at x, y, z and t. Same mechanics as
`panel_cube_limits.py`: every model asked the same prompt independently (in parallel here), every
non-empty answer cached by (model, prompt, settings), the answers written side by side.

    python validation/panel_fluent_harness.py            # ask, cache, write the report
    python validation/panel_fluent_harness.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin")):
    if p not in sys.path:
        sys.path.insert(0, p)
if not os.environ.get("OPENROUTER_API_KEY") and (ROOT / "notebooks" / ".api.key").exists():
    os.environ.setdefault("OPENROUTER_KEY_FILE", str(ROOT / "notebooks" / ".api.key"))   # off-repo key

from openrouter import API, _key  # noqa: E402

CACHE = ROOT / "standin" / "data" / "out" / "panel_cache"
OUT = ROOT / "docs" / "panel_fluent_harness.md"
PANEL = [
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "moonshotai/kimi-k3",
    "x-ai/grok-4.6",
    "deepseek/deepseek-v3.2",
]

SYSTEM = (
    "You are advising on the architecture of a small neurosymbolic language system. You are one of "
    "several independent reviewers; be specific, and be willing to say a premise is wrong. Prefer one "
    "concrete mechanism with its failure mode over a survey of options. For anything you propose, say how "
    "it would be VALIDATED: what measurement would show it works, and what would show it does not. Where "
    "a number matters, estimate it and say that it is an estimate."
)

QUESTION = r"""
# The system (CubbyLLM, as of 2026-09-28)

One principle runs through it: the model proposes, the host disposes, a verifier decides. Nothing is
spoken unless a chain verified by a symbolic VM reaches it, and knowledge lives outside the weights.

**Layers**
- Representation: hypervectors. Words map to quantised FHRR block codes (80 blocks x 128 phases,
  D = 10,240) from a fixed 60,151-word table; an 80-byte compact form is searched without a codebook.
  The perception side (an SNN front end over camera and audio) reads out into the same space.
- Stores, outside the weights: a history graph (940,897 events, 45,588 causal and temporal links); a
  wiki world of 552,297 template facts; capsule facts with provenance and verified-use counts; an
  episodic store of VM-certified chains; a skill library of composition rules mined nightly; a signed
  ledger of every VM verdict; an encrypted vault for conversations.
- Verifier: CubeLang, a Rust VM for vector-symbolic programs (bind, unbind, cleanup). Concepts go in and
  (symbol, similarity) comes out; raw vectors never cross. Every hop of a reasoning chain must clear a
  threshold, and an absent-role control must stay silent.
- Host (the "brain", Python): reads the kind of question, resolves "it"/"that" from a context graph
  (long-term events, the conversation, per-ask reasoning traces), looks up the event, runs a plan through
  the walk and the VM, then speaks, refuses, or asks a clarifying question. Every draft passes a
  name-and-number guard, a value check (the reply may state only the value the VM returned) and a claim
  check (everything named from the facts must be tied to what the facts tie it to). A 5-hormone
  neurochemical ODE bends routing caution and tone, never facts.
- Trunk: base450m, our own model.
  - d1024, 32 layers: 21 MinGRU-style recurrent layers and 11 window-512 attention layers (every 3rd,
    4 heads x 256); SwiGLU FFN 2048.
  - 128k BBPE tokenizer with CubeLang opcodes as atomic tokens; untied embedding and head (262M of its
    576M parameters; the trunk is 314M).
  - A context-generated per-layer adapter (theta = f(c): 16 bases x rank 8, mixed by a frozen,
    offline-trained context router).
  - 2.77B training tokens (~9 per trunk parameter), 16.8 A100-hours; held-out loss 2.36. Serves on our
    own Vulkan engine on a 12 GB consumer AMD GPU at ~40 tok/s decode, batch 1.
  - Probes: fluent English and French prose; in-context copying works (copy loss 1.2); but it does not
    ground. It binds a value copied from context to the entity asked about at chance (52%), and answers
    from memory over a contradicting context 22 times in 24.
- Adapters on the base, routed per turn by the host (a frozen tag router, never learned online): a talk
  adapter (LoRA r16 on all 181 projections, 7.9M parameters) trained to say what the VM returned, and a
  program emitter (text -> CubeLang; still served by a stand-in model behind the same interface).
- Learning. Nothing learned changes what may be spoken; it only proposes. The VM verdict is the only
  reward (a striatum-like arbiter reorders plan proposers by reward-prediction error). A refusal becomes
  fetch -> gate -> store -> a second walk. A nightly sleep cycle replays the day into episodes, skills and
  adapter training data. A forge writes new CubeLang tools that the VM must certify. An adapter
  lifecycle (detect -> spawn -> train -> promote -> route/prune) is specified; its detector and promote
  rule are not built yet.
- An embodied agent, Cubby-Man, lives on the same stack in a 3-D maze it learns by exploring: walls
  learned from refused moves, hypotheses kept apart from facts, power-moves it invents and the VM certifies.

**Measured state of the talk adapter** (held-out data, pre-registered gates; "behind the host" = after
the host's checks)
- The mix trade-off. With fact records at 11% of the talk mix (more conversational data), the fact
  families were: relation 96.2%, bind 87.0%, counter-fact 91.2%, "the facts don't say" 70.3%, and 18
  answers that were wrong yet named only grounded things. With facts at 63%: 98.8 / 100 / 93.8 / 89.1%,
  and 6 such answers. More "don't say" records lifted that family but cost right answers elsewhere (384
  vs 389 of 407 behind the host). Behind the host's checks, 0 wrong answers were spoken in all three.
- History questions in plain words: spoken and correct 59% when reworded, 90% when the event is named;
  1% spoken wrong. Follow-ups that name no event: 97% resolved through the context graph, 94.7% spoken
  and correct.
- It answers like a form, by design: the adapter reads only the host's canonical restatement of the
  question, never the person's words or earlier turns; the value check lets it say one value; 63% of its
  data is one-sentence fact answers; the host has three moves (answer, ask which one, "The facts don't say.").

**Other recent results**
- Growth. The base was grown exactly (identical loss at step 0) to 1.29B (depth x2 + FFN x2, zeroed
  exits) and 1.78B (width x2). After one equal hour of training, every grown arm was behind the 450M
  continued on all 9 shared held-out sources (+0.03 to +0.08 nats). The base is short of data before
  parameters, and it is now being continued on 3.2B unseen tokens. The target stays 450M.
- The system's parts exist as separate modules: a serving pipeline (sense -> neurochemistry -> fast
  route -> cortex router -> memory / reasoning / talk / plugin cortices), the verified ask loop with an
  event stream to a live 3-D control panel, the context graph, the sleep cycle, the forge. No single
  process owns boot, state, scheduling and the turn loop.

# Three problems

## A. Fluent without losing capacity, at 450M

We need Cubby to hold a natural conversation in English and French: small talk, follow-ups,
corrections, "tell me more", varied wording, a fuller "I don't know" ("I don't have what caused it - I
know it was 1683"). It must keep every grounded behaviour above: the fact families, zero wrong answers
behind the host, the follow-ups, and the program emitter's VM-verified rates. The data says fluency and
grounding compete, at least in one mix, for the same 7.9M adapter parameters or the same base.

Options we have written down: a mode line; two adapters the host switches per turn; a larger LoRA
(rank 64, ~31M); new identity-initialised blocks inserted into the frozen base, trained alone and skipped
on the facts path; experts in two tiers (the host picks the group by turn kind; a learned mixture of
LoRA experts only within that group); rank-wise gating; per-layer rank. Separately: moving the host's
scaffolding into the weights (train with the scaffold, update as if it had not been there, rewarded only
by the host's checks).

Which is right, and why? Or is the premise wrong? Is the bottleneck the 450M base's own language
ability (so fluency should come from continued pretraining and its data mix, not an adapter), the
adapter's size, or interference between tasks? What would you measure first to tell these apart?

## B. A bootstrap and an orchestrator: the whole system as an agent harness

We want the whole system to act as an agent harness: one process that boots the brain, owns its state,
runs the turn loop, schedules day and night, and exposes the host's capabilities as tools.
1. The loop: what one step of the harness is (observe -> plan -> call tools -> verify -> speak or ask ->
   record), where multi-step tasks and sub-goals live, and what stops a runaway.
2. Tools: VM programs, lookups over the stores, fetch-and-learn, the forge (new tools, VM-certified),
   world plugins (the maze, future worlds), memory reads and writes. How are they declared, permissioned
   (deny by default; the guards above; the explicit-content gate; the vault) and audited (the signed ledger)?
3. Where the 450M model sits: does it propose tool calls (as a policy), only phrase results, or both
   through separate adapters? Since it does not ground on its own, what must the harness never let it decide?
4. Bootstrap, in both senses: (i) cold start of the running system - load stores and adapters, self-test
   against the gates, calibrate thresholds, refuse to serve if a check fails; (ii) a capability or a world
   starting from almost nothing (Cubby-Man starts knowing two facts) and growing from its own verified traces.
5. The smallest first version, and how to show the harness beats today's separate modules beyond "it runs".

## C. An HDC-based FFN whose units react like linked neurons

The idea is the owner's and deliberately rough. Replace or augment the dense FFN with a feed-forward
memory built in the same hypervector space the rest of the system uses. Its units are hypervectors, and
they are linked: activating one also activates the units bound to it, like linked neurons (spreading
activation, cleanup). Further, a multi-layer version in which every unit is anchored at a position
x, y, z and a time t, so that units near each other in space or time respond together and sequences are
laid down along t. The block codes are FHRR phases, so positions can be encoded by fractional powers of
base vectors (X^x * Y^y * Z^z * T^t, with * as binding), and similarity falls off with distance.

Hoped for: more capacity per parameter at 450M; associations that can be written without gradient
descent; a representation the symbolic side can read, without breaking the rule that only symbols cross
into the VM.

Is this sound? Relate it to what is known: FFNs as key-value memories, product-key and memory layers,
modern Hopfield layers, topographic models, spatial semantic pointers and other VSA/FHRR position codes.
How would it be trained, and inserted into a pretrained 450M model? What is its capacity limit
(crosstalk in superposition), and its failure mode? Does it conflict with "knowledge lives outside the
weights", or is it the natural bridge between the trunk and the stores? What is the cheapest experiment
that would tell us it is worth building?

# Constraints that rule out the easy answers

1. No external LLM at serve time. Frontier models may write training data and judge tests, never answer a user.
2. The target is 450M: growth was just measured and lost at our budget. A proposal that needs a bigger
   base must say so and say why.
3. Serving is local: a 12 GB consumer GPU, our own Vulkan engine, batch 1. Training is one rented GPU
   for hours, not a cluster.
4. The host disposes and the VM verifies. Nothing may let the model's output reach speech unchecked, and
   a learned router may not replace the host's frozen routing online (a router that learns online
   forgets its own routing).
5. Every claim must be measurable against a pre-registered gate; "should improve" is not an answer.

# What we want back

For each of A, B and C: your recommendation, the reasoning, the specific failure mode it introduces, and
the measurement that would show it working or not. Then, across all three, the order you would do them
in and why. Disagree with the framing where it is wrong.
"""


def ask(model: str, offline: bool, max_tokens: int, effort: str, system: str = "", question: str = "") -> dict:
    """One model, one question. Reasoning tokens count against max_tokens (see panel_cube_limits.ask):
    the budget covers thinking AND answering. Settings are in the cache key; an empty answer is never cached."""
    system, question = system or SYSTEM, question or QUESTION     # the defaults keep round 1's cache keys
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f"{model}|{system}|{question}|{max_tokens}|{effort}".encode()).hexdigest()[:20]
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        return {"model": model, "text": "", "error": "not cached and --offline"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.3,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": question}]}
    if effort:
        payload["reasoning"] = {"effort": effort}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Grillcheese-AI/CubbyLLM", "X-Title": "CubbyLLM panel"})
    t0 = time.time()
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except Exception as e:                               # a model that fails is a row, not a crash
        return {"model": model, "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    ch = (raw.get("choices") or [{}])[0]
    rec = {"model": model, "text": (ch.get("message") or {}).get("content") or "",
           "finish": ch.get("finish_reason"), "usage": raw.get("usage") or {},
           "wall_s": round(time.time() - t0, 1), "effort": effort, "budget": max_tokens, "error": None}
    if rec["text"].strip():
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        rec["error"] = (f"no answer: finish={rec['finish']}, "
                        f"{(rec['usage'] or {}).get('completion_tokens', '?')} tokens all spent on reasoning")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-render from cache, spend nothing")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--effort", default="low", help="OpenRouter reasoning effort; empty to omit")
    ap.add_argument("--models", nargs="*", default=PANEL)
    a = ap.parse_args()

    with cf.ThreadPoolExecutor(max_workers=len(a.models)) as ex:
        futs = {m: ex.submit(ask, m, a.offline, a.max_tokens, a.effort) for m in a.models}
        answers = []
        for m in a.models:
            rec = futs[m].result()
            answers.append(rec)
            u = rec.get("usage") or {}
            print(f"  {m:34s} {'ERROR: ' + rec['error'] if rec.get('error') else ''}"
                  f"{len(rec['text'].split()):>6} words  {rec.get('wall_s', 0)}s  finish={rec.get('finish')}"
                  f"  in={u.get('prompt_tokens', '?')} out={u.get('completion_tokens', '?')}"
                  f"  cost={u.get('cost', '?')}", flush=True)

    ok = [r for r in answers if r["text"]]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Panel: fluent at 450M, the agent harness, and an HDC FFN (2026-09-28)\n\n")
        f.write("Asked independently, same prompt, temperature 0.3, reasoning effort "
                f"`{a.effort}`. Cached under `standin/data/out/panel_cache/`; `--offline` re-renders "
                "without spending.\n\n**These are model opinions, not findings.** Each proposal still has to "
                "pass the measurement it names. Where the answers disagree, our framing is probably "
                "underspecified.\n\n")
        f.write("## The question\n\n<details><summary>full prompt</summary>\n\n```\n" + QUESTION.strip()
                + "\n```\n\n</details>\n\n")
        f.write(f"## Answers ({len(ok)} of {len(answers)} returned)\n\n")
        for r in answers:
            f.write(f"### {r['model']}\n\n")
            if r.get("error"):
                f.write(f"*failed: {r['error']}*\n\n")
                continue
            if r.get("finish") == "length":
                f.write("*(truncated at the token budget)*\n\n")
            f.write(r["text"].strip() + "\n\n---\n\n")
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers)
    print(f"\nwrote {OUT}  ({len(ok)}/{len(answers)} answered, ${cost:.2f} reported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
