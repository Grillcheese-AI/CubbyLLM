"""panel_compiler — round 2 of the 2026-09-28 panel: the trunk as a compiler of branch programs.

Wired: STANDALONE (validation only; the final model never calls OpenRouter).

Nick, 2026-09-28, after round 1 (`panel_fluent_harness.py`, `docs/panel_fluent_harness.md`): what he
wants is the model compiling a question into CubeLang programs, one per branch (choices x timelines),
the VM running each branch in a sandbox and feeding the outcome back. He brought a proposal from
another model (a CfC-style "liquid" module emitting an opcode matrix per branch). This round gives the
panel the system again, round 1's consensus to object to, the proposal, and the VM's real state.

    python validation/panel_compiler.py            # ask, cache, write docs/panel_compiler.md
    python validation/panel_compiler.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from panel_fluent_harness import PANEL, QUESTION as ROUND1, SYSTEM, ask  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "panel_compiler.md"

SYSTEM_PART = ROUND1.split("# Three problems")[0].strip()

QUESTION = SYSTEM_PART + r"""

# Round 1, in brief (six reviewers, asked independently)

- A (fluency at 450M): all six said two adapters switched by the host per turn (the grounded talk
  adapter frozen as it is; a separate conversation adapter), before any larger LoRA, mixture or inserted
  block, and a diagnostic first (fact-only vs chat-only vs mixed adapters at equal tokens). Several
  added: continued pretraining is where fluency comes from; the chat path needs its own guard (every
  entity and number it names must be in a per-turn sanctioned set, since it has no VM value to check);
  a partial answer ("I don't have the cause - it was 1683") is a slot format for the fact adapter, not
  fluency. Most said not to distil the host's scaffold into the weights.
- B (harness): one deterministic process; typed tools with effect classes, deny by default, every call
  ledgered; a task stack or DAG with budgets; boot preflight that refuses to serve on a failed gate, with
  thresholds as signed config (not recalibrated at boot); the model proposes (programs) and phrases, never
  decides truth, permissions, memory writes or tool safety. Four of six: build a minimal harness first.
- C (HDC FFN): not as an FFN replacement (crosstalk; learned keys beat random codes); the x, y, z, t
  anchoring is sound (spatial semantic pointers) but belongs in the stores and Cubby-Man's episodic
  memory; the version worth one experiment is a zero-gated associative memory in one layer, written from
  the current context only, aimed at the measured binding deficit (52%).

# D. The trunk as a compiler of branch programs (the owner's goal)

What the owner wants: the model reads a question, lays out the alternatives (choices a, b, c) against the
contexts they are judged in (timelines or eras x, y, z: "what would change if X had not happened", "in
1900" vs "in 2028"), and emits one CubeLang program per branch. The VM executes each branch in isolation,
and what comes back (the verdict, how far it ran, what changed) steers what is emitted next. Today
counterfactual history questions ("what would change if X had not happened") are answered by following the
history graph's links downstream of X: 18% spoken and correct on reworded questions, the weakest kind.

A proposal the owner received from another model (summarised): a CfC-style "liquid" module takes the
question embedding, a [choices] tensor and a [timelines] tensor, broadcasts them into a
[choices x timelines] grid, fuses each cell with an MLP, applies a continuous-time gate
(state = f*g + (1-f)*h, with f = sigmoid(...) described as the time constant tau), and projects each
cell to a FIXED-LENGTH block of opcode logits (max_code_len x vocab) that are argmax-decoded into
bytecode and run in a VM sandbox. The time constant is said to set how deep a branch runs; faulty branches
"adjust the weights via backpropagation" at runtime. It also asserted, wrongly, that in our system "only
matrices cross into the VM" (the rule is the opposite: only symbols cross), and asked whether the liquid
parameters should update during runtime or only in the nightly loop.

The VM as it is (CubeLang, from its source and spec):
- A typed scripting language compiled to a 0x00-0xFF bytecode: CREATE, ASSIGN, arithmetic, PUSH/POP,
  QUERY, STORE/RECALL, BIND_ROLE, UNIFY, COND and jumps to labels, CALL (intra-program, deny-by-default
  resolution of modules), RETURN, arrays; plus VSA ops (bind, unbind, cleanup against a codebook).
- Registers, stack, storage and accumulator are VM-global; there are no call frames. LOOP is currently a
  structural marker executed as a single pass (iterating a collection is a documented follow-up). Every
  jump spends a 1,000,000-jump budget. A debug trace of executed ops exists.
- ASK suspends a program when it has grounded several real candidates and cannot choose; the host answers
  and the program resumes, and resume rejects anything that was not offered (the host may choose, never
  supply). The VM runs as a resident process over protobuf on stdio, verifies a program before executing
  it, and returns (symbol, similarity) per hop, plus verified or refused with a reason.
- The spec describes ctx.snapshot / restore / fork / merge (try a branch in an isolated copy, then
  merge or discard it); fork is specified, not implemented.
- The 128k tokenizer holds CubeLang opcodes as atomic tokens. The program emitter is trained on
  VM-verified programs (517 verified chain programs plus arithmetic, role-binding and kernel families) and
  is today served by a stand-in model behind the same interface.
- The trunk's recurrent layers (MinGRU-style) already update as a gated interpolation of the kind the
  proposal calls liquid; they have no explicit time input.

Questions:
1. Is "the trunk compiles a question into branch programs, the VM runs each branch, the outcome steers
   the next emission" sound for this system? Give the right split of work between the model (what it
   emits), the host (who enumerates branches, sets budgets, compares outcomes) and the VM (what it runs,
   what it returns).
2. Critique the proposal's mechanics specifically: fixed-length opcode logits per grid cell, argmax
   decoding, a gate standing in for execution depth, runtime backpropagation. Keep what is right.
3. The "liquid" part: does a time-conditioned gate in the trunk (CfC-style, time as an input) buy anything
   for reasoning across eras, compared with putting time into the data and the VM (a timeline as a role
   bound in the program, T^t position codes in the stores)?
4. Feedback: what exactly should the VM return per branch, and how should it train the emitter - at
   night through the gated promotion rule, or ever at runtime? Our rule is that serving never changes
   weights.
5. Where this sits in the order you would do A, B, C in, the smallest first version, and its pre-registered
   gate. Object to round 1's consensus if you think it is wrong.

Constraints as before: no external LLM at serve time; 450M target; local serving on a 12 GB GPU, batch 1;
the host disposes and the VM verifies; only symbols cross into the VM; every claim measurable.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--models", nargs="*", default=PANEL)
    a = ap.parse_args()
    with cf.ThreadPoolExecutor(max_workers=len(a.models)) as ex:
        futs = {m: ex.submit(ask, m, a.offline, a.max_tokens, a.effort, SYSTEM, QUESTION) for m in a.models}
        answers = []
        for m in a.models:
            rec = futs[m].result()
            answers.append(rec)
            u = rec.get("usage") or {}
            print(f"  {m:34s} {'ERROR: ' + rec['error'] if rec.get('error') else ''}"
                  f"{len(rec['text'].split()):>6} words  {rec.get('wall_s', 0)}s  finish={rec.get('finish')}"
                  f"  out={u.get('completion_tokens', '?')}  cost={u.get('cost', '?')}", flush=True)
    ok = [r for r in answers if r["text"]]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Panel, round 2: the trunk as a compiler of branch programs (2026-09-28)\n\n"
                "Round 1: `docs/panel_fluent_harness.md`. Same six models, asked independently, temperature 0.3, "
                f"reasoning effort `{a.effort}`. **Model opinions, not findings.**\n\n")
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
