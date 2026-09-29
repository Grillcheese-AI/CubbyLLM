"""panel_round3 -- the harness spec, the branch-compiler goal and the CubeLang VM as it is, put to the strongest
models on OpenRouter (2026-09-28, Nick: "give the specs and goal including vm cubelang specs to more powerful
models ... openai luna and terra ... glm latest").

Wired: STANDALONE (validation only; the final model never calls OpenRouter). Model ids are read from the live
list, never remembered (`panel_cube_limits.py`'s rule): the ids below were listed on 2026-09-28.

    python validation/panel_round3.py            # ask, cache, write docs/panel_round3.md
    python validation/panel_round3.py --offline  # re-render from cache, no spend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from panel_fluent_harness import QUESTION as ROUND1, SYSTEM, ask  # noqa: E402
from panel_compiler import QUESTION as ROUND2  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "panel_round3.md"
PANEL = [                                   # the strongest tier per vendor on the live list, 2026-09-28
    "openai/gpt-6-luna-pro",
    "openai/gpt-5.6-terra-pro",
    "anthropic/claude-opus-5.5",
    "google/gemini-3.1-pro-preview",
    "z-ai/glm-5.3-prime",
    "qwen/qwen3.8-max-prime",
    "x-ai/grok-4.7",
    "deepseek/deepseek-v4-pro-0813",
]

SYSTEM_PART = ROUND1.split("# Three problems")[0].strip()
ROUND1_BRIEF = ROUND2.split("# Round 1, in brief")[1].split("# D. The trunk as a compiler")[0].strip()
HARNESS_SPEC = (ROOT / "docs" / "research" / "2026-09-28-harness-v0.md").read_text(encoding="utf-8")

VM_SPEC = r"""
# CubeLang and its VM, as they are (from the source and docs of the `cubelang` repo, 2026-09-28)

**Language.** A typed scripting language for "self-evolving reasoning modules": `program X implements I { storage {...}
@system @once public function constructor() {...} function f(a: T): U {...} }`. Programs are always bound to an
interface; the VM registry seeds ISolver (parse/solve/verify), ISolverLearn (+ learn), IAgent (think(input, ctx),
act(decision, ctx), observe(result, ctx)). Statements are three-address: `create x : type; assign x = v; add x, n;
push x; pop x; query x; store x, "key"; recall "key"; bind x, ROLE, val; remember x; sum x; return v;` plus
`if`, `while`, `for (let x of arr)` and `let`. A non-trivial expression as an argument or a `let` right-hand side
(a binary op, a struct or map literal, a method call, an index) is a compile error, by design: the three-address
surface IS the language. `match` arms all executed unconditionally until 2026-09-14 (now a strict error); 24
"extended" mnemonics (infer, map_roles, filter, score, debate, forge, explore, sync, temporal_bind, analogy, ...)
lex, parse and compile but only write a trace line. `asm { MNEMONIC operands; }` is an escape hatch to raw opcodes.
Permission attributes (@external, @internal, @restricted, @ratelimit, ...) are advertised and not enforced.

**Bytecode.** One byte per opcode (0x00-0xFF). Real: CREATE, DESTROY, ASSIGN, ADD, SUB, MUL, DIV, TRANSFER, COPY,
PUSH, POP, COMPARE, QUERY, STORE, RECALL, BIND_ROLE, UNBIND, COND, JMP, LABEL, CALL, RETURN, MAKE_ARRAY, LEN, INDEX,
SET_INDEX, REMEMBER, FORGET, SUM, ASK, NEWVAR. `while` and `for` lower to COMPARE + COND + JMP with labels, so the
language iterates for real; the bare LOOP opcode (0x12) is a structural marker. Every jump spends a per-run budget of
1,000,000; recursion is capped at depth 64.

**Values.** Int, Float, Str, Bool, Null, Array, Map, Hvec (a bipolar hypervector in the memory's space).

**VM state is global**: named registers, a stack, storage (STORE/RECALL, exact key), a hippocampal associative
memory (STORE also binds the key's hypervector, RECALL is a cosine cleanup over stored keys), a codebook (symbol ->
deterministic hypervector), a knowledge store (QUERY: exact normalized-key lookup with provenance; a miss pushes an
EMPTY chunk array, never a nearest neighbour), an accumulator, an event log. There are no call frames: CALL swaps
the register map and the frame map for the callee and restores them after; ASK inside a CALL is an error.

**Binding.** BIND_ROLE binds a filler into a register's frame; UNBIND (`recover`) is a cosine cleanup whose candidate
pool is the frame's own bindings (scoped in 2026-09, after an absent role returned "umbrella" at 0.02 from an
unrelated frame). The winning similarity is surfaced (`last_recover_similarity`). An unbound role still returns
noise, not Null (an open item).

**ASK.** `ask "question", c0, c1, ...;` suspends the program with the candidates. The host answers; `resume` pushes
the answer on the stack and REJECTS any answer that is not structurally identical to an offered candidate: the host
may choose, never supply. Resumption captures only the interpreter locals (pc, flag, jump budget) since state is
global.

**Transport.** `cubelang run-proto`: a resident process reading length-prefixed protobuf RunRequests on stdin
{program source, args, fn_name, answers (one JSON value per ASK, in order; the program is re-executed from
scratch and consumes them), knowledge_path (a facts .jsonl the process caches by path and mtime and CLONES into
the fresh per-request VM)}. Every request gets a fresh VM, so requests never share state. RunResult {ok, one of:
symbol | error | suspended{question, candidates, program, function}, optional similarity}. The host's Python
client (`cubbyllm/bridges/cubelang_client.py`) verifies before executing (strict) and returns (symbol, similarity)
per hop plus verified or refused with a reason. run-proto is unconditionally strict.

**Context in the spec, not in the VM.** `docs/SPEC.md` describes `ctx.snapshot() / restore() / fork() / merge() /
discard()` (try a branch in an isolated copy of registers, stack, storage, history), `agent` and `cloned_agent`
(forked state, independent evolution), containers, events and channels, `@cron`, `extend` (runtime
self-modification). None of these are implemented; the parser knows `ctx` only as a type name.

**Deny by default.** A module not named in the program's `use` list is invisible at CALL time (a registry of
native modules, `override` resolution with compile-time validation).

**The emitter that writes programs.** A stand-in model (a 2.6B LoRA fine-tune on 517 VM-verified chain programs plus
arithmetic, role-binding and kernel families) behind the same interface base450m will implement. The tokenizer holds
the opcodes as atomic tokens. Every emitted program is verified by the VM before it runs; the harvest of verified
programs is the emitter's training set.
"""

QUESTION = SYSTEM_PART + "\n\n# The two panels so far (six mid-tier models, 2026-09-28)\n\n" + ROUND1_BRIEF + r"""

Round 2 (the owner's goal: the trunk compiles a question into one CubeLang program per branch -- choices a, b, c
against timelines x, y, z -- the VM runs each branch in isolation, the outcome steers the next emission; a
counterfactual history question is today answered by walking the graph's downstream links, 18% spoken and
correct when reworded): all six said the goal is sound with the host enumerating branches and setting budgets,
the model emitting one program per branch autoregressively (not fixed-length opcode blocks), the VM returning a
structured report per branch (verdict, reason, hops, jumps used, facts touched, the diff between branches) and
the emitter trained only at night through a promotion gate. All six said a CfC-style "liquid" time gate in the
trunk buys nothing for eras: time is a role bound in the program and a T^t position code in the stores. All six
named `ctx.fork` (specified, not implemented) as the prerequisite. Order: harness first, then the two adapters,
then the branch runner, then the HDC memory experiment.

""" + VM_SPEC + "\n\n# The harness spec, v0 (written today; a first implementation exists)\n\n" + HARNESS_SPEC + r"""

# What we want from you

You are a stronger reviewer than the first two panels. The system brief, the VM as it is, and the harness spec are
above. Be specific; name what is wrong; keep what is right; every proposal with its failure mode and its
measurement.

1. **The harness spec.** Where does it fall short of an agent harness the system can grow into (multi-step tasks,
   sub-goals, worlds, the night)? What would you change before it is built further? Is anything in it wrong for a
   system whose model does not ground (binds at chance, memory over context 22/24)?
2. **Fork, and the VM.** Two designs for the branch runner: (a) at the wire, a `branch` on the RunRequest (facts to
   exclude, facts to assume, applied to the cloned store before the run; each branch a fresh VM; the report per
   branch); (b) inside the VM, `ctx.snapshot/restore/fork/discard` as opcodes, so one program can try a branch and
   back out. Which first, and what exactly should the branch report carry? What else in this VM would you change
   for the branch runner and the harness -- call frames, Null for an absent role, the permission attributes,
   the loop budget, the trace-only mnemonics?
3. **The program the harness runs.** Should the harness's turn itself be a CubeLang program (an IAgent whose
   think/act/observe are the step, the host answering its ASKs), or Python with the VM as a tool? What does each
   cost in safety and in what the emitter can learn?
4. **The 450M and fluency.** The two panels' answer was two host-switched adapters plus a sanctioned-set guard for
   chat. Is that right at 450M, given the base does not ground? What would you measure first?
5. **The HDC memory** (a zero-gated associative layer written from the current context, aimed at binding 52% ->
   85%): worth one experiment, or not? If yes, the exact experiment; if no, what instead for binding.
6. **Order and the first gate.** The smallest sequence of builds that turns the modules into one agent, each with
   its pre-registered gate. Disagree with the two panels where they were wrong.

Constraints: no external LLM at serve time; the target is 450M; local serving on a 12 GB consumer GPU at batch 1;
the host disposes and the VM verifies; only symbols cross into the VM; serving never changes weights; every claim
measurable against a pre-registered gate.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--effort", default="medium")
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
                  f"  in={u.get('prompt_tokens', '?')} out={u.get('completion_tokens', '?')}  cost={u.get('cost', '?')}", flush=True)
    ok = [r for r in answers if r["text"]]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Panel, round 3: the harness spec, fork and the VM, put to the strongest models (2026-09-28)\n\n"
                "Rounds 1 and 2: `docs/panel_fluent_harness.md`, `docs/panel_compiler.md`. Asked independently, "
                f"temperature 0.3, reasoning effort `{a.effort}`. **Model opinions, not findings.**\n\n")
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
