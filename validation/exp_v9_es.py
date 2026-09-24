"""H-A9 step 1 — tune the program adapter with evolution strategies, scored
by the cubelang VM.

    python validation/exp_v9_es.py --iterations 200
    python validation/exp_v9_es.py --iterations 200 --null-control

The base stays frozen and int8; only the LoRA factors move. The reward is
the VM's verdict, which has no gradient, so the optimizer is forward-only
(H-A9; TODO.md's RLVR rung names ES as the alternative to GRPO).

FITNESS is cubby-lm's TRUNK_VM_EMISSION_CONTRACT §3a ladder, per prompt:

    0.1 * parses + 0.2 * compiles + 0.3 * executes + 1.0 * (result == gold)

a member's fitness is the mean over its minibatch. The ladder matters more
than it looks: a binary right/wrong reward is flat almost everywhere at the
start, and ES estimates a gradient from *differences between members*, so a
flat reward gives it nothing to rank. The rungs make partial progress
visible.

THE REWARD-HACK GUARD. A program that ignores its input and returns the
gold scores 1.0 on that ladder and is worthless. So the 1.0 rung counts
only if the program also survives a **counterfactual**: the prompt's
numbers are perturbed, the same substitution is applied to the reference
program to get the perturbed gold, and the candidate must match that too. A
hardcoder's output does not move when the numbers do, so it caps at 0.6.

    Deviation, stated: the brief named "the counterfactual planter's
    perturbed numbers and entities". That planter
    (`exp_m3_cot_pipeline.py::_fault_instances`) plants faults on *chains*
    — triples and relations — and does not apply to arithmetic, which step
    0 selected as the target family. The guard here is built for this
    family and needs no new semantics: the record's own verified program is
    the oracle for the perturbed problem, so the perturbed gold comes from
    the VM rather than from a re-derivation that could itself be wrong.

CONTROLS, fixed before the run:

  null control  the same budget with fitness shuffled across members, so
                the update is uncorrelated with what was measured. It must
                not improve held-out; if it does, the gain is noise or
                leakage rather than learning.
  kill          held-out verify-to-gold must beat v14e by >= 2 SE inside
                the budget. v14e is 0.775 on arithmetic and the held-out
                read is the whole 329-record split, so SE is 0.023 and the
                bar is 0.821 (H-A9). A 40-record read would put the bar at
                0.907, which is unreachable by construction — the progress
                checks use 40, the decision uses 329.
  no regression the candidate must hold v14e on every other family.

Everything grilly is reached through `cubbyllm.ops` (the single-seam rule).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# appended: standin/hypothesis.py shadows the installed `hypothesis` package
sys.path.append(str(ROOT / "standin"))

from cubbyllm.core.protocols import Wiring  # noqa: E402

DATA = ROOT / "standin" / "data" / "out" / "emitter_sft_v14e_nochain.jsonl"
LOGS = ROOT / "validation" / "logs"
#: v14e's arithmetic verify-to-gold and the bar ES must clear (H-A9 step 0)
BASELINE, BAR = 0.775, 0.821


# ── the ladder ───────────────────────────────────────────────────────────


def strip_output(text: str) -> str:
    """The program out of a generation, as the eval harness reads it."""
    from eval_emitter_vm import strip_fences, strip_think

    return strip_fences(strip_think(text))


def numbers_in(text: str) -> list:
    """The integer literals a word problem states."""
    return [int(m) for m in re.findall(r"(?<![\d.])\d+(?![\d.])", text)]


def perturb(values: list, rng: random.Random) -> dict:
    """A substitution for the prompt's numbers.

    Small and multiplicative so the problem stays sane — a doubled count is
    still a count — and never the identity, or the counterfactual would
    prove nothing.
    """
    mapping = {}
    for v in values:
        if v <= 1:
            continue
        new = v + rng.choice([1, 2, 3, v])          # v + v doubles it
        if new != v:
            mapping[v] = new
    return mapping


def substitute(program: str, mapping: dict) -> str:
    """Rewrite integer literals in a program by ``mapping``.

    Longest first, so replacing 4 does not corrupt 48; a placeholder pass
    keeps a substituted value from being substituted again.
    """
    out = program
    for i, (old, new) in enumerate(sorted(mapping.items(), key=lambda kv: -kv[0])):
        out = re.sub(rf"(?<![\d.]){old}(?![\d.])", f"\x00{i}\x00", out)
    for i, (_, new) in enumerate(sorted(mapping.items(), key=lambda kv: -kv[0])):
        out = out.replace(f"\x00{i}\x00", str(new))
    return out


def vm_result(source: str):
    """``(executed, result)`` from the real cubelang VM, strict compile."""
    from eval_emitter_vm import run_vm

    ok, result, _err = run_vm(source)
    return bool(ok), result


def counterfactual_holds(candidate: str, record: dict, rng: random.Random) -> bool:
    """Does the candidate still track the answer when the numbers move?

    The record's own program is the oracle: substituted the same way and
    run through the same VM, it defines the perturbed problem's gold. A
    candidate that hardcodes its answer cannot follow.
    """
    from eval_emitter_vm import gold_matches

    mapping = perturb(numbers_in(record["prompt"]), rng)
    if not mapping:
        return True                     # nothing to perturb: the guard abstains
    ok_ref, gold = vm_result(substitute(record["program"], mapping))
    if not ok_ref or gold is None:
        return True                     # no oracle for this one, so no verdict
    ok, result = vm_result(substitute(candidate, mapping))
    return bool(ok and gold_matches(result, gold))


def score(candidate: str, record: dict, rng: random.Random) -> float:
    """The §3a ladder for one prompt, with the hack guard on the top rung."""
    from eval_emitter_vm import gold_matches

    if not candidate.strip():
        return 0.0
    total = 0.1                                       # parses: non-empty output
    if "implements ISolver" in candidate and "function solve" in candidate:
        total += 0.2                                  # compiles, structurally
    ok, result = vm_result(candidate)
    if not ok:
        return total
    total += 0.3                                      # executes
    if gold_matches(result, record.get("gold")) and counterfactual_holds(candidate, record, rng):
        total += 1.0
    return total


# ── the loop ─────────────────────────────────────────────────────────────


def records(task: str, split: str) -> list:
    out = []
    for line in DATA.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == task and r.get("split") == split and r.get("gold") is not None:
            out.append(r)
    return out


def fitness(ops, model, tokenizer, batch, rng, max_new_tokens) -> float:
    total = 0.0
    for record in batch:
        text = ops.greedy(model, tokenizer, record["prompt"], max_new_tokens,
                          system=record.get("system"))
        total += score(strip_output(text), record, rng)
    return total / max(len(batch), 1)


def held_out(ops, model, tokenizer, split, rng, max_new_tokens) -> float:
    """Verify-to-gold on held-out prompts — the number the kill criterion
    reads. No ladder: this is the same measurement step 0 made."""
    from eval_emitter_vm import gold_matches

    hits = 0
    for record in split:
        text = strip_output(ops.greedy(model, tokenizer, record["prompt"], max_new_tokens,
                                       system=record.get("system")))
        ok, result = vm_result(text)
        hits += int(bool(ok and gold_matches(result, record.get("gold"))))
    return hits / max(len(split), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:/My Drive/cubbyllm/standin/"
                                       r"emitter_lfm25_2p6b_v14e_nochain_full/merged")
    ap.add_argument("--adapter", default=r"D:/My Drive/cubbyllm/standin/"
                                         r"emitter_lfm25_2p6b_v14e_nochain_full/adapter")
    ap.add_argument("--task", default="arithmetic")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=20, help="iterations between held-out reads")
    ap.add_argument("--probe", type=int, default=40, help="held-out prompts per progress check")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--null-control", action="store_true",
                    help="shuffle fitness across members: must NOT improve held-out")
    ap.add_argument("--final-n", type=int, default=0,
                    help="held-out prompts for the DECISION read; 0 means the whole split, "
                         "which is what the 2-SE bar was computed from. Only a smoke run "
                         "should set this, and its number does not decide anything.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from cubbyllm import ops
    from cubbyllm.ops.es import available

    if not available():
        print("this interpreter has no grilly ES/LFM2 support; run it where grilly2 is "
              "importable", file=sys.stderr)
        return 2
    train, test = records(args.task, "train"), records(args.task, "val")
    if not train or not test:
        print(f"no {args.task} records with gold in {DATA}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)

    print(f"[v9] {args.task}: {len(train)} train, {len(test)} held-out | "
          f"population {args.population}, minibatch {args.minibatch}, "
          f"sigma {args.sigma}, lr {args.lr}"
          f"{' | NULL CONTROL' if args.null_control else ''}", flush=True)

    t0 = time.perf_counter()
    model, tokenizer, factors = ops.load_emitter(args.model, args.adapter)
    print(f"[v9] loaded in {time.perf_counter() - t0:.1f}s | {len(factors)} LoRA factors", flush=True)
    es = ops.evolution_strategy(factors, sigma=args.sigma, lr=args.lr,
                                population=args.population, seed=args.seed)

    curve, checks = [], []
    start = held_out(ops, model, tokenizer, test[: args.probe], rng, args.max_new_tokens)
    print(f"[v9] held-out before any step ({args.probe} prompts): {start:.3f}", flush=True)
    checks.append({"iteration": 0, "n": args.probe, "verify_to_gold": start})

    for iteration in range(1, args.iterations + 1):
        batch = rng.sample(train, min(args.minibatch, len(train)))
        step_t0 = time.perf_counter()
        scores = []
        for member in range(es.population):
            with es.member(member):
                scores.append(fitness(ops, model, tokenizer, batch, rng, args.max_new_tokens))
        used = list(scores)
        if args.null_control:
            # the update is then uncorrelated with what was measured
            rng.shuffle(used)
        es.step(used)
        curve.append({"iteration": iteration, "mean": sum(scores) / len(scores),
                      "best": max(scores), "seconds": time.perf_counter() - step_t0})
        print(f"[v9] {iteration:>4}/{args.iterations} train fitness "
              f"mean {curve[-1]['mean']:.4f} best {curve[-1]['best']:.4f} "
              f"({curve[-1]['seconds']:.1f}s)", flush=True)

        if iteration % args.every == 0:
            value = held_out(ops, model, tokenizer, test[: args.probe], rng, args.max_new_tokens)
            checks.append({"iteration": iteration, "n": args.probe, "verify_to_gold": value})
            print(f"[v9] held-out at {iteration} ({args.probe} prompts): {value:.3f}", flush=True)

    # the decision read: the WHOLE split, because the bar is set by its SE
    decision = test if args.final_n <= 0 else test[: args.final_n]
    final = held_out(ops, model, tokenizer, decision, rng, args.max_new_tokens)
    se = math.sqrt(final * (1 - final) / len(decision)) if decision else float("nan")
    decides = args.final_n <= 0
    beat = bool(final >= BAR) if decides else None
    verdict = ("BEATS" if beat else "does not beat") + " it" if decides else         "NOT A DECISION (partial read; the bar needs the whole split)"
    print(f"[v9] FINAL held-out over {len(decision)}: {final:.4f} (SE {se:.4f}) | "
          f"v14e {BASELINE} | bar {BAR} | {verdict}", flush=True)

    LOGS.mkdir(parents=True, exist_ok=True)
    name = f"exp_v9_es{'_null' if args.null_control else ''}{args.tag}"
    payload = {
        "provenance": {
            "what": "H-A9 step 1 — ES over the program adapter, scored by the cubelang VM",
            "task": args.task, "data": str(DATA), "model": args.model, "adapter": args.adapter,
            "null_control": args.null_control, "baseline_v14e": BASELINE, "bar_2se": BAR,
            "args": vars(args),
        },
        "train_curve": curve, "held_out_checks": checks,
        "final": {"verify_to_gold": final, "n": len(decision), "se": se,
                  "beats_bar": beat, "is_decision": decides},
    }
    (LOGS / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[v9] wrote {LOGS / (name + '.json')}", flush=True)
    return 0


__wiring__ = Wiring.WIRED

if __name__ == "__main__":
    raise SystemExit(main())
