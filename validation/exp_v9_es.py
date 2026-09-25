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
import statistics
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


def fitness(ops, model, tokenizer, batch, rng, max_new_tokens) -> list:
    """One ladder score **per prompt**, not their mean.

    The mean is what gets logged, but the optimizer wants the row: EGGROLL
    §6.3's shaping is measured across the whole (member, prompt) matrix,
    and a mean has already collapsed the axis it needs to see.
    """
    return [score(strip_output(ops.greedy(model, tokenizer, record["prompt"],
                                          max_new_tokens,
                                          system=record.get("system"))),
                  record, rng)
            for record in batch]


def _member_effect(scores: list) -> float:
    """How much of the score spread is the *member*, not the prompt.

    ES ranks members against each other, so it only has something to learn
    from if a member that did well on one prompt tends to do well on the
    next. This is that fraction: the variance of a member's mean (after
    each prompt's own mean is removed) over the variance of the individual
    cells.

    0 is pure scatter — the perturbation changed the output without
    changing the adapter's quality, and every iteration steps along noise.
    1 is a member that is uniformly better or worse. It relates to the
    logged ``shaped_std`` as ``shaped_std ~ sqrt(signal + (1-signal)/m)``,
    so a run with no member effect sits at ``1/sqrt(m)`` and that is what
    the first smoke measured: 0.500 and 0.452 at m=4, against 0.496 for
    noise.
    """
    n, m = len(scores), len(scores[0]) if scores else 0
    if n < 2 or m < 2:
        return float("nan")
    col = [sum(scores[i][j] for i in range(n)) / n for j in range(m)]
    dev = [[scores[i][j] - col[j] for j in range(m)] for i in range(n)]
    cells = [d for row in dev for d in row]
    total = statistics.pvariance(cells)
    if total <= 0:
        return 0.0
    rows = [sum(row) / m for row in dev]
    # a row mean of m independent cells carries total/m of variance by
    # chance alone; subtract it so pure noise reads 0, not 1/m
    return max(0.0, (statistics.pvariance(rows) - total / m) / total)


def held_out(ops, model, tokenizer, split, rng, max_new_tokens, batch: int = 1) -> float:
    """Verify-to-gold on held-out prompts — the number the kill criterion
    reads. No ladder: this is the same measurement step 0 made. ``batch``
    > 1 decodes that many prompts at once, left-padded."""
    from eval_emitter_vm import gold_matches

    if batch > 1:
        texts = ops.greedy_many(model, tokenizer, [r["prompt"] for r in split], max_new_tokens,
                                systems=[r.get("system") for r in split], batch=batch)
    else:
        texts = [ops.greedy(model, tokenizer, r["prompt"], max_new_tokens, system=r.get("system"))
                 for r in split]
    hits = 0
    for record, raw in zip(split, texts):
        text = strip_output(raw)
        ok, result = vm_result(text)
        hits += int(bool(ok and gold_matches(result, record.get("gold"))))
    return hits / max(len(split), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    # the BASE, not the merged checkpoint: the adapter goes on unmerged, so on a merged model it is applied
    # twice. 2026-09-24 smoke run: merged + adapter read 0.000 held-out where merged alone ran as v14e.
    ap.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    ap.add_argument("--adapter", default=r"D:/My Drive/cubbyllm/standin/"
                                         r"emitter_lfm25_2p6b_v14e_nochain_full/adapter")
    ap.add_argument("--task", default="arithmetic")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=16)
    # 0.003 is measured, not copied: it is the knee of the sigma curve in
    # cubbyllm/ops/es.py. Above it a member stops terminating and one hung
    # row holds the batch open (0.01 measured 7.5x slower); below it every
    # member decodes identically and there is nothing to rank.
    ap.add_argument("--sigma", type=float, default=0.003)
    # lr/sigma is the paper's alpha and is what sets the step; 3e-6 at
    # sigma 0.003 is its 0.001. Move sigma, move this. See ops/es.py.
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd",
                    help="sgd is EGGROLL's own choice for the reasoning runs "
                         "(Table 10) and stores no gradient; adam/adamw is what "
                         "it uses for RL and for quantised distillation, and "
                         "costs three copies of the factors (~966 MB here)")
    ap.add_argument("--opt-lr", type=float, default=1e-4,
                    help="learning rate for --optimizer adam/adamw. NOT the same "
                         "quantity as --lr and cannot share a sweep with it: Adam "
                         "normalises by its second moment, so its step is ~opt-lr "
                         "per coordinate whatever the estimate's scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=20, help="iterations between held-out reads")
    ap.add_argument("--probe", type=int, default=40, help="held-out prompts per progress check")
    # 768, as step 0 (eval_emitter_vm) read v14e: the arithmetic programs are p50 282 / p90 425 / p99 577
    # tokens (max 855), so the old 320 cut a third of them off -- 8 of the 14 prompts llama.cpp got right
    # and grilly got wrong in the 2026-09-24 parity check were truncated, not wrong
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--null-control", action="store_true",
                    help="shuffle fitness across members: must NOT improve held-out")
    ap.add_argument("--final-n", type=int, default=0,
                    help="held-out prompts for the DECISION read; 0 means the whole split, "
                         "which is what the 2-SE bar was computed from. Only a smoke run "
                         "should set this, and its number does not decide anything.")
    ap.add_argument("--serial", action="store_true",
                    help="score members one at a time (es.member) and read held-out one prompt at a time, "
                         "as before 2026-09-24; the default batches both (grilly.infer.lora.population, "
                         "left-padded prompts)")
    ap.add_argument("--prompts-per-batch", type=int, default=4,
                    help="minibatch prompts decoded together, every member a row per prompt "
                         "(population x this many rows); bounded by memory")
    ap.add_argument("--held-out-batch", type=int, default=16,
                    help="held-out prompts decoded together (ignored with --serial)")
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
          f"sigma {args.sigma}, lr {args.lr} (alpha {args.lr / args.sigma:.4g}), "
          f"optimizer {args.optimizer}"
          f"{f' @ {args.opt_lr}' if args.optimizer != 'sgd' else ''}"
          f"{' | NULL CONTROL' if args.null_control else ''}", flush=True)

    t0 = time.perf_counter()
    model, tokenizer, factors = ops.load_emitter(args.model, args.adapter)
    print(f"[v9] loaded in {time.perf_counter() - t0:.1f}s | {len(factors)} LoRA factors", flush=True)
    es = ops.evolution_strategy(factors, sigma=args.sigma, lr=args.lr,
                                population=args.population, seed=args.seed)
    opt = ops.optimizer(factors, args.optimizer, args.opt_lr)

    curve, checks = [], []
    read_batch = 1 if args.serial else args.held_out_batch
    start = held_out(ops, model, tokenizer, test[: args.probe], rng, args.max_new_tokens, read_batch)
    print(f"[v9] held-out before any step ({args.probe} prompts): {start:.3f}", flush=True)
    checks.append({"iteration": 0, "n": args.probe, "verify_to_gold": start})
    if args.probe >= 10 and start < 0.3:
        # v14e reads 0.775 on this split; below 0.3 the loaded weights are not v14e, and every
        # iteration after this would tune something else for hours
        print(f"[v9] ABORT: {start:.3f} before any step -- the base + adapter does not reproduce v14e "
              "(is --model the merged checkpoint? the adapter must go on the base)", file=sys.stderr)
        return 3

    for iteration in range(1, args.iterations + 1):
        batch = rng.sample(train, min(args.minibatch, len(train)))
        step_t0 = time.perf_counter()
        scores = []
        if args.serial:
            for member in range(es.population):
                with es.member(member):
                    scores.append(fitness(ops, model, tokenizer, batch, rng, args.max_new_tokens))
        else:
            # members x prompts as left-padded batches, every member a row per prompt; every member of
            # a prompt faces the SAME counterfactual numbers (a per-prompt rng), so members are ranked
            # on one test, not 32
            scores = [[0.0] * len(batch) for _ in range(es.population)]
            texts = ops.population_greedy_many(model, tokenizer, es, [r["prompt"] for r in batch],
                                               args.max_new_tokens, systems=[r.get("system") for r in batch],
                                               per_batch=args.prompts_per_batch)
            for j, record in enumerate(batch):
                for m in range(es.population):
                    prng = random.Random(f"{args.seed}:{iteration}:{record.get('id', j)}")
                    scores[m][j] = score(strip_output(texts[m][j]), record, prng)
        used = list(scores)
        if args.null_control:
            # the update is then uncorrelated with what was measured
            rng.shuffle(used)
        ops.update(es, opt, used)
        means = [sum(row) / max(len(row), 1) for row in scores]
        # the shaped spread is what actually multiplies the step, and it is
        # not fixed under prompt-centring the way centred ranks' 0.289 is.
        # Logging it is how a diverging run gets diagnosed rather than guessed.
        shaped = statistics.pstdev(es.shaped) if len(es.shaped) > 1 else 0.0
        curve.append({"iteration": iteration, "mean": sum(means) / len(means),
                      "best": max(means), "shaped_std": shaped,
                      # the whole (member, prompt) matrix, because the summary
                      # cannot answer the question that decides whether the run
                      # is worth its budget: is a member that scored well on one
                      # prompt more likely to score well on another? If not,
                      # every iteration ranks scatter. `signal` below is that
                      # read; `shaped_std` alone cannot separate it from noise.
                      "scores": scores, "signal": _member_effect(scores),
                      "seconds": time.perf_counter() - step_t0})
        print(f"[v9] {iteration:>4}/{args.iterations} train fitness "
              f"mean {curve[-1]['mean']:.4f} best {curve[-1]['best']:.4f} "
              f"shaped_sd {shaped:.3f} ({curve[-1]['seconds']:.1f}s)", flush=True)

        if iteration % args.every == 0:
            value = held_out(ops, model, tokenizer, test[: args.probe], rng, args.max_new_tokens, read_batch)
            checks.append({"iteration": iteration, "n": args.probe, "verify_to_gold": value})
            print(f"[v9] held-out at {iteration} ({args.probe} prompts): {value:.3f}", flush=True)

    # the decision read: the WHOLE split, because the bar is set by its SE
    decision = test if args.final_n <= 0 else test[: args.final_n]
    final = held_out(ops, model, tokenizer, decision, rng, args.max_new_tokens, read_batch)
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
