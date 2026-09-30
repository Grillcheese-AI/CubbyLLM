"""train_school_torch -- H-E19's school on Colab: the curriculum (`cubbyllm/reasoning/school.py`) with the 450M
emitter adapter as the pupil and the VM as the only judge.

One round:
  1. the curriculum plans a batch by level (the current level, a review share of mastered ones, extra practice
     on the level before when the current one struggles);
  2. each problem is tried up to its level's budget: k programs sampled at temperature in ONE batch, each
     checked for slots (a copied literal / unknown slot = REFUSED), filled, run in the VM against the gold;
     the first correct one ends the count (order is the sampling order, so "tries until right" is exact);
  3. the curriculum records the outcomes -> the dopamine (reward - expected at that level) and each attempt's
     weight; `mistake_rows` makes the training rows -- the solve, the shown solution when it was not solved,
     and every (wrong, right) pair with its first wrong step and its kind; each solve is written to the
     arithmetic world as an attested procedure;
  4. the update: solve / shown rows get cross-entropy on the program weighted by the dopamine (the dip, for a
     shown one); a (wrong, right) pair gets a preference loss on length-normalized log-probs,
     softplus(-beta * (lp(right) - lp(wrong))), weighted by the dip -- the wrong plan pushed down, the right up;
  5. the gate: a fixed held-out slice is read at one try, greedy, in the VM; the round is KEPT only if it does
     not drop below the best so far (else the adapter goes back to the best): improvement over time, never a
     regression shipped -- the harness's generation rule, applied to the weights.

    python validation/train_school_torch.py --ckpt <base.pt> --tokenizer <bbpe128k.json> \
        --adapter <emitter adapter dir> --pool <slots.jsonl> [...] --heldout <slots.jsonl> --out <dir> \
        [--rounds 20] [--problems 256] [--max-tries 16] [--temperature 0.8]
    python validation/train_school_torch.py --smoke --tokenizer <bbpe128k.json> --pool <slots.jsonl> --heldout <...>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if "--smoke" in sys.argv:
    os.environ.update(CB_D="64", CB_L="4", CB_HEADS="4", CB_WINDOW="16", CB_GEN_BASIS="4", CB_GEN_RANK="2")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from emitter_data import decode_program, table_of  # noqa: E402
from cubbyllm.reasoning.arith_world import ArithmeticWorld  # noqa: E402
from cubbyllm.reasoning.school import (CORRECT, REFUSED, WRONG, Curriculum, answer_written_in,  # noqa: E402
                                       arith_step_values, level_of, mistake_rows)
from cubbyllm.reasoning.slots import SLOT_RX, num_value  # noqa: E402

ADAPTER = "emitter_lora"
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# -- the pool ---------------------------------------------------------------------------

def slot_difficulty(rec: dict) -> dict:
    """What the curriculum orders by, read off a slot record: steps, numbers in words, world constants and
    constants left, distractors (numbers the question states that the program never uses)."""
    prog = rec["program"]
    used = {m.group(0) for m in SLOT_RX.finditer(prog)}
    spans = rec.get("spans", [])
    n_ids = {s["id"] for s in spans if s["kind"] == "N"}
    word_ids = {s["id"] for s in spans if s["kind"] == "N" and s.get("value") is not None}
    k_ids = {s["id"] for s in spans if s["kind"] == "K"}
    left = len(re.findall(r"(?:assign\s+s\d+\s*=|(?:add|sub|mul|div)\s+s\d+\s*,)\s*-?\d+(?:\.\d+)?\s*;", prog)) \
        - len(re.findall(r"assign\s+s\d+\s*=\s*0\s*;", prog))
    return {"steps": len(re.findall(r"create\s+s\d+\s*:\s*quantity", prog)),
            "word_numbers": len(word_ids & used), "unit_constants": len(k_ids & used) + max(left, 0),
            "distractors": len(n_ids - used)}


def load_pool(paths: list[str], split: str = "train") -> dict[int, list[dict]]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("task") != "arithmetic" or r.get("split") != split or r.get("gold") is None:
                continue
            lv = level_of(slot_difficulty(r))
            if lv is not None:
                by_level[lv].append(r)
    return by_level


# -- the pupil tries ------------------------------------------------------------------------

def sample_programs(model, tk, eos, prompt: str, k: int, temperature: float, max_new: int, dev,
                    top_k: int = 50) -> list[str]:
    """k programs for one prompt in one batch: the prompt is read once per row (the recurrent state is per
    row), then each row samples until </s>. temperature 0 = greedy (the gate's one try)."""
    ids = tk.encode(prompt).ids
    state = model.init_state(k, dev)
    out: list[list[int]] = [[] for _ in range(k)]
    with torch.no_grad():
        for t in ids[:-1]:
            _, state = model.step(torch.full((k,), t, dtype=torch.long, device=dev), state)
        cur = torch.full((k,), ids[-1], dtype=torch.long, device=dev)
        done = torch.zeros(k, dtype=torch.bool, device=dev)
        for _ in range(max_new):
            logits, state = model.step(cur, state)
            logits = logits.float()
            if temperature > 0:
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            for i in (~done).nonzero().flatten().tolist():
                if int(nxt[i]) != eos:
                    out[i].append(int(nxt[i]))
            done |= nxt == eos
            if bool(done.all()):
                break
            cur = nxt
    return [decode_program(tk, o, eos) for o in out]


class Judge:
    """The VM, and only the VM, says whether an attempt is right: slot check -> fill -> run -> compare to gold."""

    def __init__(self, fake: bool = False) -> None:
        self.fake = fake
        self.session = None
        self.runs = 0

    def _run(self, program: str):
        from build_emitter_sft import answer_fn, shim_isolver
        src = shim_isolver(program)
        if self.fake:                                    # smoke only: our interpreter stands in for the VM
            vals = arith_step_values(program)
            return (vals is not None), (vals[-1] if vals else None)
        if self.session is None:
            from cubbyllm.bridges.cubelang_client import CubelangSession
            self.session = CubelangSession()
        try:
            out = self.session.run(src, fn=answer_fn(src))
            return bool(out.get("ok")), out.get("result")
        except Exception:
            return False, None

    def attempt(self, rec: dict, slotted: str) -> dict:
        self.runs += 1
        table = table_of(rec["spans"], rec.get("question", ""))
        verdict = table.check(slotted)
        stated = [num_value(s.filled) for s in table.spans if s.kind == "N"]
        if not verdict.ok:
            return {"program": slotted, "outcome": REFUSED, "values": None, "stated": stated, "why": verdict.reason}
        hack = answer_written_in(slotted, rec.get("gold"), stated)
        if hack:                                          # the reward is the VM's, and it cannot be gamed this way
            return {"program": slotted, "outcome": REFUSED, "values": None, "stated": stated, "why": hack}
        filled = table.fill(slotted)
        ok, res = self._run(filled)
        v, g = num_value(str(res)) if ok else None, num_value(str(rec["gold"]))
        good = v is not None and g is not None and abs(v - g) <= 1e-6 * max(1.0, abs(g))
        return {"program": slotted, "outcome": CORRECT if good else (WRONG if ok else REFUSED),
                "values": arith_step_values(filled), "stated": stated, "why": "" if ok else "does not run"}


# -- the update ------------------------------------------------------------------------------

def row_logprobs(model, tk, eos, pairs: list[tuple[str, str]], dev, amp: bool) -> torch.Tensor:
    """Length-normalized log-prob of each (prompt, program) -- the program's tokens and its </s> only."""
    from talk_data import make_batch
    rows = [{"p": tk.encode(p).ids, "t": tk.encode(t.rstrip() + "\n").ids + [eos]} for p, t in pairs]
    x, y, m, _ = make_batch(rows)
    x, y, sel = (torch.from_numpy(a).to(dev) for a in (x, y, m > 0))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h = model.features(x)
        logits = model.logits_from(h[sel].unsqueeze(0))[0]
    lp = -F.cross_entropy(logits.float(), y[sel], reduction="none")
    row = sel.nonzero()[:, 0]
    tot = torch.zeros(len(rows), device=dev).index_add_(0, row, lp)
    cnt = torch.zeros(len(rows), device=dev).index_add_(0, row, torch.ones_like(lp))
    return tot / cnt.clamp(min=1.0)


def update(model, tk, eos, rows: list[dict], opt, params, dev, amp: bool, beta: float = 2.0, micro: int = 8) -> dict:
    """One pass over a round's rows: weighted cross-entropy on the solves and shown solutions, a preference on
    the (wrong, right) pairs. Weights are the dopamine; rows that weigh 0 teach nothing."""
    sft = [r for r in rows if r["kind"] in ("solve", "shown") and r["weight"] > 0]
    pref = [r for r in rows if r["kind"] == "mistake" and r["weight"] != 0]
    stats = Counter(sft=len(sft), pref=len(pref))
    wsum = sum(r["weight"] for r in sft) + sum(abs(r["weight"]) for r in pref) or 1.0
    opt.zero_grad(set_to_none=True)
    for s in range(0, len(sft), micro):
        chunk = sft[s:s + micro]
        lp = row_logprobs(model, tk, eos, [(r["prompt"], r["program"]) for r in chunk], dev, amp)
        w = torch.tensor([r["weight"] for r in chunk], device=dev)
        loss = -(w * lp).sum() / wsum
        loss.backward()
        stats["sft_loss"] += float(loss)
    for s in range(0, len(pref), micro):
        chunk = pref[s:s + micro]
        lc = row_logprobs(model, tk, eos, [(r["prompt"], r["chosen"]) for r in chunk], dev, amp)
        lr = row_logprobs(model, tk, eos, [(r["prompt"], r["rejected"]) for r in chunk], dev, amp)
        w = torch.tensor([abs(r["weight"]) for r in chunk], device=dev)
        loss = (w * F.softplus(-beta * (lc - lr))).sum() / wsum
        loss.backward()
        stats["pref_loss"] += float(loss)
        stats["pref_margin"] += float((lc - lr).mean()) / max(1, math.ceil(len(pref) / micro))
    if stats["sft"] or stats["pref"]:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return dict(stats)


# -- the gate: one try, greedy, the same held-out slice every round ------------------------

def read_heldout(model, tk, eos, recs: list[dict], judge: Judge, dev, max_new: int) -> float:
    good = 0
    for r in recs:
        prog = sample_programs(model, tk, eos, r["prompt"], 1, 0.0, max_new, dev)[0]
        good += judge.attempt(r, prog)["outcome"] == CORRECT
    return good / max(1, len(recs))


def load_regression(paths: list[str], per_family: int) -> list[dict]:
    """The other families' val records (chain, plan, kernel, role-binding): what the school must not break."""
    from emitter_data import val_records
    out = []
    for p in paths:
        out += [r for r in val_records(p, per_family) if r["task"] != "arithmetic"]
    return out


def read_regression(model, tk, eos, recs: list[dict], dev, max_new: int) -> int:
    """How many of the other families' records the adapter still gets right, scored by H-E18's own rule
    (`exp_he18_ab.score`: plan by its binds, the rest by the VM's value or by running)."""
    if not recs:
        return 0
    import exp_he18_ab as ab
    outs = []
    for r in recs:
        gen = sample_programs(model, tk, eos, r["prompt"], 1, 0.0, max_new, dev)[0]
        table = table_of(r["spans"], r.get("question", ""))
        v = table.check(gen)
        outs.append({"task": r["task"], "gold": r.get("gold"), "reference": r.get("reference", ""),
                     "generated": table.fill(gen), "slots_ok": v.ok, "slot_reason": v.reason, "subtype": ""})
    s = ab.score(outs)["_all"]
    return round(s["correct"] * s["n"])


def snapshot(params) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore(params, snap) -> None:
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


# -- the school day -------------------------------------------------------------------------------

def run_round(model, tk, eos, dev, amp, cur: Curriculum, pool: dict, judge: Judge, world: ArithmeticWorld,
              rng: random.Random, n: int, temperature: float, max_new: int) -> tuple[list[dict], dict]:
    rows, st = [], Counter()
    for level in cur.plan_batch(n):
        if not pool.get(level):
            continue
        rec = rng.choice(pool[level])
        budget = cur.budget(level)
        progs = sample_programs(model, tk, eos, rec["prompt"], budget, temperature, max_new, dev)
        attempts = []
        for p in progs:                                   # in sampling order; the first correct ends the count
            a = judge.attempt(rec, p)
            attempts.append(a)
            if a["outcome"] == CORRECT:
                break
        result = cur.record(level, [a["outcome"] for a in attempts])
        ref_filled = table_of(rec["spans"]).fill(rec["program"])
        reference = {"program": rec["program"], "values": arith_step_values(ref_filled)}
        new = mistake_rows(rec["prompt"], attempts, result, reference=reference)
        for r in new:
            if r["kind"] == "mistake":
                cur.learn(level, r["error"])
                st[f"error:{r['error']}"] += 1
        rows += new
        if result["solved"] and "$N" in attempts[-1]["program"]:     # a procedure transfers through its slots
            world.remember(rec["prompt"], attempts[-1]["program"], attested=True, source="curriculum")
        st["problems"] += 1
        st["solved"] += int(result["solved"])
        st["tries"] += result["tries"]
        st["dopamine"] += result["dopamine"]
        st[f"level{level}:n"] += 1
        st[f"level{level}:solved"] += int(result["solved"])
        if result["event"]:
            log(f"    level {level} ({cur.levels[level].name}): {result['event']}")
    st["dopamine"] = round(st["dopamine"] / max(1, st["problems"]), 4)
    return rows, dict(st)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", default="", help="the emitter adapter to start from (the SFT arm)")
    ap.add_argument("--pool", nargs="+", required=True, help="slot-form jsonl(s): the problems (train split)")
    ap.add_argument("--heldout", nargs="+", required=True, help="slot-form jsonl(s): the gate's problems (val split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--problems", type=int, default=256, help="problems per round")
    ap.add_argument("--max-tries", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gate-n", type=int, default=128, help="held-out problems read at one try each round")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fake-vm", action="store_true", help="smoke only: our interpreter judges instead of the VM")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="the ordering control: every level open from the start, drawn uniformly")
    ap.add_argument("--regress", nargs="*", default=[], help="slot jsonl(s) whose other families' val records are "
                                                             "the regression suite (catastrophic forgetting)")
    ap.add_argument("--regress-n", type=int, default=8, help="records per family in the regression suite")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    amp = dev.type == "cuda"

    from tokenizers import Tokenizer
    from train_talk_torch import apply_lora, load_adapter, save_adapter
    from talk_data import lora_targets
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    if a.smoke:
        from train_base import arch_meta, build_model
        meta = arch_meta(tk.get_vocab_size(), ((tk.get_vocab_size() + 127) // 128) * 128)
        model = build_model(meta, dev)
        a.rounds, a.problems, a.max_tries, a.max_new, a.gate_n, a.window = 2, 6, 2, 24, 3, 4
    else:
        from train_base import load_base
        model = load_base(a.ckpt, str(dev))
        meta = torch.load(a.ckpt, map_location="cpu", weights_only=False)["meta"]
    for p in model.parameters():
        p.requires_grad_(False)
    kept = os.path.join(a.out, f"{ADAPTER}.safetensors")
    start = a.out if os.path.exists(kept) else a.adapter        # a restarted school resumes from its last kept round
    if start:
        ameta = load_adapter(model, start, ADAPTER)
        targets, rank, alpha = ameta["targets"], ameta["rank"], ameta["alpha"]
        from train_emitter_torch import _find
        wrapped = {t: _find(model, t) for t in targets}
    else:
        targets, rank, alpha = lora_targets(meta["L"], meta["attn_every"]), 16, 32.0
        wrapped = apply_lora(model, targets, rank, alpha)
    params = [p for w in wrapped.values() for p in (w.lora_A.weight, w.lora_B.weight)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    os.makedirs(a.out, exist_ok=True)
    school_path, world_path = os.path.join(a.out, "school.json"), os.path.join(a.out, "arith_world.jsonl")
    cur = Curriculum.load(school_path, window=a.window, open_all=a.no_curriculum) if os.path.exists(school_path) else \
        Curriculum(max_tries=a.max_tries, window=a.window, seed=a.seed, open_all=a.no_curriculum)
    world = ArithmeticWorld.load(world_path) if os.path.exists(world_path) else ArithmeticWorld()
    pool = load_pool(a.pool, "train")
    held = [r for lv in load_pool(a.heldout, "val").values() for r in lv]
    random.Random(1).shuffle(held)
    held = held[:a.gate_n]
    judge = Judge(fake=a.fake_vm)
    log(f"school | {dev} | pool by level {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(pool.items()))}}} | "
        f"held-out {len(held)} | level {cur.current} ({cur.levels[cur.current].name}) | budget {cur.budget(cur.current)}")
    regress = load_regression(a.regress, a.regress_n if not a.smoke else 1)
    best = read_heldout(model, tk, eos, held, judge, dev, a.max_new)
    reg0 = read_regression(model, tk, eos, regress, dev, a.max_new)
    best_snap = snapshot(params)
    log(f"  round 0: held-out one-try {best:.3f} | regression suite {reg0}/{len(regress)} (other families)")
    lf = open(os.path.join(a.out, "school_rounds.jsonl"), "a", encoding="utf-8")
    for rnd in range(1, a.rounds + 1):
        t0 = time.time()
        rows, st = run_round(model, tk, eos, dev, amp, cur, pool, judge, world, rng, a.problems, a.temperature, a.max_new)
        up = update(model, tk, eos, rows, opt, params, dev, amp, beta=a.beta)
        acc = read_heldout(model, tk, eos, held, judge, dev, a.max_new)
        reg = read_regression(model, tk, eos, regress, dev, a.max_new)
        kept = acc >= best and reg >= reg0 - 1          # better at arithmetic, and nothing else forgotten
        if kept:
            best, best_snap = acc, snapshot(params)
            save_adapter(wrapped, a.out, {"targets": targets, "rank": rank, "alpha": alpha, "adapter": ADAPTER,
                                          "school_round": rnd, "heldout_one_try": acc}, ADAPTER)
        else:
            restore(params, best_snap)                     # never ship a regression
        cur.save(school_path)
        world.save(world_path)
        row = {"round": rnd, **st, **{f"update_{k}": v for k, v in up.items()}, "heldout": acc, "kept": kept,
               "regression": reg, "regression_n": len(regress),
               "best": best, "level": cur.current, "budget": cur.budget(cur.current),
               "procedures": len(world.procedures), "wall_s": round(time.time() - t0, 1)}
        lf.write(json.dumps(row) + "\n"); lf.flush()
        log(f"  round {rnd}: solved {st.get('solved', 0)}/{st.get('problems', 0)} in {st.get('tries', 0)} tries, "
            f"dopamine {st.get('dopamine', 0):+.3f} | rows {up.get('sft', 0)} sft + {up.get('pref', 0)} pairs | "
            f"held-out {acc:.3f}, regression {reg}/{len(regress)} ({'kept' if kept else 'reverted'}, best {best:.3f}) | "
            f"level {cur.current} "
            f"({cur.levels[cur.current].name}) budget {cur.budget(cur.current)} | {row['wall_s']:.0f}s")
    lf.close()
    with open(os.path.join(a.out, "school.log"), "a", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()