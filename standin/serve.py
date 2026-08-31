"""serve — the stand-in serve loop, wired end to end.

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this).

One process, one turn at a time:

    question ──route──► TASK: plan+walk the fact store with the measured CoT
                        pipeline (cubbyllm.reasoning.answer — v3's training
                        prompts carried WALKED facts, so serve must too; flat
                        top-k is the fallback when the planner can't parse)
                        ──► Facts block ──► v3 emitter (CotChain opening
                        prefilled) ──► VM executes (answer_fn) ──► ground
                        check (the answer must be an object of an offered
                        fact) ──► spoken reply
              └───────► CHAT: CubbyChat.candidates (voice-filtered)
    …and EVERY spoken reply — task or chat — goes through CubbyTalk's ASK
    (`CubbyChat.mediate`): the VM offers the candidates and rejects any
    selection that was not offered. The don't-know line (user's language) is
    always a candidate and is what an ungrounded task answer degrades to.

Routing v0: if the best retrieval score for the question is below
`route_tau`, the turn is chat; otherwise it is a task. The hormonal state is
shared (CubbyChat owns it; `nudge` per turn) and reaches the emitter only
through chat turns — task programs are never modulated (facts, not tone).

`--selftest N` is the serve loop's OWN measured number: it takes N chain
questions from the val split, STRIPS their given Facts block, retrieves with
its own retriever, and reports how often the VM's answer matches gold — the
link (flat retrieval instead of the walked facts) that no prior eval covered.

  python standin/serve.py --gguf standin/models/emitter_v3.Q4_K_M.gguf              # REPL
  python standin/serve.py --gguf ... --selftest 25 --tag _v3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "standin"),
          os.path.join(ROOT, "standin", "data")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_emitter_sft import answer_fn, shim_isolver  # noqa: E402
from chat import RESTING, CubbyChat  # noqa: E402
from identity import EMITTER_SYSTEM, T, guess_lang, load_facts  # noqa: E402

from cubbyllm.reasoning import answer as pipeline_answer  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact  # noqa: E402

_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)

# every v3 chain target opens exactly like this; prefilling it pins the task
# style (without it, distractor-laden Facts blocks flip the emitter into the
# event/kernel program shapes — measured 0/25 on the first selftest pass) and
# must run through `create frame: number;` — cutting it at `{` leaves the model
# mid-line and it degenerates into flattened non-syntax (second 0/25 pass)
PROGRAM_PREFIX = ("use vsa;\n\nprogram CotChain implements ISolve {\n"
                  "    public function solve(mention: str): str {\n"
                  "        create frame: number;\n")


def _clean(gen: str) -> str:
    """Strip think blocks and fences, then cut from the first real program
    line — v3 sometimes echoes the prompt as a header, and an echoed Facts
    bullet is a parse error at top level. No program line -> ""."""
    s = _THINK_RE.sub("", gen, count=1) if "</think>" in gen else gen
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    lines = s.strip().splitlines()
    for i, l in enumerate(lines):
        if l.startswith(("use ", "program ")):
            return "\n".join(lines[i:]).strip() + "\n"
    return ""


class CubbyServe:
    """The assembled stand-in: retriever + emitter + VM + chat, one state."""

    # the measured verify/walk operating point (cot_harvest_v2:
    # validation/logs/exp_m3_cot_pipeline_v2.log); tau_vm=0.0 is degenerate —
    # the control role (sim ~0.05) then always reads as a violation and
    # every good walk gets banned
    TAU_VM = 0.22021484375
    TAU_RET = 0.595863088965416

    def __init__(self, emitter, retriever, store_texts: list[str], facts: dict | None = None,
                 exe: str | None = None, route_tau: float = 0.30, k_facts: int = 3,
                 tau_vm: float = TAU_VM, tau_ret: float = TAU_RET) -> None:
        self.emitter = emitter
        self.retriever = retriever                       # (query, k) -> [(score, fact_text)]
        self.store_texts = store_texts
        self.facts = facts or load_facts()
        self.exe = exe
        self.route_tau = float(route_tau)
        self.k_facts = int(k_facts)
        self.tau_vm = float(tau_vm)
        self.tau_ret = float(tau_ret)
        self.chat = CubbyChat(emitter, self.facts, exe=exe)

    # ── the task half ───────────────────────────────────────────────────────
    def gather_facts(self, question: str) -> list[tuple[float, str]]:
        """Top-k for the question plus ONE expansion hop through the best
        facts' objects (multi-hop questions need the next link, which does not
        share words with the question). Deduped, discovery order."""
        seen, out = set(), []

        def take(hits):
            for s, f in hits:
                key = " ".join(f.split())
                if key not in seen:
                    seen.add(key)
                    out.append((float(s), f))

        first = self.retriever(question, self.k_facts)
        take(first)
        for _, f in first[:2]:
            t = parse_fact(f)
            if t is not None:
                take(self.retriever(f"{t.obj} ", 2))
        # v3 trained on 1-3 walked facts; past ~4 the block is out of
        # distribution and the emitter derails (echoes the list, wrong style)
        return out[: self.k_facts + 1]

    def walk_facts(self, question: str) -> tuple[list[str], dict]:
        """Plan+walk through the measured CoT pipeline: the walked facts, in
        hop order, are what v3's chain prompts were trained on. Empty when
        the planner can't parse the question or the walk finds no path."""
        from cubbyllm.bridges import cubelang_client as cc

        def run_fn(source: str, fn: str) -> dict:
            return cc.run_program_proto(source, fn=fn, exe=self.exe)

        res = pipeline_answer(question, self.retriever, run_fn, tau_vm=self.tau_vm,
                              tau_ret=self.tau_ret, top_k=self.k_facts, max_repairs=3)
        facts = [h.fact for h in res.trace if h.fact]
        meta = {"walk_answer": res.answer, "walk_verified": res.verified,
                "walk_reason": res.reason, "repairs": res.repairs_used}
        return facts, meta

    def task_answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        facts, walk = self.walk_facts(question)
        scores: list[float] = []
        if not facts:                                    # planner miss -> flat retrieval fallback
            hits = self.gather_facts(question)
            facts, scores = [f for _, f in hits], [s for s, _ in hits]
        prompt = question + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts) if facts else question
        gen = self.emitter.emit(prompt, max_new_tokens=450, system=EMITTER_SYSTEM,
                                prefix=PROGRAM_PREFIX)
        raw = gen
        cleaned = _clean(gen)
        program = shim_isolver(cleaned) if cleaned else ""
        from cubbyllm.bridges import cubelang_client as cc
        if not program:
            vm_answer, vm_error = None, "no program emitted"
        else:
            try:
                out = cc.run_program_proto(program, fn=answer_fn(program), exe=self.exe)
                vm_answer = None if out.get("result") is None else str(out["result"])
                vm_error = None
            except cc.CubelangRunError as e:
                vm_answer, vm_error = None, str(e)[:200]
        # ground check: the VM's answer must be the object of an offered fact —
        # the emitter may only ever bind what retrieval put on the table.
        grounded = False
        if vm_answer is not None:
            objs = {normalize(t.obj) for t in (parse_fact(f) for f in facts) if t is not None}
            grounded = normalize(vm_answer) in objs
        # consistency gate: a VERIFIED walk that disagrees with the emitter is a
        # caught inconsistency — never speak the suspect answer (selftest: 5 of
        # the 6 emitter misses had a verified walk holding the gold answer)
        agrees = (walk["walk_answer"] is not None and vm_answer is not None
                  and normalize(vm_answer) == normalize(walk["walk_answer"]))
        speak_ok = grounded and (agrees or not walk["walk_verified"])
        return {"kind": "task", "question": question, "facts": facts, "scores": scores,
                "walk": walk, "program": program, "raw": raw, "vm_answer": vm_answer,
                "vm_error": vm_error, "grounded": grounded, "walk_agrees": agrees,
                "speak_ok": speak_ok, "wall_s": round(time.perf_counter() - t0, 3)}

    # ── one turn, VM-mediated speech for both paths ─────────────────────────
    def turn(self, user_text: str, feedback: str | None = None) -> dict:
        self.chat.nudge(user_text)
        hits = self.retriever(user_text, 1)
        top = float(hits[0][0]) if hits else 0.0
        dont_know = T(self.facts, "dont_know_line", guess_lang(user_text))
        if top < self.route_tau:
            rec = self.chat.turn(user_text, feedback)
            rec.update({"kind": "chat", "route_score": top})
            return rec
        task = self.task_answer(user_text)
        offered = ([task["vm_answer"], dont_know] if task["speak_ok"] else [dont_know])
        rec = self.chat.mediate(user_text, offered, rejected=[], feedback=feedback)
        rec.update({"kind": "task", "route_score": top, "task": task})
        return rec


# ── assembly from the real artifacts ────────────────────────────────────────
def build_serve(gguf: str, table: str | None, n_store: int, exe: str | None,
                route_tau: float, n_gpu_layers: int,
                extra_facts: list[str] | None = None) -> CubbyServe:
    from exp_m3_cot_pipeline import V4_TABLE, load_sample, make_retriever
    from exp_m3_domain_routing import _load_semantic_words

    from standin.emitter import LlamaCppEmitter
    print(f"loading fact store ({n_store} sampled questions' fact pool) ...", flush=True)
    _q, _a, _h, _chains, store = load_sample(n_store, seed=0)
    if extra_facts:                                      # selftest: the walked facts must BE in the store
        seen = {" ".join(f.split()) for f in store}
        store = store + [f for f in extra_facts if " ".join(f.split()) not in seen]
    print(f"  {len(store)} facts | loading fastword table ...", flush=True)
    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(str(table or V4_TABLE))
    retr = make_retriever(store, enc)
    print(f"loading emitter {gguf} (n_gpu_layers={n_gpu_layers}) ...", flush=True)
    emitter = LlamaCppEmitter(gguf, n_ctx=2048, n_gpu_layers=n_gpu_layers)
    return CubbyServe(emitter, retr, store, exe=exe, route_tau=route_tau)


def load_val_chains(n: int) -> list[dict]:
    val_path = os.path.join(ROOT, "standin", "data", "out", "emitter_sft.jsonl")
    chains = [json.loads(l) for l in open(val_path, encoding="utf-8")]
    return [r for r in chains if r["task"] == "chain" and r["split"] == "val"][:n]


def given_facts(r: dict) -> list[str]:
    if "\nFacts:" not in r["prompt"]:
        return []
    return [l[2:] for l in r["prompt"].split("\nFacts:\n", 1)[1].splitlines() if l.startswith("- ")]


def selftest(serve: CubbyServe, chains: list[dict], tag: str) -> dict:
    """Re-answer val chain questions WITHOUT their given facts: retrieval's
    own read. Reports gold-match and whether retrieval recovered the walked
    facts (the failure split: retrieval miss vs emitter miss)."""
    rows, hit, ret_ok = [], 0, 0
    spoken = {"correct": 0, "dont_know": 0, "wrong": 0}
    for i, r in enumerate(chains, 1):
        question = r["prompt"].split("\nFacts:")[0].strip()
        given = given_facts(r)
        out = serve.task_answer(question)
        gold = r.get("gold")
        ok = out["vm_answer"] is not None and gold is not None and normalize(out["vm_answer"]) == normalize(str(gold))
        recovered = all(any(" ".join(g.split()) == " ".join(f.split()) for f in out["facts"]) for g in given) if given else None
        # what the TURN would actually say: the answer if the consistency gate
        # passes, the don't-know line otherwise
        say = "correct" if (out["speak_ok"] and ok) else ("wrong" if out["speak_ok"] else "dont_know")
        spoken[say] += 1
        hit += int(ok)
        ret_ok += int(bool(recovered))
        rows.append({"question": question, "gold": gold, "vm_answer": out["vm_answer"], "gold_match": ok,
                     "grounded": out["grounded"], "speak_ok": out["speak_ok"], "spoken": say,
                     "retrieval_recovered_walked_facts": recovered,
                     "facts": out["facts"], "vm_error": out["vm_error"], "walk": out["walk"],
                     "program": out["program"], "raw": out["raw"]})
        print(f"  {i}/{len(chains)} gold_match={ok} spoken={say} retrieval_recovered={recovered} ({out['wall_s']}s)", flush=True)
    n = max(1, len(chains))
    summary = {"n": len(chains), "gold_match": hit / n,
               "retrieval_recovered_walked_facts": ret_ok / n,
               "spoken": {k: v / n for k, v in spoken.items()}}
    print(f"\n[stand-in] serve selftest: {summary}")
    out_path = os.path.join(ROOT, "standin", "data", "out", f"serve_selftest{tag}.json")
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-store", type=int, default=2000, help="questions sampled to build the fact store")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--route-tau", type=float, default=0.30)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    chains = load_val_chains(args.selftest) if args.selftest else []
    extra = [f for r in chains for f in given_facts(r)]
    serve = build_serve(args.gguf, args.table, args.n_store, None, args.route_tau, args.n_gpu_layers,
                        extra_facts=extra)
    if args.selftest:
        selftest(serve, chains, args.tag)
        return
    print("\nCubby [stand-in] — Ctrl+C to quit. Every reply is VM-mediated.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        rec = serve.turn(text)
        extra = f"  [task, grounded={rec['task']['grounded']}]" if rec["kind"] == "task" else f"  [chat, {rec['register']}]"
        print(f"cubby> {rec['reply']}{extra}")


if __name__ == "__main__":
    main()
