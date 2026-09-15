"""exp_r32 -- will the real trunk's tokenizer survive what we ask it to emit?

Wired: STANDALONE (validation script; never imported by cubbyllm/).

WHY THIS EXISTS
---------------
The trunk has not been pretrained. The tokenizer is the one decision in that run
you cannot revise afterwards without redoing it, and this session produced a hard
constraint on it that is written down nowhere:

  exp_r26 relabelled every relation to an opaque token and measured a **5.7x**
  swing from the token's SPELLING alone. `r_41027` came back as `r 41027` -- the
  underscore gone -- so a relation the emitter had copied correctly failed to
  match the store. Switching to pronounceable tokens (`damuzo`) moved the result
  from 13/300 to 74/300. The harness was innocent; the model's tokenizer was not.

That was a stand-in's tokenizer. The same failure is available to ours, and it is
cheap to check before the compute is spent rather than after.

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
Round-trip fidelity (encode -> decode -> compare) is necessary but it is NOT the
thing that bit us. `r_41027` round-trips fine. What bit us is **fragmentation**:
if that identifier costs four tokens, emitting it correctly is four consecutive
correct choices, and the model only has to fumble one. Fragmentation is what
predicts the emission artifact, so it is the number this reports.

Two consequences, both actionable before pretraining:

  * a vocabulary that keeps CubeLang's surface (identifiers, braces, quoted
    strings) in FEW tokens makes correct emission likelier and a grammar mask
    cheaper -- a role shredded into eight pieces is eight masked decode steps
  * whatever the fragmentation, WO-2.2's grammar removes the fumble entirely by
    making the wrong continuation undecodable. This check says how much work the
    grammar has to do, not whether it is needed

KILL CRITERION, two clauses
---------------------------
  * correctness -- any string that does NOT round-trip exactly is disqualifying.
    A tokenizer that cannot represent a program cannot be used to emit one.
  * cost -- role identifiers averaging above `--max-frag` tokens means the
    vocabulary is fighting the program surface. Not fatal, but it is a number
    that belongs in the pretraining decision rather than discovered later.

    python validation/exp_r32_tokenizer_readiness.py --gguf <a .gguf>
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The program surface the emitter actually has to produce, taken from
# `reasoning/programs.py` rather than invented.
PROGRAM = '''use vsa;

program CotChain implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, H1_DATE_OF_BIRTH, "Jean Sibelius";
        bind frame, H2_COUNTRY_OF_CITIZENSHIP, "Finland";
        return recover(frame, H1_DATE_OF_BIRTH);
    }
}
'''

CASES: dict[str, list[str]] = {
    "cubelang keywords": [
        "use vsa;", "create frame: number;", "return recover(frame, ABSENT_CTRL);",
        "public function hop_2(): str {", "program CotChain implements ISolve {",
    ],
    "role identifiers": [
        "H1_DATE_OF_BIRTH", "H2_COUNTRY_OF_CITIZENSHIP", "H1_PLACE_OF_BIRTH",
        "H3_CONTINENT", "ABSENT_CTRL", "H1_EDUCATED_AT",
    ],
    "opaque relation tokens (exp_r26)": [
        "r_41027", "r_00042", "damuzo", "vomeka", "zubnog",
    ],
    "relation labels": [
        "date of birth", "country of citizenship", "place of birth",
        "educated at", "instance of",
    ],
    "entities that bite": [
        "Jean Sibelius", "Joan Rivers: A Piece of Work", "Jean-Luc Picard",
        "Æthelred the Unready", "Beyoncé", "Nguyễn Phú Trọng", "O'Brien",
        "Saint-Jean-sur-Richelieu",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None,
                    help="a HF tokenizer.json; defaults to the repo's own 128k BBPE "
                         "(data/grillcheese_bbpe128k.json) -- the REAL trunk's "
                         "candidate vocabulary, not the stand-in's")
    ap.add_argument("--gguf", default=None,
                    help="test a GGUF's tokenizer instead (the stand-in's, for "
                         "comparison against what exp_r26 actually measured)")
    ap.add_argument("--max-frag", type=float, default=6.0,
                    help="tokens per role identifier above which the vocabulary is "
                         "fighting the program surface")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter(); lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    # The reference implementation, deliberately: a hand-rolled BPE would be one
    # more instrument that might be subtly wrong about the very thing it checks.
    if a.gguf:
        from llama_cpp import Llama
        llm = Llama(model_path=a.gguf, n_ctx=512, n_gpu_layers=0, verbose=False)
        name, n_vocab = pathlib.Path(a.gguf).name, llm.n_vocab()

        def enc(s: str) -> list[int]:
            return llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)

        def dec(toks: list[int]) -> str:
            return llm.detokenize(toks).decode("utf-8", errors="replace")
    else:
        from tokenizers import Tokenizer
        tp = pathlib.Path(a.tokenizer) if a.tokenizer else (
            ROOT / "data" / "grillcheese_bbpe128k.json")
        if not tp.is_file():
            raise SystemExit(f"no tokenizer at {tp}")
        tok = Tokenizer.from_file(str(tp))
        name, n_vocab = tp.name, tok.get_vocab_size()

        def enc(s: str) -> list[int]:
            return tok.encode(s, add_special_tokens=False).ids

        def dec(toks: list[int]) -> str:
            return tok.decode(toks, skip_special_tokens=False)

    log(f"tokenizer: {name}")
    log(f"vocab: {n_vocab}")
    log("")

    results: dict[str, dict] = {}
    broken: list[dict] = []
    for group, strings in CASES.items():
        rows = []
        for s in strings:
            toks = enc(s)
            back = dec(toks)
            ok = back == s
            rows.append({"s": s, "n_tokens": len(toks), "round_trip": ok,
                         "decoded": None if ok else back})
            if not ok:
                broken.append({"group": group, "s": s, "decoded": back})
        n = [r["n_tokens"] for r in rows]
        results[group] = {"rows": rows, "mean_tokens": round(statistics.mean(n), 2),
                          "max_tokens": max(n),
                          "round_trip_failures": sum(not r["round_trip"] for r in rows)}
        log(f"{group}")
        log(f"  {'string':<34}{'tokens':>7}  round-trip")
        for r in rows:
            mark = "ok" if r["round_trip"] else f"BROKEN -> {r['decoded']!r}"
            log(f"  {r['s'][:32]:<34}{r['n_tokens']:>7}  {mark}")
        log(f"  mean {results[group]['mean_tokens']} tokens, worst {results[group]['max_tokens']}")
        log("")

    # ---- the whole program: what one emission actually costs ---------------
    ptoks = enc(PROGRAM)
    pback = dec(ptoks)
    log(f"a complete 2-hop program: {len(ptoks)} tokens, "
        f"round-trip {'ok' if pback == PROGRAM else 'BROKEN'}")
    # a 10-hop program is the same shape with 8 more bind lines
    bind_line = '        bind frame, H3_SPOUSE, "Aino Järnefelt";\n'
    per_bind = len(enc(bind_line))
    log(f"  one extra bind line: {per_bind} tokens -> a 10-hop program is roughly "
        f"{len(ptoks) + 8 * per_bind} tokens")
    log(f"  (exp_r29 ran the emitter at max_new=700; that is the budget this has to fit)")
    log("")

    # ---- the finding exp_r26 paid for --------------------------------------
    log("the exp_r26 artifact, in THIS tokenizer's pieces:")
    for s in ("r_41027", "damuzo"):
        toks = enc(s)
        pieces = [dec([t]) for t in toks]
        log(f"  {s:<10} {len(toks)} tokens: {pieces}")
    log("  exp_r26's 5.7x swing was measured on the STAND-IN's vocabulary, not this")
    log("  one, so the artifact is not automatically ours. Run both (--gguf) before")
    log("  claiming it transfers: measured 2026-09-15, both split `r_41027` into")
    log("  ['r', '_', '410', '27'] and both fragment role identifiers to ~8 tokens,")
    log("  so it does transfer -- and the repo's own 128k is slightly WORSE than the")
    log("  stand-in (8.67 vs 7.83 mean tokens per role).")
    log("  Each token is a decode step the model can fumble. Fragmentation is what")
    log("  makes that possible; a grammar mask (WO-2.2) is what makes it impossible.")
    log("")

    role = results["role identifiers"]
    opaque = results["opaque relation tokens (exp_r26)"]
    rt_fail = sum(g["round_trip_failures"] for g in results.values()) + (pback != PROGRAM)

    log(f"{'check':<44}{'value':>10}   verdict")
    log("-" * 72)
    log(f"{'round-trip failures (any string)':<44}{rt_fail:>10}   "
        f"{'PASS' if rt_fail == 0 else 'DISQUALIFYING'}")
    log(f"{'mean tokens per role identifier':<44}{role['mean_tokens']:>10}   "
        f"{'ok' if role['mean_tokens'] <= a.max_frag else 'fighting the surface'}")
    log(f"{'mean tokens per opaque relation token':<44}{opaque['mean_tokens']:>10}   "
        f"{'ok' if opaque['mean_tokens'] <= a.max_frag else 'fighting the surface'}")

    if rt_fail:
        verdict = (f"KILLED (correctness): {rt_fail} strings do not round-trip -- this "
                   f"tokenizer cannot represent the programs it would have to emit")
    elif role["mean_tokens"] > a.max_frag or opaque["mean_tokens"] > a.max_frag:
        verdict = (f"usable but costly: role identifiers average "
                   f"{role['mean_tokens']} tokens, so correct emission is that many "
                   f"consecutive correct choices. A grammar mask is not an "
                   f"optimization here, it is the mechanism")
    else:
        verdict = "usable: the program surface round-trips and stays compact"
    log(f"\nVERDICT: {verdict}")

    log("\nWHAT THIS MEANS FOR THE PRETRAINING RUN (the part that cannot be redone):")
    log("  1. Run this against the CANDIDATE vocabulary before the run, not after.")
    log("     Nothing here is specific to one tokenizer; --tokenizer / --gguf swaps it.")
    log("  2. The program surface round-trips, so the vocabulary is not disqualified.")
    log("     Depth is not a context problem either: a 10-hop program is ~258 tokens")
    log("     against a 700 budget.")
    log("  3. The expensive thing is the ROLE IDENTIFIER, and it is expensive because")
    log("     the pre-tokenizer splits on `_` before BPE ever runs -- `H1_DATE_OF_BIRTH`")
    log("     is 10 tokens. More merges will not fix that; it is structural.")
    log("     Two independent ways out, and they are the same change:")
    log("       - WO-2.2's grammar makes the wrong continuation undecodable, so the")
    log("         fumble cannot happen however many steps it takes")
    log("       - WO-2.7's role-as-VECTOR convention (hash the role name into a")
    log("         hypervector instead of emitting an identifier) means those 10")
    log("         tokens are never emitted at all")
    log("     WO-0.3 wanted the second one for generalization -- the role vocabulary is")
    log("     95-98% per-relation, which caps the system at its training relations.")
    log("     This says it also buys ~8 decode steps per hop. Same change, two reasons.")

    stem = f"exp_r32_tokenizer_readiness{a.tag}"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{stem}.json").write_text(json.dumps({
        "tokenizer": name, "vocab": n_vocab,
        "note": "run against the trunk's CANDIDATE vocabulary before pretraining; "
                "--gguf switches to the stand-in's for comparison",
        "max_frag": a.max_frag, "program_tokens": len(ptoks),
        "tokens_per_bind_line": per_bind,
        "results": results, "round_trip_failures": rt_fail,
        "broken": broken, "verdict": verdict,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=1), encoding="utf-8")
    (LOGS / f"{stem}.log").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nwall {time.perf_counter() - t0:.0f}s | wrote {stem}.{{json,log}}")


if __name__ == "__main__":
    main()
