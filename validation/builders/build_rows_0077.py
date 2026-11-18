"""Order 0077, build 2: the description reader's rows around real prompts (`long_rows_0077`, off-repo; counts and
SHA-256 in the repo). 0074's builder (`build_rows_0073.finish`: parts drawn, worded, their folded said forms joined,
every reply kept run in the VM against the evaluator), with the task a real prompt (`real_prompts_0077`):
- each row draws 2 to 8 checks, each length as often within a set, on 0074's pools (80 replies for 2 to 4 parts, 160
  for 5 to 8; held-out rows also search ``--reserve`` more for the replies that can tell);
- the composed checks (with any asked, process or context sentences from the lists: training phrases in train and
  dev, the held-out phrases in the test sets) go after the prompt, before it, or after its first sentence;
- each line of the prompt is one ``task`` segment, so the reader copies the prompt as it stands, line by line;
- each prompt is used once; a held-out row whose folded parts and groups are a train row's (0074's train set and
  train_real) is drawn again.

  python validation/builders/build_rows_0077.py [--seed 77] [--train 2000] [--dev 300] [--test 300] [--reserve 640]
      [--out $CUBBY_DATASETS/long_rows_0077]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import bench_overlap as BO  # noqa: E402
import build_rows_0073 as B  # noqa: E402
import build_rows_0074 as B4  # noqa: E402
import real_prompts_0077 as RP  # noqa: E402
import real_rows  # noqa: E402
import rows_0073_stats as ST  # noqa: E402
import view_draw as D  # noqa: E402
import view_replies as VR  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402

__wiring__ = "STANDALONE"

SETS = ("train_real", "dev_real", "test_real", "test_dolly")
AS = {"train_real": "train", "dev_real": "dev", "test_real": "test", "test_dolly": "test"}   # finish's checks
PLACES = ("after", "before", "around")
TRIES = 40


def _first(prompt: str):
    return re.match(r"(.+?[.!?])([ \t]+|\n+)(\S.*)", prompt, re.S)


def task_segments(text: str, at: int) -> list[dict]:
    """One ``task`` segment a non-empty line of ``text``, placed at ``at``."""
    out, k = [], 0
    for line in text.split("\n"):
        if line.strip():
            out.append({"kind": "task", "task": "real", "start": at + k, "end": at + k + len(line)})
        k += len(line) + 1
    return out


def place(rng, prompt: str, text: str, segs: list, where: str) -> tuple[str, list]:
    """The composed ``text`` (its ``segs``) put ``where`` in the prompt."""
    sep = rng.choice([" ", "\n", "\n\n"])
    if where == "around":
        m = _first(prompt)
        head, tail = m.group(1), m.group(3)
        sep2 = rng.choice([" ", "\n"]) if "\n" not in m.group(2) else m.group(2)
        pieces = [(head, True), (sep, None), (text, False), (sep2, None), (tail, True)]
    elif where == "before":
        pieces = [(text, False), (sep, None), (prompt, True)]
    else:
        pieces = [(prompt, True), (sep, None), (text, False)]
    out, new = "", []
    for s, is_prompt in pieces:
        if is_prompt:
            new += task_segments(s, len(out))
        elif is_prompt is False:
            new += [dict(g, start=g["start"] + len(out), end=g["end"] + len(out)) for g in segs]
        out += s
    return out, sorted(new, key=lambda g: g["start"])


def build_set(name, prompts, want, a, vm, bench, phrases, train_keys, why, stop_dir) -> list[dict]:
    rng, tally, seen, rows = random.Random(a.seed * 100 + SETS.index(name)), Counter(), Counter(), []
    pools, made = {}, 0
    for n, (sid, lang, prompt) in enumerate(prompts):
        if len(rows) == want:
            break
        k = 2 + len(rows) % 7
        for t in range(TRIES):
            size = B4.pool_size(k)
            if size not in pools or made - pools[size][2] >= 5 or t % 10 == 9:
                replies = VR.pool(rng, size)
                reserve = [(x, RV.views(x)) for x in VR.pool(rng, a.reserve)] if name != "train_real" else []
                pools[size] = (replies, [RV.views(x) for x in replies], made, reserve)
            replies, views, _, reserve = pools[size]
            parts = D.draw_parts(rng, views, k, tally)
            if parts is None:
                why[f"{name}: no parts drawn"] += 1
                continue
            where = rng.choice([p for p in PLACES if p != "around" or _first(prompt)])
            wrap = lambda r, text, segs: place(r, prompt, text, segs, where)      # noqa: E731
            row, reason = B.finish(rng, vm, views, replies, parts, D.groups_for(rng, parts, seen=seen), lang,
                                   phrases, bench, AS[name], stop_dir, reserve=reserve, wrap=wrap)
            if row is not None and name != "train_real" and B.key(row) in train_keys:
                row, reason = None, "a train row's parts and groups"
            if row is None:
                why[f"{name}: {reason}"] += 1
                continue
            if a.size and name in ("train_real", "dev_real") and a.size(row) > a.max_len:
                why[f"{name}: a record over the trainer's {a.max_len} tokens, its prompt replaced"] += 1
                break                                    # the prompt, not the draw, is too long: the next prompt
            row.update(id=f"r{SETS.index(name)}{n:05d}", split=name, source=sid.split(":")[0], source_id=sid,
                       prompt_at=where, prompt_chars=len(prompt), prompt_lines=sum(1 for x in prompt.split("\n") if x.strip()))
            D.tally_row(tally, [s["w"] for s in row["said"]])
            seen[(row["n_parts"], D.placement(row["groups"]))] += 1
            if name == "train_real":
                train_keys.add(B.key(row))
            rows.append(row)
            made += 1
            break
        else:
            why[f"{name}: a prompt left out after {TRIES} draws"] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--reserve", type=int, default=640)
    ap.add_argument("--band", type=float, nargs=2, default=(0.45, 0.55))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default=None, help="train and dev prompts whose SFT record would pass --max-len "
                    "tokens are replaced by the next prompt (the trainer would skip the record)")
    ap.add_argument("--max-len", type=int, default=1280)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    out = Path(a.out or ds / "long_rows_0077")
    out.mkdir(parents=True, exist_ok=True)
    vm, bench = cc.CubelangSession(), BO.benchmark_grams(ifbench=True)
    a.size = None
    if a.tokenizer:                                       # the record's tokens as the trainer makes them
        from tokenizers import Tokenizer
        sys.path.insert(0, str(ROOT / "validation" / "sft"))
        import key_tokens as K
        from cubbyllm.speech import view_split as V
        tk = Tokenizer.from_file(a.tokenizer)
        a.size = lambda r: len(tk.encode(V.prompt(r["instruction"])).ids) + len(K.weights(tk, V.target(r, "m"), 4.0))
    prompts, pcounts = RP.sets(a.seed, a.train, a.dev, a.test, bench)
    train_keys = {B.key(json.loads(ln)) for f in ("rows.jsonl", "pair_rows.jsonl")
                  for ln in open(ds / "long_rows_0074" / f, encoding="utf-8") if json.loads(ln)["split"] in ("train", "pairs")}
    train_ph, held_ph = real_rows._phrases(9)              # 0074's phrases: held-out keys only in held-out rows
    why, by, seconds = Counter(), {}, {}
    for name in SETS:
        t0 = time.time()
        want = {"train_real": a.train, "dev_real": a.dev}.get(name, a.test)
        by[name] = build_set(name, prompts[name], want, a, vm, bench, train_ph if name in ("train_real", "dev_real")
                             else held_ph, train_keys, why, out)
        seconds[name] = round(time.time() - t0, 1)
        print(f"  {name}: {len(by[name])} rows of {want} wanted, {seconds[name]} s", flush=True)
    rows = [r for s in SETS for r in by[s]]
    with open(out / "rows.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(ST.line(r) for r in rows)
    shares = {s: ST.bounds(by[s])["all"]["lower share"] for s in SETS}

    def count(s, f):
        return dict(sorted(Counter(f(r) for r in by[s]).items()))
    man = {"order": "0077", "what": "build 2: the description reader's rows with a real prompt as the task (2 to 8 "
           "checks inside it), on 0074's builder; each prompt line one task segment", "seed": a.seed,
           "train and dev records within the trainer's tokens": a.max_len if a.size else None,
           "prompts": pcounts, "rows": len(rows), "by_set": {s: len(by[s]) for s in SETS},
           "by_source_and_lang": {s: count(s, lambda r: f"{r['source']} {r['lang']}") for s in SETS},
           "by_parts": {s: count(s, lambda r: r["n_parts"]) for s in SETS},
           "by_prompt_place": {s: count(s, lambda r: r["prompt_at"]) for s in SETS},
           "by_prompt_length_100": {s: count(s, lambda r: r["prompt_chars"] // 100 * 100) for s in SETS},
           "prompt_lines": {s: count(s, lambda r: r["prompt_lines"]) for s in SETS},
           "lower bounds' share of the bounds": shares, "band": list(a.band),
           "sets": {s: ST.summary(by[s]) for s in SETS}, "drop_reasons": dict(sorted(why.items())), "seconds": seconds,
           "vm_runs_agreeing": sum(len(r["replies"]) for r in rows), "vm_disagreements": 0,
           "benchmarks_checked": sorted(bench), "vm": vm_identity.identity(),
           "sha256": {"rows.jsonl": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
                      "the lines of rows.jsonl by set": {s: ST.digest(by[s]) for s in SETS},
                      "the source ids by set": {s: hashlib.sha256("\n".join(r["source_id"] for r in by[s]).encode())
                                                .hexdigest() for s in SETS}}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("vm", "sets")}, indent=1, ensure_ascii=False))
    off = {s: v for s, v in shares.items() if by[s] and (v is None or not a.band[0] <= v <= a.band[1])}
    if off:
        raise SystemExit(f"the lower bounds' share is outside {a.band[0]} to {a.band[1]}: {off}; the set is not to be used")


if __name__ == "__main__":
    main()
