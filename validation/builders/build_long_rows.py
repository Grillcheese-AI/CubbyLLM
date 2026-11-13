"""Order 0067, the second reader's step 3 (Nick, 2026-11-13: "go ahead with step 2" ... step 3 next): long instructions
built in reverse, each with its split and its join, run before kept.

A row: 2-4 parts (``train`` / ``dev`` / ``test``) or 5-8 (``test_long``, line 4: trained on up to 4, read on 5-8). Parts
are atoms of `view_programs` (or their negation), drawn one by one from the replies that still pass the parts drawn so
far, so the joined instruction both holds and fails on its replies. Groups join them: one part, "A unless B", "if B,
A", "either A or B" (`view_join`). Mixed in: a task sentence (Nick's task words), phrases no measurement covers
(``asked``; held-out phrases only in held-out rows), a process step, sentences that are not instructions. English or
French, by hand-written templates (`view_wording`, `view_long_pools`).

Targets: ``segments`` (every sentence or clause with its character span and kind: check (its part), task, asked,
process, none) and ``join`` (the program the reader writes). Kept when the VM's verdict on the joined program is the
evaluator's on all 8 replies (half pass, half fail), no reply's views too long for the VM and no two strings sharing
the VM's name key; a disagreement stops the build. Train and dev rows sharing 5 words with IFEval or IFBench are
dropped.

  python validation/builders/build_long_rows.py [--train 3000] [--held 300] [--seed 0] [--out $CUBBY_DATASETS/long_rows_0067]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import bench_overlap as BO  # noqa: E402
import real_rows  # noqa: E402
import view_compile as VC  # noqa: E402
import view_join as VJ  # noqa: E402
import view_long_pools as P  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VR  # noqa: E402
import view_wording as W  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity, vm_names  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402

__wiring__ = "STANDALONE"


def positive(p: dict) -> dict | None:
    """A negated atom as the atom it equals, when there is one (so `held_out` sees "no word X" as forbidden words)."""
    a = p["not"]
    if a["q"] in ("count", "n"):
        return dict(a, op=W.INV[a["op"]])
    if a["q"] in ("any", "none"):
        return dict(a, q="none" if a["q"] == "any" else "any")
    return None


def draw_parts(rng, views, k) -> list[dict] | None:
    passing, parts = list(range(len(views))), []
    for _ in range(k):
        best = None
        for _ in range(6):
            a = VP.sample_atom(rng, [views[j] for j in passing])
            if a is None:
                continue
            for cand in (a, {"not": a}):
                eq = positive(cand) if "not" in cand else cand
                if eq is not None and VP.held_out(eq):
                    continue
                share = sum(VP.evaluate(cand, views[j]) for j in passing) / len(passing)
                if share < 0.95 and (best is None or share > best[0]):
                    best = (share, cand)
        if best is None:
            return None
        parts.append(best[1])
        passing = [j for j in passing if VP.evaluate(best[1], views[j])]
        if len(passing) < 4:
            return None
    return parts


def groups_for(rng, parts) -> list[tuple]:
    out, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and rng.random() < 0.3:
            kind = rng.choice(["unless", "if", "either"])
            plain = [x for x in (i + 1, i) if "not" not in parts[x]]    # a condition is said of a plain atom
            if kind == "either" or not plain:
                out.append(("either", i, i + 1))
            else:
                c = plain[0]
                other = i if c == i + 1 else i + 1
                out.append(("unless", other, c) if kind == "unless" else ("if", c, other))
            i += 2
        else:
            out.append(("one", i))
            i += 1
    return out


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:]


def _imp(p, lang, rng) -> str:
    return W.say_not(p["not"], lang, rng) if "not" in p else W.say(p, lang, rng)


def compose(rng, parts, groups, lang, phrases) -> tuple[str, list[dict], list[int]]:
    """(instruction, segments, the order the text states the parts in)."""
    pieces = []                                           # (text, segment) ; a group may give two segments
    order = []
    for g in groups:
        if g[0] == "one":
            pieces.append([(_imp(parts[g[1]], lang, rng), {"kind": "check", "part": g[1]})])
            order.append(g[1])
        elif g[0] == "if":
            c, a = g[1], g[2]
            head = ("If " if lang == "en" else "Si ") + W.say_cond(parts[c], lang) + ", "
            pieces.append([(head, {"kind": "check", "part": c, "role": "condition"}),
                           (_lower1(_imp(parts[a], lang, rng)), {"kind": "check", "part": a})])
            order += [c, a]
        else:
            a, b = g[1], g[2]
            first = _imp(parts[a], lang, rng).rstrip(".")
            if g[0] == "unless":
                tail = (", unless " if lang == "en" else ", sauf si ") + W.say_cond(parts[b], lang) + "."
                pieces.append([(first, {"kind": "check", "part": a}), (tail, {"kind": "check", "part": b, "role": "condition"})])
            else:
                second = _lower1(_imp(parts[b], lang, rng))
                pieces.append([(("Either " if lang == "en" else "Soit ") + _lower1(first), {"kind": "check", "part": a}),
                               ((", or " if lang == "en" else ", soit ") + second, {"kind": "check", "part": b})])
            order += [a, b]
    others = []
    if rng.random() < 0.8:
        s, key = P.task_sentence(rng, lang)
        others.append(("task", [(s, {"kind": "task", "task": key})]))
    for _ in range(rng.choice([0, 1, 1, 2])):
        key, w = rng.choice(phrases[lang])
        others.append(("asked", [(w, {"kind": "asked", "asked": key})]))
    if rng.random() < 0.4:
        others.append(("process", [(rng.choice(P.PROCESS[lang]), {"kind": "process"})]))
    for _ in range(rng.choice([0, 1, 1, 2])):
        others.append(("none", [(rng.choice(P.CONTEXT[lang]), {"kind": "none"})]))
    seq = [("check", p) for p in pieces]
    for kind, p in others:                                # a task mostly first, the rest anywhere
        at = 0 if kind == "task" and rng.random() < 0.7 else rng.randrange(len(seq) + 1)
        seq.insert(at, (kind, p))
    text, segs = "", []
    for n, (_, p) in enumerate(seq):
        if n:
            text += rng.choice([" ", " ", "\n"])
        for t, seg in p:
            segs.append(dict(seg, start=len(text), end=len(text) + len(t)))
            text += t
    return text, segs, order


def renumber(parts, groups, segs, order):
    """Number the parts in the order the text states them."""
    new = {old: i for i, old in enumerate(order)}
    parts2 = [parts[old] for old in order]
    groups2 = [(g[0], *[new[x] for x in g[1:]]) for g in groups]
    return parts2, groups2, [dict(s, part=new[s["part"]]) if "part" in s else s for s in segs]


def build_one(rng, vm, views, replies, k, lang, phrases, bench, split, stop_dir):
    parts = draw_parts(rng, views, k)
    if parts is None:
        return None, "no parts drawn"
    groups = groups_for(rng, parts)
    text, segs, order = compose(rng, parts, groups, lang, phrases)
    parts, groups, segs = renumber(parts, groups, segs, order)
    if split in ("train", "dev") and BO.overlaps(text, bench):
        return None, "shares 5 words with a benchmark"
    join, spec = VJ.join(parts, groups), VJ.spec(parts, groups)
    used = sorted({v for p in parts for v in VC.views_of(p)})
    cand = [(t, VP.evaluate(spec, v)) for t, v in zip(replies, views)
            if all(len(v[u]) <= RV.MAX_ROWS for u in used) and not vm_names.collisions(VJ.program(join, parts, t))]
    yes, no = [c for c in cand if c[1]], [c for c in cand if not c[1]]
    if len(yes) < 2 or len(no) < 2:
        return None, "too few usable replies on one side"
    chosen = rng.sample(yes, min(4, len(yes))) + rng.sample(no, min(len(no), 8 - min(4, len(yes))))
    for t, want in chosen:
        got = str(vm.run(VJ.program(join, parts, t), fn="solve").get("result")).strip() in ("1", "1.0")
        if got != want:
            (stop_dir / "_disagreement.json").write_text(json.dumps({"parts": parts, "groups": groups, "reply": t,
                                                                     "evaluator": want, "vm": got}, ensure_ascii=False, indent=1), encoding="utf-8")
            raise SystemExit(f"the VM disagrees with the evaluator on a joined program; kept in {stop_dir}/_disagreement.json")
    return {"split": split, "lang": lang, "n_parts": len(parts), "instruction": text, "segments": segs, "parts": parts,
            "groups": [list(g) for g in groups], "join": join, "replies": [{"text": t, "held": h} for t, h in chosen]}, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--held", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or Path(local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067")
    out.mkdir(parents=True, exist_ok=True)
    vm, bench = cc.CubelangSession(), BO.benchmark_grams(ifbench=True)
    train_ph, held_ph = real_rows._phrases(9)             # round 6's split of the phrases (held-out keys stay held out)
    plan = [("train", a.train, (2, 4), train_ph), ("dev", a.held, (2, 4), train_ph), ("test", a.held, (2, 4), held_ph),
            ("test_long", a.held, (5, 8), held_ph)]
    rows, why = [], Counter()
    for n_split, (split, n, (lo, hi), phrases) in enumerate(plan):
        rng = random.Random(a.seed * 100 + n_split)        # each split its own stream: held-out compositions are unseen
        kept, replies, views = 0, [], []
        while kept < n:
            if kept % 5 == 0 or not replies:
                replies = VR.pool(rng, 160 if hi > 4 else 80)
                views = [RV.views(t) for t in replies]
            row, reason = build_one(rng, vm, views, replies, rng.randint(lo, hi), "fr" if rng.random() < 0.4 else "en",
                                    phrases, bench, split, out)
            if row is None:
                why[f"{split}: {reason}"] += 1
                continue
            row["id"] = f"l{len(rows):06d}"
            rows.append(row)
            kept += 1
        print(f"  {split}: {kept}", flush=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"order": "0067", "what": "the second reader's step 3: long instructions with their split and join, run before kept",
           "seed": a.seed, "rows": len(rows), "by_split": dict(Counter(r["split"] for r in rows)),
           "by_lang": dict(Counter(r["lang"] for r in rows)), "by_parts": dict(sorted(Counter(r["n_parts"] for r in rows).items())),
           "segments_by_kind": dict(Counter(s["kind"] for r in rows for s in r["segments"])),
           "groups": dict(Counter(g[0] for r in rows for g in r["groups"])), "drop_reasons": dict(why),
           "vm_runs_agreeing": sum(len(r["replies"]) for r in rows), "vm_disagreements": 0,
           "benchmarks_checked": sorted(bench), "vm": vm_identity.identity(),
           "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "vm"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
