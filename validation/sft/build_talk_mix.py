"""The talk_v2 training set: every family the talk adapter learns from, sampled to a fixed mix, in one
jsonl the trainers read (`validation/talk_data.py`: prompt, answer, family, split).

    python standin/data/build_talk_mix.py [--out standin/data/out/talk_v2.jsonl] [--copy-to <drive dir>]

Train records are sampled per family to the counts in MIX (seeded); held records of every family are kept
in full, so each family's gate can be scored on entities, passages and speakers never trained on. The
counts are the owner's to move; the manifest records what was drawn from where.

Families and their builders (all under standin/data/):
  fact (ground_sft)        build_ground_sft.py   facts blocks from the wikikg graph: profile, relation, bind,
                                                 counter, absent
  passage / _absent        build_passage_sft.py  ChatQA passages (SQuAD, Quoref, TAT-QA, NarrativeQA)
  passage_web / _absent    build_webqa_sft.py    Nemotron-CC web documents with their Q/A
  quote / _absent          build_quotes_sft.py   who said it, over 3-4 quotes
  chat / chat_gated        build_chat_sft.py     Nemotron instruction-following chats, host guard only
  story_open / _gated      build_explicit_sft.py adult story continuations behind the explicit gate (H-E7)
  fact_* (v2.1)            build_hdc_sft.py      the same fact families from the hypernet scaling-law set,
                                                 plus 2-3 hop chains
  event_* (v2.1)           build_events_sft.py   years, regions and participants of historical events
  hist_* (v2.1)            build_history_sft.py  the history graph (build_history_graph.py over the history books,
                                                 build_nyt_events.py over the newspaper archive): causes, effects,
                                                 causal chains, what follows from an event, order in time

`--mix v2.1` (2026-09-25): talk_v2 held the fact families at 11% of its mix and lost bind, absent and profile
on the H-E6 gate; v2.1 brings the facts-block form back to about half of it -- ground_sft twice, the hdc and
event families -- and draws fewer passages and chats. `--mix v2.2` is v2.1 with the facts-block absent records
drawn up (hist_absent 15k, every hdc fact_absent), the one change H-E6's last missed bar asks for.
`--mix v2` rebuilds talk_v2 exactly.
"""
from __future__ import annotations

import argparse, collections, json, pathlib, random, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "standin" / "data" / "out"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

# (file, family) -> train records drawn; None = all of them
MIX = {
    ("ground_sft.jsonl", None): None,                    # every fact record (its families are kept as they are)
    ("passage_sft.jsonl", "passage"): 20000,
    ("passage_sft.jsonl", "passage_absent"): 12000,
    ("webqa_sft.jsonl", "passage_web"): 20000,
    ("webqa_sft.jsonl", "passage_web_absent"): 8000,
    ("quotes_sft.jsonl", "quote"): 8000,
    ("quotes_sft.jsonl", "quote_absent"): 3000,
    ("chat_sft.jsonl", "chat"): 25000,
    ("chat_sft.jsonl", "chat_gated"): None,              # every gate twin the chat set has
    ("explicit_sft.jsonl", "story_open"): None,          # H-E7's open side: behind `Explicit: allowed` only
    ("explicit_sft.jsonl", "story_gated"): None,         # ... and the same requests without the line
}

MIX_V21 = {
    ("ground_sft.jsonl", None): None,
    ("hdc_sft.jsonl", "fact_relation"): 8000,
    ("hdc_sft.jsonl", "fact_bind"): 8000,
    ("hdc_sft.jsonl", "fact_counter"): 4000,
    ("hdc_sft.jsonl", "fact_chain"): 4000,
    ("hdc_sft.jsonl", "fact_absent"): 8000,
    ("events_sft.jsonl", None): None,
    ("history_sft.jsonl", "hist_cause"): 16000,         # the history graph's links: what led to what
    ("history_sft.jsonl", "hist_effect"): 16000,
    ("history_sft.jsonl", "hist_chain"): 3000,
    ("history_sft.jsonl", "hist_downstream"): 2000,     # what a branch that changes an event has to re-evaluate
    ("history_sft.jsonl", "hist_part"): 3000,
    ("history_sft.jsonl", "hist_order"): 4000,
    ("history_sft.jsonl", "hist_when"): 4000,
    ("history_sft.jsonl", "hist_where"): 2000,
    ("history_sft.jsonl", "hist_who"): 2000,
    ("history_sft.jsonl", "hist_fact"): 4000,
    ("history_sft.jsonl", "hist_absent"): 5000,
    ("passage_sft.jsonl", "passage"): 12000,
    ("passage_sft.jsonl", "passage_absent"): 6000,
    ("webqa_sft.jsonl", "passage_web"): 12000,
    ("webqa_sft.jsonl", "passage_web_absent"): 5000,
    ("quotes_sft.jsonl", "quote"): 5000,
    ("quotes_sft.jsonl", "quote_absent"): 2000,
    ("chat_sft.jsonl", "chat"): 22000,
    ("chat_sft.jsonl", "chat_gated"): None,
    ("explicit_sft.jsonl", "story_open"): None,
    ("explicit_sft.jsonl", "story_gated"): None,
}
# v2.2 (2026-09-25): v2.1 missed H-E6's absent bar by one answer (57/64), its misses a relation stitched out of
# other lines; v2.2 changes that and nothing else -- the facts-block absent records the history graph and the hdc set
# still had unused (hist_absent 5k -> 15k of 46k, fact_absent 8k -> all), so the gate reads one change
MIX_V22 = dict(MIX_V21)
MIX_V22.update({("history_sft.jsonl", "hist_absent"): 15000, ("hdc_sft.jsonl", "fact_absent"): None})
# train records written more than once (a second pass inside one epoch); held records are never repeated
REPEAT = {"v2": {}, "v2.1": {("ground_sft.jsonl", None): 2}, "v2.2": {("ground_sft.jsonl", None): 2}}
# held records kept per family (seeded sample) where a file's held side runs to tens of thousands; others keep all
HELD_CAP = {"v2": {}, "v2.1": {"history_sft.jsonl": 1000}, "v2.2": {"history_sft.jsonl": 1000}}
MIXES = {"v2": MIX, "v2.1": MIX_V21, "v2.2": MIX_V22}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mix", choices=sorted(MIXES), default="v2")
    ap.add_argument("--out", default="", help="default standin/data/out/talk_<mix>.jsonl")
    ap.add_argument("--manifest", default="", help="default validation/logs/talk_<mix>.manifest.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-to", default="", help="also copy the mix here (the Drive folder Colab reads)")
    args = ap.parse_args()
    name = "talk_" + args.mix.replace(".", "_")
    args.out = args.out or str(OUT / f"{name}.jsonl")
    args.manifest = args.manifest or str(ROOT / "validation" / "logs" / f"{name}.manifest.json")
    mix, repeat = MIXES[args.mix], REPEAT[args.mix]
    rng = random.Random(args.seed)
    held_cap = HELD_CAP[args.mix]
    drawn, have, out = collections.Counter(), collections.Counter(), []
    loaded = (None, [])                                        # one file read once for its run of families
    for (fname, fam), n in mix.items():
        path = OUT / fname
        if not path.exists():
            print(f"missing {fname}: skipped"); continue
        if loaded[0] != fname:
            loaded = (fname, [json.loads(l) for l in open(path, encoding="utf-8")])
        rows = [r for r in loaded[1] if fam is None or r.get("family") == fam]
        train = [r for r in rows if r.get("split") == "train"]
        held = [r for r in rows if r.get("split") == "held"]
        pick = train if n is None or n >= len(train) else rng.sample(train, n)
        if fname in held_cap and len(held) > held_cap[fname]:
            held = rng.sample(held, held_cap[fname])
        key = fam or fname.split(".")[0]
        have[key] = len(train); drawn[key] = len(pick)
        drawn[key + ":held"] = len(held)
        times = repeat.get((fname, fam), 1)
        if times > 1:
            drawn[key + ":repeat"] = times
        for k in range(times):
            for r in pick + (held if k == 0 else []):
                out.append({"id": r["id"] + (f"#{k + 1}" if k else ""), "family": r.get("family", key),
                            "split": r["split"], "prompt": r["prompt"], "answer": r["answer"],
                            **({"explicit": True} if r.get("explicit") else {})})
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = collections.Counter((r["family"], r["split"]) for r in out)
    manifest = {"mix_version": args.mix, "seed": args.seed, "records": len(out),
                "train": sum(v for (f, s), v in fams.items() if s == "train"),
                "held": sum(v for (f, s), v in fams.items() if s == "held"),
                "by_family": {f"{f}:{s}": v for (f, s), v in sorted(fams.items())},
                "drawn_of_available": {k: f"{drawn[k]} of {have[k]}" for k in have},
                "mix": {f"{fn}:{fam or '*'}": n for (fn, fam), n in mix.items()},
                "repeat": {f"{fn}:{fam or '*'}": n for (fn, fam), n in repeat.items()},
                **({"held_cap": held_cap} if held_cap else {})}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_to:
        pathlib.Path(args.copy_to).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.out, pathlib.Path(args.copy_to) / pathlib.Path(args.out).name)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
