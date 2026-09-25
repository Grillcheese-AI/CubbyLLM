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
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT / "talk_v2.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "talk_v2.manifest.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-to", default="", help="also copy the mix here (the Drive folder Colab reads)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    drawn, have, out = collections.Counter(), collections.Counter(), []
    for (fname, fam), n in MIX.items():
        path = OUT / fname
        if not path.exists():
            print(f"missing {fname}: skipped"); continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if fam is None or r.get("family") == fam]
        train = [r for r in rows if r.get("split") == "train"]
        held = [r for r in rows if r.get("split") == "held"]
        pick = train if n is None or n >= len(train) else rng.sample(train, n)
        key = fam or fname.split(".")[0]
        have[key] = len(train); drawn[key] = len(pick)
        drawn[key + ":held"] = len(held)
        for r in pick + held:
            out.append({"id": r["id"], "family": r.get("family", key), "split": r["split"], "prompt": r["prompt"],
                        "answer": r["answer"], **({"explicit": True} if r.get("explicit") else {})})
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = collections.Counter((r["family"], r["split"]) for r in out)
    manifest = {"seed": args.seed, "records": len(out),
                "train": sum(v for (f, s), v in fams.items() if s == "train"),
                "held": sum(v for (f, s), v in fams.items() if s == "held"),
                "by_family": {f"{f}:{s}": v for (f, s), v in sorted(fams.items())},
                "drawn_of_available": {k: f"{drawn[k]} of {have[k]}" for k in have},
                "mix": {f"{fn}:{fam or '*'}": n for (fn, fam), n in MIX.items()}}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_to:
        pathlib.Path(args.copy_to).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.out, pathlib.Path(args.copy_to) / pathlib.Path(args.out).name)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
