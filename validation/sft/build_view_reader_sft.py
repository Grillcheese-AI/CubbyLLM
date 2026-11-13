"""Order 0067, the second reader's step 5: its SFT set, in the talk trainer's record form
(``validation/train_talk_torch.py``: {"id", "split", "family", "prompt", "answer"}).

Families:
- ``long_<n>``: step 3's long instructions (`$CUBBY_DATASETS/long_rows_0067`, ``train`` -> train, ``dev`` -> held);
  ``test`` and ``test_long`` are never trained on (the reader's lines read them);
- ``single``: step 2's checks that are one atom or its negation, said in English or French (`view_wording`), as a
  one-part instruction with its split and join; a seeded tenth held;
- ``curriculum``: step 4's compiled programs as continuations ("continue this CubeLang program"), cut at a line,
  answers up to `WINDOW` lines; a seeded twentieth held.
The prompt and answer are the reader's own form (`cubbyllm/speech/view_reader.py`). With ``--tokenizer`` the records
longer than ``--max-len`` are counted (the trainer skips them). Every instruction is checked for 5-word overlap with
IFEval and IFBench (none may reach a training record).

  python validation/sft/build_view_reader_sft.py --out <view_reader_sft.jsonl> [--tokenizer <json>] [--max-len 1280]
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
sys.path[:0] = [str(ROOT), str(ROOT / "validation" / "builders")]
import bench_overlap as BO  # noqa: E402
import view_join as VJ  # noqa: E402
import view_wording as W  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import view_reader as VR  # noqa: E402

__wiring__ = "STANDALONE"

CODE_HEAD = "Continue this CubeLang program:\n"
WINDOW = 24


def long_records(ds: Path) -> list[dict]:
    out = []
    for r in map(json.loads, open(ds / "long_rows_0067" / "rows.jsonl", encoding="utf-8")):
        if r["split"] in ("train", "dev"):
            out.append({"id": r["id"], "split": "train" if r["split"] == "train" else "held", "family": f"long_{r['n_parts']}",
                        "lang": r["lang"], "prompt": VR.prompt(r["instruction"]), "answer": VR.target(r)})
    return out


def single_records(ds: Path, rng: random.Random) -> list[dict]:
    out = []
    for r in map(json.loads, open(ds / "view_programs_0067" / "rows.jsonl", encoding="utf-8")):
        s = r["spec"]
        if "q" not in s and not ("not" in s and "q" in s["not"]):
            continue
        lang = "fr" if rng.random() < 0.4 else "en"
        text = W.say_not(s["not"], lang, rng) if "not" in s else W.say(s, lang, rng)
        row = {"instruction": text, "segments": [{"kind": "check", "part": 0, "start": 0, "end": len(text)}],
               "join": VJ.join([s], [("one", 0)])}
        out.append({"id": "s" + r["id"], "split": "held" if rng.random() < 0.1 else "train", "family": "single",
                    "lang": lang, "prompt": VR.prompt(text), "answer": VR.target(row)})
    return out


def curriculum_records(ds: Path, rng: random.Random) -> list[dict]:
    out = []
    for r in map(json.loads, open(ds / "cubelang_curriculum_0067" / "programs.jsonl", encoding="utf-8")):
        lines = r["text"].rstrip("\n").split("\n")
        if len(lines) < 4:
            continue
        cut = rng.randint(1, max(1, len(lines) - 2))
        answer = "\n".join(lines[cut:cut + WINDOW])
        out.append({"id": r["id"], "split": "held" if rng.random() < 0.05 else "train", "family": f"curriculum_{r['kind']}",
                    "lang": "code", "prompt": CODE_HEAD + "\n".join(lines[:cut]) + "\n", "answer": answer})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    rng = random.Random(a.seed)
    recs = long_records(ds) + single_records(ds, rng) + curriculum_records(ds, rng)
    bench = BO.benchmark_grams(ifbench=True)
    leaks = [r["id"] for r in recs if r["lang"] != "code" and BO.overlaps(r["prompt"].split("Instruction: ", 1)[-1], bench)]
    recs = [r for r in recs if r["id"] not in set(leaks)]
    over, dropped = Counter(), Counter()
    if a.tokenizer:                                       # longer than the trainer takes: dropped here and counted
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
        keep = []
        for r in recs:
            if len(tk.encode(r["prompt"]).ids) + len(tk.encode(" " + r["answer"]).ids) + 1 > a.max_len:
                dropped[r["family"]] += 1
            else:
                keep.append(r)
        recs = keep
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"order": "0067", "what": "the second reader's SFT set (step 5)", "records": len(recs),
           "by_split": dict(Counter(r["split"] for r in recs)), "by_family": dict(sorted(Counter(r["family"] for r in recs).items())),
           "by_lang": dict(Counter(r["lang"] for r in recs)), "benchmarks_checked": sorted(bench),
           "records_sharing_5_words_with_a_benchmark_dropped": len(leaks), "max_len": a.max_len,
           "records_over_max_len": dict(over) if a.tokenizer else None,
           "records_dropped_for_length": dict(dropped) if a.tokenizer else None,
           "row_licences": {"templates": "this repo's own (Apache-2.0)", "cubelang_examples_and_tests": "Nick's cubelang repo",
                            "programs_seat_rows": "this repo's own programs (no benchmark rows)"},
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
