"""Order 0067, step 3's adapter round (Nick, 2026-11-12: "yes"): the SFT set that teaches the 450M to read an
instruction into a rule plan, in the reader's own form (`cubbyllm/speech/rule_reader.py`).

From `build_rule_rows.py`'s rows: ``train`` -> train, ``dev`` -> held (the trainer's held-out loss, and the read that
picks the checkpoint at home). ``new_wording`` rows are left out: line 3 reads them once, after. No held-out kind has
a row at all. A record (``validation/train_talk_torch.py``'s format):
    {"id", "split", "family": <kind>, "prompt": HEAD + "Instruction: <instruction>\\nRule:", "answer": <surface> + "\\n"}
The trainer writes " " + answer, then </s>; the reader ends a rule at the newline.

The reader tokenizes each slot after the text written so far; training tokenizes the whole answer. With
``--tokenizer``, every slot boundary of every record is checked: encode(text) + encode(" " + slot) must equal
encode(text + " " + slot). Mismatches are counted in the manifest (they mean the read meets tokens training never
showed).

  python validation/sft/build_rule_reader_sft.py [--rows <rows.jsonl>] --out <rule_reader_sft.jsonl> [--tokenizer <json>]
      [--repeat-asked N] [--max-len N]   # N: the notebook's MAX_LEN; the records over it are counted, not written apart
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import rule_reader as rr  # noqa: E402

__wiring__ = "STANDALONE"

SPLIT = {"train": "train", "dev": "held"}


def record(r: dict) -> dict:
    return {"id": r["id"], "split": SPLIT[r["split"]], "family": r["kind"], "lang": r["lang"],
            "prompt": rr.HEAD + f"Instruction: {r['instruction']}\nRule:",
            "answer": ("asked" if r.get("asked") else rr.surface(r["plan"])) + "\n"}


def slot_texts(rec: dict) -> list[tuple[str, str]]:
    """(text written so far, the next slot) at every boundary the reader writes."""
    out, text = [], rec["prompt"]
    if rec["answer"].strip() in ("none", "asked"):       # no rule (round 4) or one no measurement covers (round 5)
        return [(text, rec["answer"].strip())]
    for leaf_i, leaf in enumerate(rec["answer"].rstrip("\n").split(" ; ")):
        if leaf_i:
            out.append((text, ";"))
            text += " ;"
        parts = leaf.split(" ")
        # measurement words up to the argument or the comparison
        ops = {"<=", ">=", "<", ">", "==", "!=", "has", "lacks"}
        k = next(i for i, p in enumerate(parts) if p in ops or p.startswith('"'))
        slots = [" ".join(parts[:k])]
        rest = parts[k:]
        if rest[0].startswith('"'):
            j = next(i for i, p in enumerate(rest) if p in ops)
            slots.append(" ".join(rest[:j]))
            rest = rest[j:]
        slots += [rest[0], " ".join(rest[1:])]
        for s in slots:
            out.append((text, s))
            text += " " + s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--repeat-asked", type=int, default=1, help="write each train record answering asked this many times")
    ap.add_argument("--max-len", type=int, default=0, help="the trainer's --max-len: count the records it would skip")
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(local_paths.value("CUBBY_DATASETS") or ".") / "rule_rows_0067" / "rows.jsonl")
    rows = [json.loads(line) for line in open(rows_p, encoding="utf-8")]
    recs = [record(r) for r in rows if r["split"] in SPLIT]
    recs += [dict(rec, id=f"{rec['id']}~{j}") for rec in list(recs) if rec["split"] == "train" and rec["answer"] == "asked\n"
             for j in range(1, a.repeat_asked)]           # round 5: abstaining weighs like the other rows
    tk = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
    bad, checked, examples, over = 0, 0, [], Counter()
    if tk and a.max_len:                                 # the trainer skips a record longer than --max-len, silently
        for rec in recs:
            if len(tk.encode(rec["prompt"]).ids) + len(tk.encode(" " + rec["answer"]).ids) + 1 > a.max_len:
                over[rec["split"]] += 1
    if tk:
        for rec in recs:
            for text, slot in slot_texts(rec):
                checked += 1
                if tk.encode(text).ids + tk.encode(" " + slot).ids != tk.encode(text + " " + slot).ids:
                    bad += 1
                    if len(examples) < 5:
                        examples.append({"text_end": text[-30:], "slot": slot})
            end = rec["prompt"] + " " + rec["answer"].rstrip("\n")
            checked += 1
            if tk.encode(end).ids + tk.encode("\n").ids != tk.encode(end + "\n").ids:
                bad += 1
    sys.path.insert(0, str(ROOT / "validation" / "builders"))
    import bench_overlap as BO                           # no benchmark prompt in a training record (Cowork, 0067)
    bench = BO.benchmark_grams(ifbench=True)             # reported only: the rows decide what is in
    leaks = [r["id"] for r in rows if r["split"] in SPLIT and BO.overlaps(r["instruction"], bench)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    man = {"order": "0067", "what": "the rule reader's SFT set (the adapter round)", "records": len(recs),
           "by_split": dict(Counter(r["split"] for r in recs)), "by_lang": dict(Counter(r["lang"] for r in recs)),
           "families": len({r["family"] for r in recs}), "rows_sha256": hashlib.sha256(rows_p.read_bytes()).hexdigest(),
           "sft_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "new_wording_rows_left_out": sum(
               1 for r in rows if r["split"] == "new_wording"),
           "benchmarks_checked": sorted(bench), "records_sharing_5_words_with_a_benchmark": len(leaks),
           "slot_boundaries_checked": checked, "slot_boundaries_tokenized_otherwise": bad, "examples": examples,
           "max_len": a.max_len or None, "records_over_max_len": dict(over) if a.max_len else None}
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
