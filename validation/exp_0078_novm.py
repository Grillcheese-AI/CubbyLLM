"""exp_0078_novm - order 0078's addendum, part 4: a control without the VM, on dev (`exp_0078_dev_read.py --arm novm`).

The 450M trunk S0 and S1 sit on, with **no adapter** (theirs write steps for the VM, not answers), does the arithmetic
itself, 4-shot:
- **the shots:** four GSM8K train problems outside dev's 329 (the train file's `dev` flag), drawn by ``--shots-seed``
  (`random.Random(seed).sample`), each with its worked solution. The calculator marks `<<expr=result>>` are taken out:
  nothing runs them here, and a shot showing them would show a tool this arm does not have.
- **the prompt:** `Question: <q>\\nAnswer: <solution>\\n\\n` for each shot, then `Question: <dev q>\\nAnswer:`.
- **decoding:** at most ``NEW`` = 256 new tokens: one greedy, and K = 16 tries at temperature 0.7, top-k 40 in one batch
  (`dev_read_torch.rows_ids`, the question's state read once; the shots' state reused between questions), seeded by the
  question's index as the dev read's first step is.
- **the answer:** the text is cut where a next `Question:` starts; then the number after the first `####`, else the last
  number written; no number is a refusal.

Each row has the dev read's shape (`greedy`, `tries` with each try's `answer`; `steps` empty), so `--score` chooses k and
m on the 209 and reads 80:200 once exactly as for S0 and S1. The answer-kind rules are not applied to it (they check the
VM's values). Nothing is chosen from it, nothing is served on it, and test is never read with it.

Wiring: STANDALONE (an experiment; order 0078).
"""
from __future__ import annotations

import json
import os
import random
import re
import time

__wiring__ = "STANDALONE"

NEW, K, TEMPERATURE, TOP_K = 256, 16, 0.7, 40
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def shots(train_path: str, seed: int = 0, n: int = 4) -> list[dict]:
    """Four train problems outside dev, fixed by ``seed``: [{"id", "question", "solution"}]."""
    rows = [json.loads(ln) for ln in open(train_path, encoding="utf-8")]
    pool = [r for r in rows if not r.get("dev")]
    pick = random.Random(seed).sample(pool, n)
    return [{"id": r["id"], "question": r["question"].strip(),
             "solution": re.sub(r"<<[^>]*>>", "", r["answer"]).strip()} for r in pick]


def prompt(sh: list[dict], question: str) -> str:
    head = "".join(f"Question: {s['question']}\nAnswer: {s['solution']}\n\n" for s in sh)
    return head + f"Question: {question.strip()}\nAnswer:"


def answer_of(text: str) -> "str | None":
    """The number after the first `####`, else the last number written (the text cut at a next `Question:`)."""
    text = re.split(r"\n\s*Question:", text, maxsplit=1)[0]
    m = re.search(r"####\s*\$?\s*(" + _NUM + ")", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(_NUM, text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "")
    try:
        float(s)
    except ValueError:
        return None
    return s


class NoVmReader:
    """The torch base with no adapter; `read` gives the greedy text and K sampled texts for one prompt."""

    def __init__(self, ckpt: str, tokenizer: str):
        import torch
        from tokenizers import Tokenizer
        from dev_read_torch import stepper_for
        from train_base import load_base
        self.torch, self.dev = torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tk = Tokenizer.from_file(tokenizer)
        self.eos = self.tk.token_to_id("</s>")
        self.model = load_base(ckpt, str(self.dev))
        self.stepper = stepper_for(self.model, self.dev)
        self.decode_kind = "rows-graphs" if type(self.stepper).__name__ == "GraphStep" else "rows"
        self.cache: list = []
        self.pin = None

    def keep_shots(self, sh: list[dict]) -> None:
        """Read the ids every prompt shares (the shots) once and keep that state in the cache for every question
        (`rows_ids` keeps only its last 8 entries)."""
        from dev_read_torch import rows_ids
        a, b = self.tk.encode(prompt(sh, "a")).ids, self.tk.encode(prompt(sh, "b")).ids
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        rows_ids(self.model, [a[:n] + [a[n]]], self.eos, 0, 0.0, 0, self.dev, self.cache, self.stepper)
        self.pin = next((e for e in self.cache if len(e[0]) == n), None)

    def read(self, text: str, qi: int, k: int = K, new: int = NEW) -> tuple:
        """(greedy text, [k sampled texts], (greedy seconds, tries seconds))."""
        from dev_read_torch import rows_ids
        ids = self.tk.encode(text).ids
        if self.pin is not None and not any(e is self.pin for e in self.cache):
            self.cache.insert(0, self.pin)
        t0 = time.perf_counter()
        g = rows_ids(self.model, [ids], self.eos, new, 0.0, 0, self.dev, self.cache, self.stepper)[0]
        t1 = time.perf_counter()
        self.torch.manual_seed(qi * 100)
        ts = rows_ids(self.model, [ids] * k, self.eos, new, TEMPERATURE, TOP_K, self.dev, self.cache, self.stepper)
        t2 = time.perf_counter()
        return self.tk.decode(g), [self.tk.decode(t) for t in ts], (round(t1 - t0, 2), round(t2 - t1, 2))


def run(a, recs: list[dict], reader=None) -> None:
    """``recs``: the dev questions (`exp_0078_dev_read.load`, offset and limit applied); rows appended to ``a.out``."""
    sh = shots(a.gsm8k_train, a.shots_seed)
    reader = reader or NoVmReader(a.ckpt, a.tokenizer)
    if hasattr(reader, "keep_shots"):
        reader.keep_shots(sh)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(ln)["id"] for ln in open(a.out, encoding="utf-8")}
    print(f"novm: shots {[s['id'] for s in sh]} (seed {a.shots_seed}), {len(recs) - len(done & {r['id'] for r in recs})}"
          f" questions to read", flush=True)
    t0, n = time.perf_counter(), 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(recs, start=a.offset):
            if r["id"] in done:
                continue
            q0 = time.perf_counter()
            g, ts, (sg, st) = reader.read(prompt(sh, r["question"]), i, a.k)
            q1 = time.perf_counter()
            ga = answer_of(g)
            f.write(json.dumps({"id": r["id"], "i": i, "gold": r["gold"], "arm": "novm", "adapter": "none (4-shot)",
                                "shots": [s["id"] for s in sh],
                                "greedy": {"answer": ga, "refused": None if ga is not None else "no number",
                                           "steps": [], "text": g},
                                "tries": [{"answer": answer_of(t), "refused": None if answer_of(t) is not None
                                           else "no number", "steps": []} for t in ts],
                                "decode": getattr(reader, "decode_kind", "rows"),
                                "seconds": {"greedy": sg, "tries": st}},
                               ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print(f"  {n} {r['id']} {q1 - q0:.1f}s | mean {(time.perf_counter() - t0) / n:.1f}s a question", flush=True)


__all__ = ["K", "NEW", "NoVmReader", "answer_of", "prompt", "run", "shots"]
