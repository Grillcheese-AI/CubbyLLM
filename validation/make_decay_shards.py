"""make_decay_shards — three token shards for the decay phase of a base run: instruction following,
agentic trajectories, and the emitter's own programs (numpy + tokenizers only).

Nick, 2026-10-17: "add instruction following, that's a weak spot"; "we also need to do something for
agentic". The base has only ever seen these through adapters (the talk LoRA's chat family, the
emitter LoRA); the decay phase is where the base run puts its best data last, and these go in beside
it, each as one more ``<source>.u32`` ``train_base.Corpus`` picks up, weighted by ``CB_DECAY_JSON``.

    instruct.u32   the host's own guarded chats (standin/data/out/chat_sft.jsonl: Nemotron-SFT-Instruction-
                   Following-Chat-v2, kept only when the answer names nothing the conversation did not --
                   build_chat_sft.py's rule), in the talk adapter's format: the prompt as written there
                   ("Conversation: ... Question: ... Answer:"), a space, the answer, </s>.
    agent.u32      open_thoughts_agent_trove: multi-turn terminal-bench trajectories -- an observation, then a
                   strict JSON reply (analysis, plan, commands, task_complete), then the next observation -- the
                   observe -> plan -> act -> observe loop in a machine-readable action format. Our tools are not
                   shell commands; the loop's shape and the format discipline are what the base takes from it.
                   Sampled evenly from the file to a token budget; a trajectory runs 3-5k tokens, so at
                   CB_S=2048 its early turns sit beyond the 512-token window: the case the episodic memory reads.
    program.u32    the emitter SFT in slot form (emitter_sft_v12e_w_slots.jsonl, + the program-first set when
                   present): "Question: ... Program: ..." with the host's $E / $N slots, train split only, so the
                   base itself learns the program shape the emitter LoRA was carrying alone.

Each shard ends with a held-out tail (``--holdout-frac`` of its tokens, document-aligned) recorded in
``_holdout.json`` beside the cache's others, and its training size in ``_sizes.json``, so a run validates on
exactly that tail and never trains on it. Documents are separated by </s>, as every shard in the cache is
(no <s>). The split of every record stays what its builder said: a held record never enters a shard's
training region.

    python validation/make_decay_shards.py --cache <token_cache_cont> [--agent-tokens 150000000]
        [--instruct standin/data/out/chat_sft.jsonl] [--program standin/data/out/emitter_sft_v12e_w_slots.jsonl ...]
        [--agent <open_thoughts_agent_trove.jsonl>] [--tokenizer <bbpe128k tokenizer.json>] [--dry]

Writes validation/logs/make_decay_shards_<date>.json with every count.
"""
from __future__ import annotations

import pathlib as _pathlib  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))      # the repo root, for:
from cubbyllm.local_paths import expand as _lp  # noqa: E402 -- local folders by name (order 0030)

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "standin", "data", "out")
MODELS = os.environ.get("CUBBY_MODELS", _lp(r"$CUBBY_MODELS"))
DATASETS = os.environ.get("CUBBY_DATASETS", "")


def log(msg: str) -> None:
    print(msg, flush=True)


class Shard:
    """Appends documents as uint32 tokens to <name>.u32, </s> after each one."""

    def __init__(self, cache: str, name: str, eos: int, dry: bool) -> None:
        self.path = os.path.join(cache, f"{name}.u32")
        self.name, self.eos, self.dry = name, eos, dry
        self.n_tokens = self.n_docs = 0
        self.doc_ends: list[int] = []                     # token offsets after each </s>, for the held-out cut
        self.f = None if dry else open(self.path + ".part", "wb")

    def add(self, ids: list[int]) -> None:
        arr = np.asarray(ids + [self.eos], dtype=np.uint32)
        if self.f is not None:
            self.f.write(arr.tobytes())
        self.n_tokens += len(arr)
        self.n_docs += 1
        self.doc_ends.append(self.n_tokens)

    def close(self, holdout_frac: float) -> tuple[int, int]:
        """Returns (training tokens, held-out tokens): the tail from the first document boundary at or
        past (1 - frac) of the shard."""
        cut = int(self.n_tokens * (1.0 - holdout_frac))
        start = next((e for e in self.doc_ends if e >= cut), self.n_tokens)
        hold = self.n_tokens - start
        if self.f is not None:
            self.f.close()
            os.replace(self.path + ".part", self.path)
        return self.n_tokens - hold, hold


def encode_batches(tk, texts: list[str], batch: int = 256):
    for i in range(0, len(texts), batch):
        for enc in tk.encode_batch(texts[i:i + batch]):
            yield enc.ids


def build_instruct(tk, shard: Shard, path: str, limit_tokens: int) -> dict:
    """chat_sft records, train split: prompt + " " + answer. The prompt already ends with "Answer:"."""
    texts, held, n = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "held":
                held += 1
                continue
            texts.append(r["prompt"].rstrip() + " " + r["answer"].strip())
    for ids in encode_batches(tk, texts):
        shard.add(ids)
        n += 1
        if limit_tokens and shard.n_tokens >= limit_tokens:
            break
    return {"source": os.path.basename(path), "records": n, "skipped_held": held, "tokens": shard.n_tokens}


def build_program(tk, shard: Shard, paths: list[str]) -> dict:
    """The emitter's own records, train split only: the prompt as written (Question / slots / Program:),
    the program, a newline -- emitter_data.encode_records's text, minus its training-time repeat."""
    out = {"sources": [], "tokens": 0}
    for path in paths:
        if not os.path.exists(path):
            log(f"  program: {path} not found, skipped")
            continue
        texts, held = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") != "train":
                    held += 1
                    continue
                texts.append(r["prompt"].rstrip() + "\n" + r["program"].rstrip() + "\n")
        n = 0
        for ids in encode_batches(tk, texts):
            shard.add(ids)
            n += 1
        out["sources"].append({"source": os.path.basename(path), "records": n, "skipped_not_train": held})
    out["tokens"] = shard.n_tokens
    return out


def build_agent(tk, shard: Shard, path: str, budget: int, seed: int, max_chars: int = 60_000) -> dict:
    """Trajectories sampled evenly across the file (the file is tens of GB; one pass, every k-th record kept,
    k chosen from the file size and a measured tokens-per-byte so the budget lands)."""
    size = os.path.getsize(path)
    rng = random.Random(seed)
    # measure tokens per byte on the first records, then the keep rate the budget allows
    sample, sb = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sample.append(r["text"]); sb += len(line.encode("utf-8"))
            if len(sample) >= 64:
                break
    st = sum(len(ids) for ids in encode_batches(tk, sample))
    tok_per_byte = st / max(sb, 1)
    total_est = size * tok_per_byte
    keep = min(1.0, budget / max(total_est, 1.0)) if budget else 1.0
    log(f"  agent: {size / 1e9:.1f} GB, ~{total_est / 1e9:.2f}B tokens at {tok_per_byte:.3f} tok/byte; keeping ~{keep:.3%}")
    n = skipped_long = 0
    batch: list[str] = []
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if rng.random() > keep:
                continue
            r = json.loads(line)
            t = r["text"]
            if len(t) > max_chars:
                skipped_long += 1
                continue
            batch.append(t)
            if len(batch) >= 256:
                for ids in encode_batches(tk, batch):
                    shard.add(ids); n += 1
                batch = []
                if budget and shard.n_tokens >= budget:
                    break
                if n % 5120 == 0:
                    log(f"    {n:,} trajectories, {shard.n_tokens / 1e6:.0f}M tokens, {time.time() - t0:.0f}s")
    for ids in encode_batches(tk, batch):
        if budget and shard.n_tokens >= budget:
            break
        shard.add(ids); n += 1
    return {"source": os.path.basename(path), "trajectories": n, "skipped_long": skipped_long,
            "tokens": shard.n_tokens, "keep_rate": keep, "tok_per_byte": tok_per_byte}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the token cache to add the shards to (has _holdout.json, _sizes.json)")
    ap.add_argument("--tokenizer", default=os.path.join(MODELS, "base450m", "grilly", "tokenizer.json"))
    ap.add_argument("--instruct", default=os.path.join(OUT, "chat_sft.jsonl"))
    ap.add_argument("--instruct-tokens", type=int, default=0, help="0 = every train record")
    ap.add_argument("--program", nargs="*", default=[os.path.join(OUT, "emitter_sft_v12e_w_slots.jsonl"),
                                                     os.path.join(OUT, "emitter_sft_v12e_pf_slots.jsonl")])
    ap.add_argument("--agent", default=os.path.join(DATASETS or _lp(r"$CUBBY_TRAINING_DATA"), "unified",
                                                    "open_thoughts_agent_trove.jsonl"))
    ap.add_argument("--agent-tokens", type=int, default=150_000_000)
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=["instruct", "agent", "program"])
    ap.add_argument("--dry", action="store_true", help="count, write nothing")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    hp, sp = os.path.join(args.cache, "_holdout.json"), os.path.join(args.cache, "_sizes.json")
    holds = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else {}
    sizes = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    report = {"args": vars(args), "shards": {}}
    t0 = time.time()
    for name in args.only:
        log(f"{name}:")
        shard = Shard(args.cache, name, eos, args.dry)
        if name == "instruct":
            rep = build_instruct(tk, shard, args.instruct, args.instruct_tokens)
        elif name == "program":
            rep = build_program(tk, shard, args.program)
        elif name == "agent":
            rep = build_agent(tk, shard, args.agent, args.agent_tokens, args.seed)
        else:
            raise SystemExit(f"unknown shard {name}")
        train, hold = shard.close(args.holdout_frac)
        rep.update(train_tokens=train, holdout_tokens=hold, documents=shard.n_docs)
        report["shards"][name] = rep
        if not args.dry:
            holds[name], sizes[name] = int(hold), int(train)
        log(f"  {name}.u32: {shard.n_docs:,} documents, {train / 1e6:.1f}M training tokens + {hold / 1e6:.2f}M held out "
            f"({time.time() - t0:.0f}s)")
    if not args.dry:
        json.dump(holds, open(hp, "w", encoding="utf-8"), indent=1)
        json.dump(sizes, open(sp, "w", encoding="utf-8"), indent=1)
        log(f"updated {hp} and {sp}")
    logp = os.path.join(HERE, "logs", f"make_decay_shards_{_dt.date.today().isoformat()}.json")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    json.dump(report, open(logp, "w", encoding="utf-8"), indent=1)
    log(f"wrote {logp}")


if __name__ == "__main__":
    main()
