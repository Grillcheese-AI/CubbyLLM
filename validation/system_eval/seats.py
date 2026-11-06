"""validation/system_eval/seats.py -- order 0054's programs seat, split between home and Colab.

The 450M is the stack's only GPU part, called once per question the host does not read: ``SlotEmitter`` annotates the
question with the names the host knows and hands the annotated prompt to the seat (``Cubby450mEmitter.emit``). So:
  ``Recorder``   the seat at home, first pass: writes down each prompt it is handed (and its token budget) and
                 proposes nothing ("no plan"), so the pass is also the stack with no model in the seat
  ``Replay``     the seat at home, second pass: the program Colab's torch seat wrote for that exact prompt; a prompt
                 with none stops the run (``MissingProgram`` is not an ``Exception``: the loop's own handler cannot
                 turn it into a refusal)
  ``Gather``     between Colab rounds: the replay where a program exists, the recorder where none does (the online
                 pass annotates some questions with names the offline pass did not know)
  ``TorchTrunk`` the 450M in torch (Colab), with ``Cubby450mTrunk``'s interface (``generate``, ``tokenizer``,
                 ``eos``, ``adapter_dirs``, ``max_ctx``): ``Cubby450mEmitter`` and the talk seat run on it unchanged.
                 One frozen base per role (the programs adapter is on base450m_cont, the talk adapter on base450m);
                 rows decode together, greedy, with no padding in any row's state (``selfimprove.trunk.sample_many``)
``programs`` turns prompts into programs exactly as ``Cubby450mEmitter.emit`` does, in batches.

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL), os.path.join(os.path.dirname(VAL), "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

NO_PLAN = "no plan\n"


class MissingProgram(BaseException):
    """A prompt the replay has no program for: the record is not the stack's, so the run stops."""


class Recorder:
    """The seat's first pass: each prompt handed to it, kept in order of first sight; proposes nothing."""
    name = "recorder"

    def __init__(self) -> None:
        self.prompts: dict[str, int] = {}
        self.calls = 0
        self.last: str | None = None
        self.last_program: str | None = None

    def emit(self, prompt: str, max_new_tokens: int = 768, **kw) -> str:
        self.calls += 1
        self.last, self.last_program = prompt, NO_PLAN
        self.prompts.setdefault(prompt, int(max_new_tokens))
        return NO_PLAN

    def save(self, path: str) -> int:
        """Appends the prompts not already in ``path`` (jsonl: {prompt, max_new}); returns how many it added."""
        have = set()
        if os.path.exists(path):
            have = {json.loads(x)["prompt"] for x in open(path, encoding="utf-8") if x.strip()}
        new = [(p, n) for p, n in self.prompts.items() if p not in have]
        with open(path, "a", encoding="utf-8") as f:
            for p, n in new:
                f.write(json.dumps({"prompt": p, "max_new": n}, ensure_ascii=False) + "\n")
        return len(new)


class Replay:
    """The seat's second pass: Colab's program for the exact prompt."""
    name = "replay"

    def __init__(self, programs: dict[str, str]) -> None:
        self.programs = programs
        self.calls = 0
        self.last: str | None = None
        self.last_program: str | None = None

    @classmethod
    def load(cls, path: str) -> "Replay":
        return cls({r["prompt"]: r["program"] for r in map(json.loads, open(path, encoding="utf-8")) if r})

    def emit(self, prompt: str, max_new_tokens: int = 768, **kw) -> str:
        self.calls += 1
        self.last = prompt
        if prompt not in self.programs:
            raise MissingProgram(prompt[:200])
        self.last_program = self.programs[prompt]
        return self.last_program


class Gather(Replay):
    """Between rounds: Colab's program where it has one, else the prompt is written down (as ``Recorder`` does) and
    nothing is proposed. Its records are not arm C's; it only collects the prompts the online pass hands the seat
    that the offline record pass did not, for the next Colab round."""
    name = "gather"

    def __init__(self, programs: dict[str, str]) -> None:
        super().__init__(programs)
        self.prompts: dict[str, int] = {}

    def emit(self, prompt: str, max_new_tokens: int = 768, **kw) -> str:
        try:
            return super().emit(prompt, max_new_tokens, **kw)
        except MissingProgram:
            self.last_program = NO_PLAN
            self.prompts.setdefault(prompt, int(max_new_tokens))
            return NO_PLAN

    save = Recorder.save


class TorchTrunk:
    """``roles``: {role: (checkpoint, adapter dir, adapter file name)}. Each role loads its base, frozen, in eval."""

    def __init__(self, roles: dict, tokenizer: str, dev: str | None = None, max_ctx: int = 2048, rows: int = 64) -> None:
        import torch
        from tokenizers import Tokenizer
        from train_base import load_base
        from train_talk_torch import load_adapter
        self.dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.eos = self.tokenizer.token_to_id("</s>")
        self.max_ctx, self.rows = max_ctx, rows
        self.adapter_dirs = {r: v[1] for r, v in roles.items()}
        self.models, self.meta = {}, {}
        for role, (ckpt, adir, name) in roles.items():
            m = load_base(ckpt, str(self.dev))
            for p in m.parameters():
                p.requires_grad_(False)
            self.meta[role] = load_adapter(m, adir, name)
            self.models[role] = m

    def generate(self, role: str, rows: list, max_new_tokens: int, temperature: float = 0.0, top_k: int = 40,
                 seed: "int | None" = None) -> list:
        """``rows`` of token ids -> the generated ids per row (no prompt, no </s>), greedy, in batches of
        ``self.rows`` sorted by length."""
        from selfimprove.trunk import sample_many
        if temperature:
            raise ValueError("order 0054 decodes greedy only")
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        out = [None] * len(rows)
        for s in range(0, len(order), self.rows):
            ids = order[s:s + self.rows]
            got = sample_many(self.models[role], [rows[i] for i in ids], 1, 0.0, [max_new_tokens] * len(ids),
                              self.eos, self.dev)
            for i, g in zip(ids, got):
                out[i] = g[0]
        return out


def programs(trunk, prompts: list[str], max_new: list[int], role: str = "programs") -> list[str]:
    """Each prompt's program, as ``Cubby450mEmitter(trunk).emit(prompt, max_new_tokens=n)`` writes it, batched by
    budget."""
    from cubbyllm.emit.format import PROMPT_HEAD, PROMPT_TAIL, decode_program
    ids = []
    for p in prompts:
        text = p if p.startswith(PROMPT_HEAD) else PROMPT_HEAD + p.strip() + PROMPT_TAIL
        ids.append(trunk.tokenizer.encode(text).ids[-trunk.max_ctx:])
    out = [None] * len(prompts)
    for n in sorted(set(max_new)):
        sel = [i for i, m in enumerate(max_new) if m == n]
        for i, g in zip(sel, trunk.generate(role, [ids[i] for i in sel], n)):
            out[i] = decode_program(trunk.tokenizer, g, trunk.eos)
    return out


__all__ = ["Gather", "MissingProgram", "NO_PLAN", "Recorder", "Replay", "TorchTrunk", "programs"]
