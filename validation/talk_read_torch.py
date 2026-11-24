"""talk_read_torch - the talk seat on torch (Colab, CUDA), as `Cubby450mTalk` renders and cuts it, for order 0079's
emotion read after training (`exp_0079_appraiser_read.py --torch-ckpt`).

`emit` is `Cubby450mTalk.emit` on the torch base: `talk_prompt` over the prompt and the context, greedy to </s>, the
answer cut by `talk_answer`; the system prompt is ignored (the adapter has no system turn). `emit_batch` gives the
same answers for many prompts at once (`dev_read_torch.rows_ids`: the shared prefix read once, one batch per prompt
length; exact, pinned in tests/validation/test_dev_read_torch.py).

Wiring: STANDALONE (order 0079's read).
"""
from __future__ import annotations

import os

__wiring__ = "STANDALONE"


class TorchTalk:
    def __init__(self, ckpt: str, adapter: str, tokenizer: str):
        import torch
        from tokenizers import Tokenizer
        from train_base import load_base
        from train_talk_torch import load_adapter
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tk = Tokenizer.from_file(tokenizer)
        self.eos = self.tk.token_to_id("</s>")
        self.model = load_base(ckpt, str(self.dev))
        load_adapter(self.model, adapter, "talk_lora")
        self.name = f"torch-talk:{os.path.basename(os.path.normpath(adapter))}"
        self.cache: list = []

    def emit_batch(self, prompts: list[str], context=None, max_new_tokens: int = 24, **kw) -> list[str]:
        from cubbyllm.emit.talk import talk_answer, talk_prompt
        from dev_read_torch import rows_ids
        ids = [self.tk.encode(talk_prompt(p, context)).ids for p in prompts]
        rows = rows_ids(self.model, ids, self.eos, max_new_tokens, 0.0, 0, self.dev, self.cache)
        return [talk_answer(self.tk.decode(r)) for r in rows]

    def emit(self, prompt: str, max_new_tokens: int = 24, system=None, context=None, **kw) -> str:
        return self.emit_batch([prompt], context, max_new_tokens)[0]


__all__ = ["TorchTalk"]
