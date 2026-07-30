"""DataPipeline — manifest-pinned corpus source (H-G3 reproducibility).

Wired: STANDALONE — data plumbing, not the forward path.

H-G3 found the candidate corpus (``unified/``) had silently drifted from its own
tokenizer report and mixes unlabeled sources. The lesson is encoded as a
contract: a pipeline must expose ``manifest_hash()`` so any training run pins
the exact mixture it consumed. ``InMemoryDataPipeline`` is the concrete,
testable reference — it holds a token array plus a manifest string and hashes
them together. ``WeightedCorpusPipeline`` (below) is the corpus-backed,
weighted, multi-source implementation over the pinned corpus manifest — it
tokenizes each source once (cached), samples windows by weight, and hashes the
source specs + every input file's signature so a run pins its exact mixture.
"""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.protocols import Wiring

# Strip CR (CRLF -> LF) and stray control chars (keep \t \n) before tokenizing —
# otherwise \r and control bytes byte-fallback into pathologically frequent junk
# tokens (e.g. wiki_full's CRLF gave a \r\n token at ~2%). Shared with the
# tokenize_parallel runner, which inlines the same normalize.
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _normalize_text(t: str) -> str:
    return _CTRL_CHARS.sub("", t.replace("\r", ""))

# Token-cache element type. The 128k BPE core vocab (H-C3/H-G1) exceeds uint16's
# 65,535 ceiling, so shards are uint32. Kept as one constant so a vocab change is
# a one-line edit; the ``.u32`` suffix names the on-disk dtype honestly.
_TOK_DTYPE = "uint32"
_SHARD_EXT = ".u32"


def _load_tokenizer(path: str):
    """Load a tokenizer by extension -> (encode, decode, eos_id, vocab_size).

    ``.json`` -> HuggingFace ``tokenizers`` byte-level BPE (the CubbyLLM default:
    byte-exact, opcode-atomic); any other extension -> SentencePiece (``.model``,
    e.g. the tinystories smoke). ``encode(text)`` returns a ``list[int]``;
    ``decode(ids)`` returns a ``str``; ``eos_id`` is ``-1`` if there is no EOS.
    """
    p = str(path)
    if p.endswith(".json"):
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(p)
        eos = tk.token_to_id("</s>")
        return (lambda s: tk.encode(s).ids), (lambda ids: tk.decode(list(ids))), \
            (eos if eos is not None else -1), tk.get_vocab_size()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(p)
    eos = sp.eos_id() if sp.eos_id() >= 0 else sp.PieceToId("</s>")
    return sp.EncodeAsIds, (lambda ids: sp.DecodeIds(list(ids))), \
        (eos if eos >= 0 else -1), sp.GetPieceSize()

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from torch import Tensor


@runtime_checkable
class DataPipeline(Protocol):
    """A reproducible, manifest-pinned batch source."""

    def manifest_hash(self) -> str:
        """Stable hash of the exact corpus mixture (sources + versions + weights)."""
        ...

    def batches(self, batch_size: int, seq_len: int) -> "Iterator[Tensor]":
        """Yield token-id batches of shape (batch_size, seq_len)."""
        ...


class InMemoryDataPipeline:
    """A token array + a manifest string, hashed for reproducibility.

    tokens   — 1-D integer token-id array (numpy).
    manifest — a human-readable description of the exact mixture (sources,
               versions, weights). Pinned into the hash so a run records what it
               consumed. ``seed`` makes batch sampling deterministic.
    """

    def __init__(self, tokens: "np.ndarray", manifest: str, seed: int = 0) -> None:
        import numpy as np

        self.tokens = np.asarray(tokens).astype(np.int64).reshape(-1)
        self.manifest = str(manifest)
        self.seed = int(seed)

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.manifest.encode("utf-8"))
        h.update(self.tokens.tobytes())
        return h.hexdigest()

    def batches(self, batch_size: int, seq_len: int) -> "Iterator[Tensor]":
        import numpy as np
        import torch

        rng = np.random.default_rng(self.seed)
        n = len(self.tokens) - seq_len - 1
        if n <= 0:
            raise ValueError("corpus shorter than seq_len + 1")
        while True:
            ix = rng.integers(0, n, size=batch_size)
            x = np.stack([self.tokens[i:i + seq_len] for i in ix])
            y = np.stack([self.tokens[i + 1:i + seq_len + 1] for i in ix])
            yield torch.from_numpy(x), torch.from_numpy(y)


class WeightedCorpusPipeline:
    """Manifest-driven, weighted, multi-source corpus pipeline (the H-G3 gate).

    The corpus-backed ``DataPipeline`` the module docstring flagged as a later
    cycle. Each ``source`` is a dict::

        {"name": str, "paths": [glob, ...], "format": "txt"|"jsonl",
         "text_key": str (jsonl only), "weight": float}

    On ``prepare`` each source's text is tokenized once with a byte-level-BPE
    model into a cached ``<name>.u32`` file (uint32 — the 128k core vocab exceeds
    uint16) with an EOS between documents, then memory-mapped. ``batches`` samples random
    ``seq_len+1`` windows across sources **by weight** (so raw size doesn't set
    the mix). ``manifest_hash`` folds in the source specs, weights, tokenizer,
    and every input file's (path, size, mtime) — so a run pins its exact mixture.

    ``max_tokens_per_source`` caps tokenization per source (for fast smoke runs);
    ``None`` tokenizes everything.
    """

    def __init__(
        self,
        sources: "list[dict]",
        spm_path: str,
        cache_dir: str,
        seed: int = 0,
        max_tokens_per_source: "int | None" = None,
        cache_only: bool = False,
    ) -> None:
        self.sources = list(sources)
        self.spm_path = str(spm_path)
        self.cache_dir = str(cache_dir)
        self.seed = int(seed)
        self.cap = max_tokens_per_source
        # cache_only: memmap pre-tokenized shards WITHOUT the source files or a
        # tokenizer — for training on a machine (e.g. Colab) that has only the
        # uploaded token cache, not the E:/D: corpus. Tokenize locally first.
        self.cache_only = bool(cache_only)
        self._arrays: "dict | None" = None
        self._weights: "dict | None" = None

    # ── internals ──────────────────────────────────────────────────────────
    def _files(self, source: dict) -> list:
        # ``paths`` is absent for cache_only sources inferred from shards (Colab):
        # no source files to stat, and none are present there anyway.
        import glob
        out = []
        for pat in source.get("paths", []):
            out.extend(sorted(glob.glob(pat, recursive=True)))
        return out

    @staticmethod
    def _join_msgs(msgs: list) -> str:
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content") or m.get("value") or m.get("text")
                if isinstance(c, str) and c:
                    parts.append(c)
            elif isinstance(m, str) and m:
                parts.append(m)
        return "\n".join(parts)

    def _extract(self, obj: dict, key: str) -> "str | None":
        """Pull training text from a jsonl row. ``key`` if given/present, else
        auto-detect across the heterogeneous schemas in ``unified/`` (plain text,
        content, or a messages/conversations list)."""
        if key and key != "auto":
            v = obj.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, list):
                return self._join_msgs(v) or None
        for k in ("text", "content", "abstract", "body"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
        for k in ("messages", "conversations", "conversation", "turns"):
            v = obj.get(k)
            if isinstance(v, list):
                t = self._join_msgs(v)
                if t:
                    return t
        return None

    def _iter_texts(self, source: dict):
        import json
        fmt, key = source["format"], source.get("text_key", "text")
        for fp in self._files(source):
            if fmt == "txt":
                try:
                    yield _normalize_text(open(fp, encoding="utf-8", errors="replace").read())
                except OSError:
                    continue
            else:  # jsonl
                with open(fp, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = self._extract(json.loads(line), key)
                        except json.JSONDecodeError:
                            continue
                        if t:
                            yield _normalize_text(t)

    def prepare(self) -> "WeightedCorpusPipeline":
        """Tokenize every source into a cached uint32 shard (skips fresh ones).

        In ``cache_only`` mode, just memory-map the existing ``<name>.u32``
        shards (no source files or tokenizer needed)."""
        import os

        import numpy as np

        if self.cache_only:
            arrays, weights = {}, {}
            for s in self.sources:
                binp = os.path.join(self.cache_dir, f"{s['name']}{_SHARD_EXT}")
                if not os.path.exists(binp):
                    raise FileNotFoundError(
                        f"cache_only: missing token shard {binp} — tokenize + "
                        f"upload the cache first")
                arrays[s["name"]] = np.memmap(binp, dtype=_TOK_DTYPE, mode="r")
                weights[s["name"]] = float(s.get("weight", 1.0))
            self._arrays, self._weights = arrays, weights
            return self

        encode, _decode, eos, _ = _load_tokenizer(self.spm_path)
        os.makedirs(self.cache_dir, exist_ok=True)

        arrays, weights = {}, {}
        for s in self.sources:
            binp = os.path.join(self.cache_dir, f"{s['name']}{_SHARD_EXT}")
            sig = os.path.join(self.cache_dir, f"{s['name']}.sig")
            # Freshness compares the stored signature to one computed NOW (over the
            # current on-disk shard). The stored signature is written AFTER the
            # shard (below), so its ``cache:<size>`` reflects the finished file —
            # computing it before the write would bake in the *pre*-write size and
            # never match again, silently re-tokenizing every run.
            fresh = (os.path.exists(binp) and os.path.exists(sig)
                     and open(sig, encoding="utf-8").read() == self._signature(s))
            if not fresh:
                cap = s.get("max_tokens", self.cap)   # per-source cap (e.g. unified)
                # Stream to disk in chunks — a multi-billion-token Python list
                # would be tens of GB of RAM (OOM). Flush ~8M-token buffers.
                buf: list = []
                written = 0
                with open(binp, "wb") as fh:
                    for text in self._iter_texts(s):
                        buf.extend(encode(text))
                        if eos >= 0:
                            buf.append(eos)
                        if len(buf) >= 8_000_000:
                            np.asarray(buf, dtype=_TOK_DTYPE).tofile(fh)
                            written += len(buf); buf.clear()
                        if cap and written >= cap:
                            break
                    if buf:
                        np.asarray(buf, dtype=_TOK_DTYPE).tofile(fh)
                # Re-sign AFTER writing so ``cache:<size>`` matches the finished shard.
                open(sig, "w", encoding="utf-8").write(self._signature(s))
            arrays[s["name"]] = np.memmap(binp, dtype=_TOK_DTYPE, mode="r")
            weights[s["name"]] = float(s.get("weight", 1.0))
        self._arrays, self._weights = arrays, weights
        return self

    def _signature(self, source: dict) -> str:
        import os
        # Tokenizer identity leads the signature: a vocab change (spm32k -> 128k
        # BPE) must invalidate every shard, else the cache silently mixes
        # vocabularies. basename catches the swap; size catches a same-name
        # retrain. Robust to a missing model file (cache_only / Colab).
        parts = [f"spm:{os.path.basename(self.spm_path)}"]
        try:
            parts.append(f"spmsz:{os.path.getsize(self.spm_path)}")
        except OSError:
            pass
        parts += [source["name"], source.get("format", ""),
                  str(source.get("weight", 1.0)), str(self.cap),
                  str(source.get("max_tokens"))]
        binp = os.path.join(self.cache_dir, f"{source['name']}{_SHARD_EXT}")
        if os.path.exists(binp):                    # the tokenized shard itself
            parts.append(f"cache:{os.path.getsize(binp)}")
        for fp in self._files(source):              # source files (empty in cache_only)
            try:
                st = os.stat(fp)
                parts.append(f"{fp}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                pass
        return "\n".join(parts)

    # ── DataPipeline protocol ──────────────────────────────────────────────
    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.spm_path.encode("utf-8"))
        for s in self.sources:
            h.update(self._signature(s).encode("utf-8"))
        return h.hexdigest()

    def batches(self, batch_size: int, seq_len: int) -> "Iterator[Tensor]":
        import numpy as np
        import torch

        if self._arrays is None:
            self.prepare()
        names = [n for n in self._arrays if len(self._arrays[n]) > seq_len + 1]
        if not names:
            raise ValueError("no source has more than seq_len+1 tokens")
        w = np.array([self._weights[n] for n in names], dtype=np.float64)
        w /= w.sum()
        rng = np.random.default_rng(self.seed)
        while True:
            xs, ys = [], []
            picks = rng.choice(len(names), size=batch_size, p=w)
            for p in picks:
                arr = self._arrays[names[p]]
                i = int(rng.integers(0, len(arr) - seq_len - 1))
                chunk = np.asarray(arr[i:i + seq_len + 1], dtype=np.int64)
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def source_token_counts(self) -> dict:
        """Tokens cached per source (after ``prepare``) — for mix inspection."""
        if self._arrays is None:
            self.prepare()
        return {n: int(len(a)) for n, a in self._arrays.items()}


__wiring__ = Wiring.STANDALONE
