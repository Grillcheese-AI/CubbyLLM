"""The talk adapter's data and adapter-file layout, shared by the grilly2 trainer (local GPU) and the
torch trainer (Colab), so an adapter trained on either loads into the other.

- records: `standin/data/build_ground_sft.py`'s jsonl -> (prompt ids, answer ids + </s>); the loss mask
  covers the answer and its end-of-text only.
- batches: right-padded (the model is causal, so padding after a row cannot change it), bucketed by
  length so a batch wastes little.
- the adapter file: LoRA factors keyed by grilly2's module paths (`model.layers.{i}.mixer.qkv.lora_A.weight`),
  y = W x + (alpha / r) * B (A x) -- grilly.infer.lora's convention -- plus a json of rank, alpha, targets.

numpy only: nothing here imports torch or grilly.
"""
from __future__ import annotations

import json

import numpy as np

# grilly2 module name -> the torch CubbyModel attribute inside a backbone layer (export_base.py's map)
ATTN = {"mixer.qkv": ("mix", "qkv"), "mixer.o_proj": ("mix", "o")}
RECURRENT = {"mixer.gate": ("mix", "proj_g"), "mixer.value": ("mix", "proj_v"), "mixer.decay": ("mix", "proj_d")}
FFN = {"ffn.gate_proj": ("ffn", "g"), "ffn.up_proj": ("ffn", "u"), "ffn.down_proj": ("ffn", "o")}


def lora_targets(n_layers: int, attn_every: int) -> list[str]:
    """Every projection of every layer, in grilly2's names and layer order."""
    out = []
    for i in range(n_layers):
        local = ATTN if i % attn_every == 0 else RECURRENT
        out += [f"model.layers.{i}.{n}" for n in list(local) + list(FFN)]
    return out


def torch_attr(target: str) -> tuple[int, str, str]:
    """'model.layers.3.mixer.gate' -> (3, 'mix', 'proj_g'): where that projection lives in the torch model."""
    parts = target.split(".")
    i, local = int(parts[2]), ".".join(parts[3:])
    group, attr = {**ATTN, **RECURRENT, **FFN}[local]
    return i, group, attr


def encode_records(path, tk, eos, split, max_len, families=None):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] != split or (families and r["family"] not in families):
            continue
        p = tk.encode(r["prompt"]).ids
        t = tk.encode(" " + r["answer"]).ids + [eos]
        if len(p) + len(t) > max_len:
            continue
        rows.append({"p": p, "t": t, "family": r["family"], "id": r["id"]})
    return rows


def make_batch(rows):
    """Right-padded (B, T) inputs, targets and a mask on the answer tokens. `keep` covers every answer
    position of the batch, counted from the end."""
    T = max(len(r["p"]) + len(r["t"]) for r in rows) - 1
    x = np.zeros((len(rows), T), np.int64)
    y = np.zeros((len(rows), T), np.int64)
    m = np.zeros((len(rows), T), np.float32)
    for i, r in enumerate(rows):
        ids = r["p"] + r["t"]
        n = len(ids) - 1
        x[i, :n], y[i, :n] = ids[:-1], ids[1:]
        m[i, len(r["p"]) - 1:n] = 1.0
    keep = T - min(len(r["p"]) for r in rows) + 1
    return x, y, m, keep


def batches(rows, batch, rng):
    """Length-bucketed: sort by length in chunks of 50 batches, shuffle the batches."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    out = []
    span = batch * 50
    for s in range(0, len(idx), span):
        chunk = sorted(idx[s:s + span], key=lambda i: len(rows[i]["p"]) + len(rows[i]["t"]))
        out += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(out)
    return out
