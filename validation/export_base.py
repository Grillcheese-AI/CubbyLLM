"""Export a trained base (``train_base.py``'s checkpoint or final export) to
the form grilly2's native model loads: ``config.json`` + ``model.safetensors``
under ``grilly.infer.models.CubbyForCausalLM``'s parameter names.

Why an export rather than grilly2 reading the checkpoint: the checkpoint is a
pickle of this repo's classes and a flat parameter list, and grilly2 must load
a model with nothing of CubbyLLM installed. The names are mapped by walking
the rebuilt torch model's own modules, never by list position, so two
same-shaped tensors (MinGRU's three D x D projections, the two norms of a
layer) cannot be swapped silently.

    python validation/export_base.py --ckpt <base450m_final.pt> --out <dir>

Needs torch and ``safetensors`` (grilly2's environment has both). Writes
float32, as trained weights are held.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from train_base import load_base  # noqa: E402


def named_tensors(model) -> dict:
    """grilly2's name -> tensor, from the torch model's modules."""
    out = {}
    bb, gen, router = model.backbone, model.memory.generator, model.context_source
    out["model.embed_tokens.weight"] = model.embedding.core.weight
    for i in range(len(bb.mix)):
        pre = f"model.layers.{i}."
        out[pre + "mixer_norm.weight"] = bb.n1[i].w
        mix = bb.mix[i]
        if bb.is_attn[i]:
            out[pre + "mixer.qkv.weight"] = mix.qkv.weight
            out[pre + "mixer.o_proj.weight"] = mix.o.weight
        else:
            for mine, theirs in (("gate", mix.proj_g), ("value", mix.proj_v), ("decay", mix.proj_d)):
                out[pre + f"mixer.{mine}.weight"] = theirs.weight
                out[pre + f"mixer.{mine}.bias"] = theirs.bias
        out[pre + "ffn_norm.weight"] = bb.n2[i].w
        out[pre + "ffn.gate_proj.weight"] = bb.ffn[i].g.weight
        out[pre + "ffn.up_proj.weight"] = bb.ffn[i].u.weight
        out[pre + "ffn.down_proj.weight"] = bb.ffn[i].o.weight
    r = "model.router."
    out[r + "proj.weight"], out[r + "proj.bias"] = router.router[0].weight, router.router[0].bias
    out[r + "out.weight"], out[r + "out.bias"] = router.router[2].weight, router.router[2].bias
    out[r + "slots"] = router.slots.weight
    g = "model.adapter."
    out[g + "ctx_proj.weight"], out[g + "ctx_proj.bias"] = gen.ctx_proj.weight, gen.ctx_proj.bias
    out[g + "ctx_norm.weight"], out[g + "ctx_norm.bias"] = gen.ctx_norm.weight, gen.ctx_norm.bias
    out[g + "layer_emb"] = gen.layer_emb
    out[g + "mix_in.weight"], out[g + "mix_in.bias"] = gen.mix[0].weight, gen.mix[0].bias
    out[g + "mix_out.weight"], out[g + "mix_out.bias"] = gen.mix[2].weight, gen.mix[2].bias
    out[g + "A_basis"], out[g + "B_basis"] = gen.A_basis, gen.B_basis
    out["lm_head.weight"] = model.head._codes()
    return {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in out.items()}


def grilly_config(meta: dict, model) -> dict:
    """``CubbyConfig``'s fields, from the checkpoint's meta and the modules."""
    gen, router = model.memory.generator, model.context_source
    ln = gen.ctx_norm
    rms_eps = 1e-6                       # _RMSNorm's constant (backbone/mingru.py)
    if not model.head.learnable or model.head.temperature != 1.0:
        raise SystemExit("only the learnable, temperature-1 head is a plain linear projection")
    return {
        "model_type": "cubby",
        "vocab_size": meta["vocab"], "vocab_real": meta["vocab_real"],
        "hidden_size": meta["D"], "num_hidden_layers": meta["L"],
        "num_attention_heads": meta["heads"], "window": meta["window"],
        "attn_every": meta["attn_every"], "ctx_dim": meta["ctx"], "n_slots": meta["slots"],
        "router_hidden": router.router[0].out_features,
        "gen_hidden": gen.ctx_proj.out_features, "gen_basis": meta["gen_basis"],
        "gen_rank": meta["gen_rank"], "gen_layers": gen.n_layers,
        "ffn_mult": model.backbone.ffn[0].g.out_features // meta["D"],
        "rope_theta": 10000.0, "rms_norm_eps": rms_eps, "layer_norm_eps": ln.eps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="", help="copied into the export as tokenizer.json, so it travels with the weights")
    args = ap.parse_args()
    from safetensors.torch import save_file

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    if meta.get("mem_every", 0):
        raise SystemExit("episodic memory layers (mem_every > 0) have no grilly2 form yet")
    model = load_base(args.ckpt, "cpu")
    tensors = named_tensors(model)
    cfg = grilly_config(meta, model)
    cfg["source"] = {"checkpoint": os.path.basename(args.ckpt), "step": ck.get("step"),
                     "tokens": ck.get("tokens"), "meta": meta}
    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    if args.tokenizer:
        import shutil
        shutil.copyfile(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    n = sum(t.numel() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {n / 1e6:.1f}M parameters -> {args.out}")


if __name__ == "__main__":
    main()
