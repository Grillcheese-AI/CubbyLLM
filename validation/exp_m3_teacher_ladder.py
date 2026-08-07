"""exp_m3_teacher_ladder — which teacher makes the best block-space word table?

The bge-m3 arm proved sentence-embedding strength is the wrong axis: the
table harvests STATIC PER-WORD vector quality, so the candidates are models
that optimize exactly that. Every arm distills into the identical recipe
(same 30,275-word vocab, same QR seed, in-vocab anisotropy, 0.5 signal
blend, same IDF) and runs the same two production screens.

Already measured (exp_m3_potion_baseline / exp_m3_bgem3_block_arm):
  routing macro: bge-m3->block 0.566 | MiniLM->block(v1) 0.595 |
                 potion-8M->block(v2) 0.646 | potion-8M native 0.722 |
                 MiniLM live 0.797
  NYT exact:     potion-8M native 0.044 | MiniLM->block 0.064 |
                 potion-8M->block 0.069 | bge-m3->block 0.070

Arms here: potion-base-32M, potion-retrieval-32M, potion-multilingual-128M
(whose own teacher IS bge-m3 — Tokenlearn-compiled static form of the model
that failed as a direct teacher), and sentence-transformers'
static-retrieval-mrl-en-v1. Standalone; never imported by cubbyllm/.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_m3_domain_routing import TABLE, _load_semantic_words  # noqa: E402
from exp_m3_nyt_years import load_nyt  # noqa: E402
from exp_m3_potion_baseline import (  # noqa: E402
    routing_arm, sample_domain_split, year_arm)

ARMS = [
    ("m2v:minishlab/potion-base-32M", "m2v"),
    ("m2v:minishlab/potion-retrieval-32M", "m2v"),
    ("m2v:minishlab/potion-multilingual-128M", "m2v"),
    ("st:sentence-transformers/static-retrieval-mrl-en-v1", "st"),
]


def main() -> None:
    import platform
    print(f"python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")

    sw = _load_semantic_words()
    enc = sw.FastWordEncoder.from_npz(TABLE)
    vocab = sorted(enc._vec.keys())
    ex_t, ex_d, ch_t, ch_d, _ = sample_domain_split(40, 40, 8, 5)
    texts, years, _ = load_nyt(100)
    print(f"{len(vocab)} vocab words | {len(ch_t)} routing challenges | "
          f"{len(texts)} NYT texts")
    print("ladder so far (routing / NYT-exact): bge-m3 0.566/0.070 | "
          "MiniLM 0.595/0.064 | potion-8M 0.646/0.069\n")

    out: dict = {"results": {}}
    for teacher, kind in ARMS:
        name = teacher.split("/")[-1]
        try:
            t0 = time.perf_counter()
            tid = teacher.partition(":")[2]
            if kind == "m2v":
                from model2vec import StaticModel
                model = StaticModel.from_pretrained(tid)
                E = np.asarray(model.encode(vocab, show_progress_bar=False),
                               dtype=np.float32)
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(tid)
                E = np.asarray(model.encode(vocab, batch_size=2048,
                                            show_progress_bar=False),
                               dtype=np.float32)
            print(f"-- {name}: dim {E.shape[1]}, embed "
                  f"{time.perf_counter() - t0:.0f}s")

            def embed_fn(ws, E=E, vocab=vocab):
                assert ws == vocab
                return E

            t = sw.FastWordEncoder.build(vocab, embed_fn, enc._idf,
                                         enc.default_idf, k=enc.k, l=enc.l,
                                         meta={"teacher": teacher})

            def e(txts, t=t):
                return np.stack([t.encode(x).ravel()
                                 for x in txts]).astype(np.float32)

            r = routing_arm(name, e, ex_t, ex_d, ch_t, ch_d)
            y = year_arm(name, e, texts, years)
            out["results"][teacher] = {"routing": r, "nyt": y,
                                       "teacher_dim": int(E.shape[1])}
        except Exception as ex:
            print(f"-- {name}: FAILED ({ex})")
            out["results"][teacher] = {"error": str(ex)}

    logs = ROOT / "validation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "exp_m3_teacher_ladder.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {logs / 'exp_m3_teacher_ladder.json'}")


if __name__ == "__main__":
    main()
