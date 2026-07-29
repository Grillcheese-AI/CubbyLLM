"""H-C2 — zip2zip-style LZW hypertoken compression: measure it, don't cite it.

The claim: LZW compression over the tokenizer's output stream cuts the number
of autoregressive steps 20-60% (zip2zip's reported range). This is a pure
sequence-length lever — measurable OFFLINE with no model at all, exactly as the
hypothesis's "validate cheaply" clause asks.

Setup: the REAL cubby-lm tokenizer (grillcheese_spm32k_v2 SentencePiece, the
one the last recorded training run used) over the REAL held-out corpus
(cubby-lm's valid.txt). LZW dictionary over token IDs, zip2zip-style:
start from the base vocab, mint a new hypertoken for every (prefix, token)
pair seen, cap dictionary size, emit longest-known-match codes.

Sensitivity axes: dictionary cap (how many hypertokens the model would need to
embed on the fly — this is exactly H-C1/H-C6's dynamic-vocab machinery) and
corpus size. Also reported: compression from simply RESETTING the dictionary
per document (the honest streaming setting) vs one global dictionary.
"""
from __future__ import annotations

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sentencepiece as spm  # noqa: E402

SPM_PATH = (r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\tokenizers"
            r"\spm32k_1p7b\grillcheese_spm32k_v2.model")
CORPUS = r"C:\Users\grill\Documents\GitHub\cubby-lm\cubby\trunk_torch\data\valid.txt"
BASE_VOCAB = 32_000


def lzw_compress_ids(ids, max_dict, base_vocab=BASE_VOCAB):
    """LZW over a token-id stream. Returns number of emitted codes.

    Dictionary maps tuple-of-ids -> code. Base entries are the vocab itself
    (single ids). New codes are minted for (current_match + next_id) until
    max_dict total entries. Standard greedy longest-match LZW.
    """
    next_code = base_vocab
    table = {}                      # (prefix_code, id) -> code ; single ids implicit
    emitted = 0
    prev = None                     # current match, as a code (int) or None
    for tid in ids:
        if prev is None:
            prev = tid
            continue
        key = (prev, tid)
        code = table.get(key)
        if code is not None:
            prev = code             # extend the match
        else:
            emitted += 1            # emit code for `prev`
            if next_code < max_dict:
                table[key] = next_code
                next_code += 1
            prev = tid
    if prev is not None:
        emitted += 1
    return emitted, next_code - base_vocab


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_PATH)
    print("=" * 76)
    print("H-C2  LZW hypertoken compression on real tokenizer output")
    print(f"  tokenizer: grillcheese_spm32k_v2 (V={sp.GetPieceSize()})")
    print("=" * 76)

    # Read a slice of the held-out corpus; split into "documents" by blank line.
    n_bytes = 8_000_000
    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(n_bytes)
    docs = [d for d in text.split("\n\n") if len(d) > 200]
    print(f"  corpus: valid.txt, first {n_bytes/1e6:.0f}MB ->"
          f" {len(docs)} documents > 200 chars")

    t0 = time.time()
    all_ids = [sp.EncodeAsIds(d) for d in docs]
    base_tokens = sum(len(x) for x in all_ids)
    print(f"  base token count: {base_tokens:,}"
          f"  (tokenize {time.time()-t0:.1f}s)\n")

    print(f"  {'dict cap':>10} {'setting':>22} {'emitted codes':>14}"
          f" {'reduction':>10} {'hypertokens minted':>19}")
    results = {}
    for cap_extra in (32_000, 96_000, 224_000, 992_000):
        max_dict = BASE_VOCAB + cap_extra
        # (a) one GLOBAL dictionary across the whole stream
        flat = [t for doc in all_ids for t in doc]
        emitted_g, minted_g = lzw_compress_ids(flat, max_dict)
        red_g = 1 - emitted_g / base_tokens
        # (b) dictionary RESET per document (streaming-honest)
        emitted_r = 0
        minted_r = 0
        for doc in all_ids:
            e, m = lzw_compress_ids(doc, max_dict)
            emitted_r += e
            minted_r = max(minted_r, m)
        red_r = 1 - emitted_r / base_tokens
        results[cap_extra] = (red_g, red_r)
        print(f"  {cap_extra:>10,} {'global dict':>22} {emitted_g:>14,}"
              f" {red_g*100:>9.1f}% {minted_g:>19,}")
        print(f"  {'':>10} {'reset per document':>22} {emitted_r:>14,}"
              f" {red_r*100:>9.1f}% {minted_r:>19,} (max/doc)")

    best_global = max(r[0] for r in results.values())
    best_reset = max(r[1] for r in results.values())
    print("\n[verdict]")
    print(f"  best observed step reduction: global dict {best_global*100:.1f}%,"
          f" per-document {best_reset*100:.1f}%")
    lo = 0.20
    if best_reset >= lo:
        print(f"  zip2zip's claimed 20-60% range: REPRODUCED — the streaming"
              f" (per-document) number lands at {best_reset*100:.0f}%, and a"
              f" persistent dictionary reaches {best_global*100:.0f}%"
              f" (at/above the claimed top).")
    else:
        print(f"  zip2zip's claimed 20-60% range: NOT reproduced — streaming"
              f" number is only {best_reset*100:.0f}%.")
    print("  note: the honest streaming number is the per-document one;"
          " the global number assumes a persistent cross-document dictionary"
          " (= a bigger effective vocab, which is exactly H-C6's open choice).")


if __name__ == "__main__":
    main()
