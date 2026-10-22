"""Build a plain-text sample for CB_TEXT from the domain-tagged Wikipedia jsonl
next to the checkpoints (``$CUBBY_DRIVE\\cubbyllm\\wikipedia\\*.jsonl``, fields
title/lang/domain/sim/text). A few articles per domain, shuffled, so a 4k-token
window spans several domains rather than one long article.

    python validation/prospection/make_text_sample.py [src_dir] [out_path] [per_domain]

Defaults: the Drive folder above, %TEMP%/cubby_wiki_sample.txt, 4 per domain.
Prints the path to pass as CB_TEXT. Never writes inside the repo.
"""
from __future__ import annotations

import pathlib as _pathlib  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[2]))      # the repo root, for:
from cubbyllm.local_paths import expand as _lp  # noqa: E402 -- local folders by name (order 0030)

import glob
import json
import os
import random
import sys


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else _lp(r"$CUBBY_DRIVE\cubbyllm\wikipedia")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("TEMP", "."), "cubby_wiki_sample.txt")
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    texts = []
    for f in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= per:
                    break
                texts.append(json.loads(line)["text"])
    random.Random(0).shuffle(texts)
    body = "\n\n".join(texts)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"{out}  ({len(texts)} articles, {len(body)} chars)")


if __name__ == "__main__":
    main()
