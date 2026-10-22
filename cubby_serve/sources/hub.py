"""cubby_serve/sources/hub.py -- a file from the Hugging Face Hub, downloaded (the wiki world's parquet export; moved
from ``cubbyllm/worlds/wikikg.ensure_data`` in order 0031, verbatim: the package takes it as its ``fetch``).

    python -m cubby_serve.sources.hub wikikg [triplets] [paths]     the wiki world's export into its cache

Wired: WIRED (the server's ``--wiki auto``: ``build_serve`` hands ``download`` to ``wikikg.ensure_data``).
"""
from __future__ import annotations

import sys
import urllib.request

__wiring__ = "WIRED"


def download(url: str, dst: str, timeout: int = 900) -> None:
    """``url`` streamed into ``dst``, a megabyte at a time."""
    req = urllib.request.Request(url, headers={"User-Agent": "cubbyllm/standin"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "wikikg":
        print(__doc__.split("\n\n")[1])
        return 2
    from cubbyllm.worlds import wikikg as wk
    wk.ensure_data(tuple(args[1:]) or ("triplets",), fetch=download)
    print("cached in", wk.DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
