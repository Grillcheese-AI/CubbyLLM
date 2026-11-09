"""validation/fact_store/dump_stream.blocks' fallback cut -- order 0055 step 1.

The challenge dump's count stalled twice at about 958 GB: with no ``data:`` header to cut at, the block buffer grew
without end. Pinned: a stretch with no entity header is cut at blank lines, and counts the same as when every cut
falls at a header (entities, statements, redirects, the side file, the sample); the cut never loses a byte; a buffer
that never finds a cut stops the pass instead of hanging; a sampled entity followed by such a stretch is capped.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "validation", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_fact_store_ttl as TT  # noqa: E402
from fact_store import dump_count as C  # noqa: E402
from fact_store import dump_stream as S  # noqa: E402
from fact_store import ttl_count as T  # noqa: E402

TAIL = 3000


def _tail() -> str:
    """A stretch with no ``data:`` header, as the dump's end may hold: redirects, each its own subject group."""
    return "".join(f"\nwd:Q{900000 + i} owl:sameAs wd:Q{i % 60 + 1} .\n" for i in range(TAIL))


def _parts(tmp_path, text: str):
    blob = bz2.compress(text.encode())
    d = tmp_path / "parts"
    d.mkdir()
    for k in range(0, len(blob), 997):
        (d / f"part_{k // 997:05d}").write_bytes(blob[k:k + 997])
    (d / "manifest.json").write_text(json.dumps({"md5": hashlib.md5(blob).hexdigest()}))
    return d


def test_a_stretch_with_no_entity_header_is_cut_at_blank_lines_and_counts_the_same(tmp_path, monkeypatch):
    d = _parts(tmp_path, TT.HEAD + "".join(TT._item(i) for i in range(1, 61)) + TT.PROPS + _tail())
    whole = C.run(str(d), str(tmp_path / "whole"), workers=1, fmt=T.Ttl, every=1e9)   # 32 MB reads: no fallback
    monkeypatch.setattr(S, "BLOCK", 3000)                                           # the tail spans many reads
    cut = C.run(str(d), str(tmp_path / "cut"), workers=1, fmt=T.Ttl, every=1e9)
    assert whole["fallback_cuts"] == 0 and cut["fallback_cuts"] > 5
    assert len(cut["stretches_kept"]) == 3 and all((tmp_path / "cut" / n).exists() for n in cut["stretches_kept"])
    c = cut["counts"]
    assert c["redirects"] == TAIL + 1 and c["entities:item"] == 60 and c["entities:property"] == 2
    assert c["headerless_bytes"] > TAIL * 20                                  # the first stretch block holds P1448 too
    same = ("headerless_bytes",)
    assert {k: v for k, v in c.items() if k not in same} == {k: v for k, v in whole["counts"].items() if k not in same}
    assert cut["hash_ok"] and cut["ended_at_close"]
    for name in ("side.tsv.gz", "sample_1pct.ttl.gz"):
        assert gzip.open(tmp_path / "cut" / name).read() == gzip.open(tmp_path / "whole" / name).read()


def test_the_fallback_cut_loses_no_byte():
    text = b"data:Q1 a x .\n\n" + b"".join(b"wd:Q%d owl:sameAs wd:Q1 .\n\n" % i for i in range(400)) + b"data:Q2 a x ."
    old = S.BLOCK
    try:
        S.BLOCK = 64
        got = list(S.blocks(io.BytesIO(text), b"\ndata:", b"\n\n"))
    finally:
        S.BLOCK = old
    assert len(got) > 10 and b"\n".join(got) == text


def test_a_buffer_that_never_finds_a_cut_stops_the_pass(monkeypatch):
    monkeypatch.setattr(S, "BLOCK", 16)
    heard = []
    with pytest.raises(SystemExit):
        list(S.blocks(io.BytesIO(b"x" * 4000), b"\ndata:", b"\n\n", note=lambda k, at, d: heard.append(k)))
    assert heard == ["stopped"]


def test_a_sampled_entity_followed_by_a_headerless_stretch_is_capped():
    i = next(i for i in range(1, 5000) if C.in_sample(f"Q{i}"))
    item = TT._item(i).lstrip("\n")
    c, _, _, sample, _ = T.work((item + _tail()).encode(), cap=len(item) + 10)
    assert c["sample_capped"] == 1 and len(sample) == 1
    assert sample[0].startswith(f"data:Q{i} ".encode()) and b"owl:sameAs" not in sample[0]
    c, _, _, sample, _ = T.work(item.encode())
    assert "sample_capped" not in c and sample[0] == item.encode()
