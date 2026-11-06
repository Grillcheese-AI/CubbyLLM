"""validation/fact_store/dump_fetch.py and dump_count.py -- order 0055 step 1, on a dump made here (no network).

Pinned: the download is resumable by part and its sha1 checked against Wikimedia's; the count reads the parts as one
compressed stream however they are cut, counts what the order lists (statements by rank, time qualifiers, references,
scholarly articles), and samples by a hash of the id (the same ids whatever the order of the dump), never the first.
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fact_store import dump_count as C  # noqa: E402
from fact_store import dump_fetch as F  # noqa: E402


def _st(pid, val, rank="normal", quals=None, refs=0):
    dv = {"type": "wikibase-entityid", "value": {"id": val}} if val.startswith("Q") else {"type": "string", "value": val}
    st = {"mainsnak": {"snaktype": "value", "property": pid, "datatype": "wikibase-item" if val.startswith("Q") else "string",
                       "datavalue": dv}, "rank": rank, "type": "statement"}
    if quals:
        st["qualifiers"] = {q: [{"snaktype": "value"}] for q in quals}
    if refs:
        st["references"] = [{"hash": hashlib.sha1(f"{pid}{val}{k}".encode()).hexdigest(), "snaks": {"P248": [{}]}}
                            for k in range(refs)]
    return st


def _dump(n=60):
    ents = []
    for i in range(1, n + 1):
        claims = {"P31": [_st("P31", "Q13442814" if i % 10 == 0 else "Q5")],
                  "P39": [_st("P39", "Q11696", quals=["P580", "P642"], refs=1)],
                  "P1448": [_st("P1448", f"name {i}", rank="deprecated" if i % 7 == 0 else "normal")]}
        ents.append({"type": "item", "id": f"Q{i}", "labels": {"en": {"language": "en", "value": f"x{i}"}},
                     "aliases": {"fr": [{"value": "a"}, {"value": "b"}]}, "claims": claims,
                     "sitelinks": {"enwiki": {"title": f"X{i}"}} if i % 2 else {}})
    text = "[\n" + ",\n".join(json.dumps(e) for e in ents) + "\n]\n"
    return ents, bz2.compress(text.encode())


def test_the_count_reads_the_parts_as_one_stream_and_counts_what_the_order_lists(tmp_path):
    ents, blob = _dump()
    for name, cut in (("a", 97), ("b", 1000)):
        d = tmp_path / name
        d.mkdir()
        for k in range(0, len(blob), cut):
            (d / f"part_{k // cut:05d}").write_bytes(blob[k:k + cut])
        res = C.run(str(d), str(tmp_path / f"out_{name}"), workers=2)
        c = res["counts"]
        assert c["entities:item"] == 60 and c["statements:item"] == 180 and c["rank:deprecated"] == 8
        assert c["items:scholarly"] == 6 and c["statements:scholarly"] == 18
        assert c["qualifier_snaks"] == 120 and c["qualifier_snaks:time"] == 60 and c["reference_blocks"] == 60
        assert c["labels:en"] == 60 and c["aliases:fr"] == 120 and c["sitelinks:enwiki"] == 30
        assert res["references_distinct_est"] == 1                              # 60 blocks, one reference: deduplicated
        got = [json.loads(x)["id"] for x in gzip.open(tmp_path / f"out_{name}" / "sample_1pct.jsonl.gz", "rt")]
        assert sorted(got) == sorted(e["id"] for e in ents if C.in_sample(e["id"]))


def test_the_sample_is_a_hash_of_the_id_not_the_first_ids():
    picked = [f"Q{i}" for i in range(1, 20001) if C.in_sample(f"Q{i}")]
    assert 150 < len(picked) < 250 and picked[0] != "Q1" and int(picked[-1][1:]) > 19000


class FakeServer:
    def __init__(self, blob):
        self.blob, self.ranges = blob, []

    def __call__(self, url, headers=None, timeout=120):
        lo, hi = map(int, headers["Range"].split("=")[1].split("-"))
        self.ranges.append(lo)
        return io.BytesIO(self.blob[lo:hi + 1])


def test_the_download_resumes_by_part_and_checks_the_sha1(tmp_path, monkeypatch):
    blob = bytes(range(256)) * 40
    monkeypatch.setattr(F, "PART", 1000)
    info = {"url": "u", "size": len(blob), "sha1": hashlib.sha1(blob).hexdigest()}
    (tmp_path / "part_00002").write_bytes(blob[2000:3000])                      # one part already in
    srv = FakeServer(blob)
    monkeypatch.setattr(F, "spans", lambda size, part=1000: [(s, min(s + 1000, size) - 1) for s in range(0, size, 1000)])
    man = F.run(str(tmp_path), info, workers=2, opener=srv)
    assert man["sha1_ok"] and man["parts"] == 11 and 2000 not in srv.ranges
    (tmp_path / "part_00003").write_bytes(b"x" * 1000)                          # a part gone bad
    srv2 = FakeServer(blob)
    with pytest.raises(SystemExit):
        F.run(str(tmp_path), info, workers=1, opener=srv2)                     # the size matches: the sha1 catches it


def test_latest_reads_the_index_and_wikimedias_sha1():
    pages = {f"{F.BASE}/": '<a href="20261026/">20261026/</a> <a href="20261102/">20261102/</a>',
             f"{F.BASE}/20261102/": "",
             f"{F.BASE}/20261026/": '<a href="wikidata-20261026-all.json.bz2">x</a>  27-Oct-2026 16:38  103565615653',
             f"{F.BASE}/20261026/wikidata-20261026-sha1sums.txt": "abc123  wikidata-20261026-all.json.bz2\n"}
    info = F.latest(opener=lambda u, h=None, timeout=120: io.BytesIO(pages[u].encode()))
    assert info["date"] == "20261026" and info["size"] == 103565615653 and info["sha1"] == "abc123"


def _parts(tmp_path, blob, cut=97, name="p"):
    d = tmp_path / name
    d.mkdir()
    for k in range(0, len(blob), cut):
        (d / f"part_{k // cut:05d}").write_bytes(blob[k:k + cut])
    return d


def test_the_count_checks_the_parts_sha1_and_the_dumps_end(tmp_path):
    _, blob = _dump()
    d = _parts(tmp_path, blob)
    (d / "manifest.json").write_text(json.dumps({"parts": len(list(d.glob("part_*"))),
                                                 "sha1": hashlib.sha1(blob).hexdigest()}))
    res = C.run(str(d), str(tmp_path / "out"), workers=1)
    assert res["sha1_ok"] is True and res["ended_at_close"] is True


def test_a_missing_or_cut_part_stops_the_count_instead_of_passing_for_the_end(tmp_path, monkeypatch):
    from fact_store import dump_stream as S
    monkeypatch.setattr(S, "WAITS", 0)
    _, blob = _dump()
    d = _parts(tmp_path, blob, name="gap")
    (d / "part_00002").rename(d / "part_00002.part")                              # a part the mount does not show
    with pytest.raises(SystemExit):
        C.run(str(d), str(tmp_path / "o1"), workers=1)
    d2 = _parts(tmp_path, blob[:len(blob) - 40], name="cut")                    # the last part short
    with pytest.raises((EOFError, SystemExit, OSError)):
        C.run(str(d2), str(tmp_path / "o2"), workers=1)
    assert not (tmp_path / "o1" / "counts.json").exists() and not (tmp_path / "o2" / "counts.json").exists()


def test_lbzip2_failing_stops_the_pass():
    from fact_store import dump_stream as S

    class Dec:
        stdout, returncode = io.BytesIO(b"[\n{}\n"), 2

        def wait(self):
            return self.returncode

    class Done:
        def join(self):
            pass
    r = S._Checked(Dec(), Done(), [])
    assert r.read(4) == b"[\n{}"
    with pytest.raises(SystemExit):
        while r.read(4):
            pass
