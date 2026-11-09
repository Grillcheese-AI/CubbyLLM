"""validation/fact_store/dump_resume.py -- order 0055 step 1: the count resumes where a dropped runtime left it.

Pinned: a pass stopped after a checkpoint and run again gives the same counts, hash, end check, sample and side files,
byte for byte decompressed, as a pass never stopped (JSON and Turtle); a checkpoint for other parts, or one whose
segment is not on the Drive, is set aside and the pass starts over; nothing is left behind (no segments, an empty
stage, the checkpoint renamed done).
"""
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "validation", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_fact_store_dump as TD  # noqa: E402
import test_fact_store_ttl as TT  # noqa: E402
from fact_store import dump_count as C  # noqa: E402
from fact_store import dump_resume as R  # noqa: E402
from fact_store import dump_stream as S  # noqa: E402
from fact_store import ttl_count as T  # noqa: E402

N = 400
RUN_ONLY = ("wall_s", "resumed_from", "skip_s")


def _parts(tmp_path, kind):
    blob, fmt = (TD._dump(N)[1], C.Fmt) if kind == "json" else (TT._dump(N), T.Ttl)
    d = tmp_path / "parts"
    d.mkdir()
    for k in range(0, len(blob), 997):
        (d / f"part_{k // 997:05d}").write_bytes(blob[k:k + 997])
    (d / "manifest.json").write_text(json.dumps({"md5": hashlib.md5(blob).hexdigest()}))
    return d, fmt


def _files(out, fmt):
    return {n: gzip.open(out / n, "rb").read() for n in (fmt.sample, fmt.side) if n}


def _stop_after(monkeypatch, n):
    real, offs = R.save, []

    def save(out, state):
        real(out, state)
        offs.append(state["offset"])
        if len(offs) == n:
            raise RuntimeError("the runtime dropped")
    monkeypatch.setattr(R, "save", save)
    return real, offs


@pytest.mark.parametrize("kind", ["json", "ttl"])
def test_a_count_stopped_after_a_checkpoint_resumes_to_the_same_result(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(S, "BLOCK", 3000)                       # many blocks, so many checkpoints
    d, fmt = _parts(tmp_path, kind)
    full = C.run(str(d), str(tmp_path / "full"), workers=1, fmt=fmt, every=1e9)
    assert full["resumed_from"] == [] and full["hash_ok"] is True and full["ended_at_close"] is True

    out, stage = tmp_path / "cut", tmp_path / "stage"
    real, offs = _stop_after(monkeypatch, 3)
    with pytest.raises(RuntimeError):
        C.run(str(d), str(out), workers=1, fmt=fmt, stage=str(stage), every=0)
    gc.collect()                                                # the dropped run's open segment
    assert (out / R.CKPT).exists() and not (out / "counts.json").exists() and 0 < offs[-1]
    monkeypatch.setattr(R, "save", real)

    res = C.run(str(d), str(out), workers=1, fmt=fmt, stage=str(stage), every=0)
    assert res["resumed_from"] == [offs[-1]]
    assert {k: v for k, v in res.items() if k not in RUN_ONLY} == {k: v for k, v in full.items() if k not in RUN_ONLY}
    assert _files(out, fmt) == _files(tmp_path / "full", fmt)
    assert any(_files(out, fmt).values())                       # the sample is not empty: the check means something
    assert not list(out.glob("*.seg*")) and not list(stage.iterdir())
    assert (out / R.DONE).exists() and not (out / R.CKPT).exists()


@pytest.mark.parametrize("spoil", ["other_parts", "segment_gone"])
def test_a_checkpoint_that_does_not_match_is_set_aside_and_the_pass_starts_over(tmp_path, monkeypatch, spoil):
    monkeypatch.setattr(S, "BLOCK", 3000)
    monkeypatch.setattr(S, "WAITS", 0)
    d, fmt = _parts(tmp_path, "ttl")
    full = C.run(str(d), str(tmp_path / "full"), workers=1, fmt=fmt, every=1e9)
    out = tmp_path / "cut"
    real, _ = _stop_after(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        C.run(str(d), str(out), workers=1, fmt=fmt, every=0)
    gc.collect()
    monkeypatch.setattr(R, "save", real)
    if spoil == "other_parts":
        st = json.loads((out / R.CKPT).read_text())
        st["key"]["parts_bytes"] += 1
        (out / R.CKPT).write_text(json.dumps(st))
    else:
        (out / f"{fmt.sample}.seg00001").unlink()

    res = C.run(str(d), str(out), workers=1, fmt=fmt, every=1e9)
    assert res["resumed_from"] == [] and (out / R.STALE).exists()
    assert res["counts"] == full["counts"] and _files(out, fmt) == _files(tmp_path / "full", fmt)


def test_the_skip_never_reads_past_the_checkpoint_and_a_short_stream_stops(monkeypatch):
    import io
    monkeypatch.setattr(S, "BLOCK", 7)
    f = io.BytesIO(b"0123456789abcdefghij")
    R.skip(f, 12)
    assert f.read() == b"cdefghij"
    with pytest.raises(SystemExit):
        R.skip(io.BytesIO(b"short"), 12)
