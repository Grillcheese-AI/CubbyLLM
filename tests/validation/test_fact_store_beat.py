"""The dump passes' beat (order 0055): a line a minute saying what the pass is doing and where it is."""
import bz2
import time

from validation.fact_store import dump_beat as B
from validation.fact_store import dump_stream as S


def test_the_line_says_the_phase_the_part_the_bytes_and_the_end():
    live = {"phase": "counting", "part": 4, "parts": 113, "at": 958_300_000_000}
    s = B.line(live, 12.4)
    assert "counting" in s and "reading part 5 of 113" in s and "958.3 GB decompressed" in s and "12 min" in s
    live["fed_all"] = True
    assert "every part read" in B.line(live, 13)


def test_the_beat_prints_until_stopped(capsys):
    stop = B.beat({"phase": "skipping"}, every=0.05)
    time.sleep(0.3)
    stop.set()
    time.sleep(0.1)
    n = capsys.readouterr().out.count("skipping")
    time.sleep(0.2)
    assert n >= 2 and capsys.readouterr().out == ""


def test_the_stream_reports_its_part_and_its_end(tmp_path):
    data = bz2.compress(b"line\n" * 50_000)
    parts = []
    for i, k in enumerate(range(0, len(data), 1000)):
        p = tmp_path / f"part_{i:05d}"
        p.write_bytes(data[k:k + 1000])
        parts.append(str(p))
    live = {}
    f = S.stream(parts, 2, live=live)
    assert f.read(10) and live["parts"] == len(parts)
    while f.read(1 << 20):
        pass
    assert live["part"] == len(parts) - 1 and live["fed_all"] is True
