"""Pins for the rows-API fetcher (standin/data/hf_rows.py): a trimmed page is not the end of the split, pages are
cached by offset, a 429 backs off. No network: urlopen is faked."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import hf_rows  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_api(pages: dict[int, list[dict]], fail_first: int = 0):
    """urlopen that serves `pages` keyed by offset, trimming nothing itself; the first `fail_first` calls 429."""
    calls = {"n": 0, "offsets": []}

    def urlopen(url, timeout=0):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        off = int(url.split("offset=")[1].split("&")[0])
        calls["offsets"].append(off)
        return _Resp(json.dumps({"rows": [{"row": r} for r in pages.get(off, [])]}).encode())
    return urlopen, calls


def test_a_trimmed_page_advances_by_the_rows_returned(tmp_path, monkeypatch):
    # the API returns 3 rows for a 100-row request (large payload), then 3 more at offset 3, then nothing at 6
    pages = {0: [{"i": 0}, {"i": 1}, {"i": 2}], 3: [{"i": 3}, {"i": 4}, {"i": 5}], 6: []}
    urlopen, calls = _fake_api(pages)
    monkeypatch.setattr(hf_rows.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(hf_rows.time, "sleep", lambda s: None)
    rows = hf_rows.fetch_rows("x/y", "default", "train", 10, cache=str(tmp_path), verbose=False)
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 4, 5], "a short page is not the end of the split"
    assert calls["offsets"] == [0, 3, 6]
    # cached: a second call makes no request
    calls["offsets"].clear()
    again = hf_rows.fetch_rows("x/y", "default", "train", 10, cache=str(tmp_path), verbose=False)
    assert [r["i"] for r in again] == [0, 1, 2, 3, 4, 5] and calls["offsets"] == []
    # n caps the result
    assert len(hf_rows.fetch_rows("x/y", "default", "train", 4, cache=str(tmp_path), verbose=False)) == 4


def test_a_429_backs_off_and_retries(tmp_path, monkeypatch):
    pages = {0: [{"i": 0}], 1: []}
    urlopen, calls = _fake_api(pages, fail_first=2)
    slept = []
    monkeypatch.setattr(hf_rows.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(hf_rows.time, "sleep", lambda s: slept.append(s))
    rows = hf_rows.fetch_rows("x/y", "default", "train", 5, cache=str(tmp_path), verbose=False)
    assert [r["i"] for r in rows] == [0] and calls["n"] >= 3
    assert slept and slept[0] == 15.0 and slept[1] == 30.0, "a 429 waits 15 s x attempt"
