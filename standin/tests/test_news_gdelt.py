"""Order 0024's GDELT ingest on synthetic 15-minute files (no network): an event touching Canada is kept, one naming a
person a world holds is kept, the rest dropped; mentions follow their kept events; a missed hour is caught up; a file
taken is never taken twice; the translingual stream's lag holds the newest stamp back.
Run: python -m pytest standin/tests/test_news_gdelt.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import news_gdelt as g  # noqa: E402


def _event(eid, a1="", a1c="", a2="", geo_cc=""):
    r = [""] * 61
    r[0], r[1], r[6], r[7], r[16], r[26], r[53], r[60] = eid, "20261022", a1, a1c, a2, "046", geo_cc, "https://x.example/a"
    return r


def _mention(eid):
    r = [""] * 16
    r[0], r[4], r[5] = eid, "x.example", f"https://x.example/{eid}"
    return r


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("f.CSV", "\n".join("\t".join(r) for r in rows))
    return buf.getvalue()


def _site(stamps, newest_en, newest_tr=None, calls=None):
    files = {}
    for i, st in enumerate(stamps):
        files[f"{st}.export.CSV.zip"] = _zip([_event(f"{i}1", "OTTAWA", "CAN"), _event(f"{i}2", "JUSTIN TRUDEAU", "", "POLICE"),
                                              _event(f"{i}3", "UNITED KINGDOM", "GBR"), _event(f"{i}4", geo_cc="CA")])
        files[f"{st}.mentions.CSV.zip"] = _zip([_mention(f"{i}1"), _mention(f"{i}3"), _mention("01")])
        files[f"{st}.translation.export.CSV.zip"] = _zip([_event(f"{i}9", "MONTREAL", "CAN")])
        files[f"{st}.translation.mentions.CSV.zip"] = _zip([_mention(f"{i}9")])
    files["lastupdate.txt"] = f"1 x {g.BASE}{newest_en}.export.CSV.zip".encode()
    files["lastupdate-translation.txt"] = f"1 x {g.BASE}{newest_tr or newest_en}.translation.export.CSV.zip".encode()

    def fetch(url, timeout=20):
        if calls is not None:
            calls.append(url)
        n = url.rsplit("/", 1)[1]
        return (200, files[n]) if n in files else (404, b"")
    return fetch


def known(name):
    return name.lower() == "justin trudeau"


def test_keep_rule():
    assert g.keep(g.pick(_event("1", "OTTAWA", "CAN"), g.EVENT))
    assert g.keep(g.pick(_event("1", geo_cc="CA"), g.EVENT))
    assert not g.keep(g.pick(_event("1", "UNITED KINGDOM", "GBR"), g.EVENT), known)
    assert g.keep(g.pick(_event("1", "JUSTIN TRUDEAU"), g.EVENT), known)
    assert not g.keep(g.pick(_event("1", "JUSTIN TRUDEAU"), g.EVENT))      # no world: Canada only


def test_due_catches_up_and_caps():
    t = dt.datetime(2026, 10, 22, 13, 45)
    assert g.due("20261022124500", t) == ["20261022130000", "20261022131500", "20261022133000", "20261022134500"]
    assert g.due(None, t) == ["20261022134500"]
    assert len(g.due("20261001000000", t)) == g.MAX_BACK and g.due("20261001000000", t)[-1] == "20261022134500"


def test_poll_keeps_filters_and_never_retakes(tmp_path):
    stamps = ["20261022130000", "20261022131500"]
    calls = []
    gd = g.Gdelt(tmp_path, _site(stamps, stamps[-1], calls=calls), known)
    (gd.root / "_taken.txt").write_text("20261022124500 {}\n", encoding="utf-8")
    row = gd.poll()
    assert row["files"] == 2 and row["new"] == 2 * 4 and row["missing"] == []      # Ottawa, Trudeau, CA place, Montreal
    evs = [json.loads(l) for l in (tmp_path / "2026-10-22.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "UNITED KINGDOM" not in {e["a1_name"] for e in evs}
    assert {e["stream"] for e in evs} == {"en", "translation"}
    mns = (tmp_path / "2026-10-22.mentions.jsonl").read_text(encoding="utf-8").splitlines()
    ids = {json.loads(l)["id"] for l in mns}
    assert "03" not in ids and "13" not in ids and {"01", "11", "09", "19"} <= ids
    n_calls = len(calls)
    again = g.Gdelt(tmp_path, _site(stamps, stamps[-1], calls=calls), known).poll()
    assert again["files"] == 0 and again["new"] == 0 and len(calls) - n_calls == 2      # the two lastupdate reads only
    assert not any(".gkg." in c for c in calls) and not any("x.example" in c for c in calls)


def test_translation_lag_holds_the_newest_back(tmp_path):
    stamps = ["20261022130000", "20261022131500"]
    gd = g.Gdelt(tmp_path, _site(stamps, stamps[1], newest_tr=stamps[0]), known)
    (gd.root / "_taken.txt").write_text("20261022124500 {}\n", encoding="utf-8")
    assert gd.poll()["files"] == 1 and gd.last() == stamps[0]
