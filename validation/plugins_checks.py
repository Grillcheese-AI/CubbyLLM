"""plugins_checks -- order 0028's measured lines on this machine, the four proof plugins on their real data.

1. **Startup**: 50 dummy plugins (each with a 200 KB payload and code that would stop the process if imported):
   ``discover`` reads their front matter alone; the time it adds (median of 5) against the 200 ms line.
2. **units**: ten conversions against their exact values.
3. **envcanada-alerts**: live, through ``ctx.fetch`` (``dd.weather.gc.ca`` only): the alerts in effect now, and the
   answer for one place that has an alert and one invented place (nothing: a refusal).
4. **gutenberg-plays**: the pack ``CUBBY_PLAYS`` names, ingested into the plugin's folder; the plugin's ten examples
   and ten not-lines asked of it: answered (with the facts) or refused.
The log keeps counts and the answers' shape; the plays' facts are in-fiction lines read from public-domain texts.

    CUBBY_PLAYS=<plays folder> python validation/plugins_checks.py
-> validation/logs/plugins_checks_<date>.json
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "standin")]


def startup() -> dict:
    from plugins_index import discover
    root = Path(tempfile.mkdtemp())
    for i in range(50):
        d = root / f"dummy-{i:02d}"
        d.mkdir()
        (d / "PLUGIN.md").write_text(f"---\nname: dummy-{i:02d}\ndescription: dummy plugin {i} for the startup check\n"
                                     f"examples: [a question {i}]\nkinds: [tool]\nentry: {{tool: \"tool.py:T\"}}\n---\n\n"
                                     + "documentation\n" * 200, encoding="utf-8")
        (d / "tool.py").write_text("raise SystemExit('imported at startup')\n", encoding="utf-8")
        (d / "payload.bin").write_bytes(os.urandom(200_000))
    took = []
    for _ in range(5):
        t0 = time.perf_counter()
        idx = discover([root])
        took.append(1000 * (time.perf_counter() - t0))
    shutil.rmtree(root, ignore_errors=True)
    imported = [m for m in sys.modules if m.startswith("cubby_plugin_dummy")]
    return {"plugins": len(idx), "errors": len(idx.errors), "median_ms": round(statistics.median(took), 1),
            "max_ms": round(max(took), 1), "imported": imported, "line_ms": 200}


def loader_for(names, data):
    import plugins_guard as guard
    from plugins_index import discover
    from plugins_load import Loader
    idx = discover([HERE.parent / "plugins"])
    for n in names:
        guard.approve(idx[n])
    return Loader(idx, data_root=data, log=lambda *_: None)


def units(loader) -> dict:
    cases = [(5, "mi", "km", 8.04672), (100, "fahrenheit", "celsius", 37.777778), (3, "ft", "m", 0.9144),
             (2, "kg", "lb", 4.409245), (60, "mph", "km/h", 96.56064), (1, "gal", "l", 3.785412),
             (20, "celsius", "fahrenheit", 68.0), (3, "hours", "seconds", 10800.0), (500, "g", "oz", 17.636981),
             (30, "cm", "in", 11.811024)]
    assert loader.load("units")
    ok = [abs(loader.call("units", {"value": v, "from": a, "to": b})["value"] - want) < 1e-5 for v, a, b, want in cases]
    return {"cases": len(cases), "exact": sum(ok)}


def alerts(loader) -> dict:
    assert loader.load("envcanada-alerts"), loader.refused.get("envcanada-alerts")
    src = loader.sources.get("envcanada-alerts") or loader.loaded["envcanada-alerts"]["kinds"]["source"]
    t0 = time.time()
    src.refresh(now=time.time() + 10_000)
    took = round(time.time() - t0, 1)
    out = {"in_effect": len(src.alerts), "refresh_s": took, "headlines": sorted({a["headline"] for a in src.alerts})[:8]}
    if src.alerts:
        place = src.alerts[0]["area"]
        out["asked_held_place"] = {"place": place, "facts": src.answer(f"is there a weather warning in {place}?")}
    out["asked_invented_place"] = src.answer("is there a weather warning in Zzyzxville?")
    return out


def plays(loader) -> dict:
    if not os.environ.get("CUBBY_PLAYS"):
        return {"skipped": "set CUBBY_PLAYS"}
    t0 = time.time()
    assert loader.load("gutenberg-plays"), loader.refused.get("gutenberg-plays")
    p = loader.index["gutenberg-plays"]
    world = loader.loaded["gutenberg-plays"]["kinds"]["world"]
    rows = []
    for kind, lines in (("example", p.examples), ("not", p.not_)):
        for q in lines:
            got = loader.ask(q)
            rows.append({"kind": kind, "question": q, "world": got["world"], "facts": got["facts"]})
    return {"ingest_s": round(time.time() - t0, 1), "works": len(world.fw.titles),
            "facts": world.fw.db.execute("SELECT COUNT(*) FROM fiction_facts").fetchone()[0],
            "examples_answered": sum(1 for r in rows if r["kind"] == "example" and r["facts"]),
            "examples_routed": sum(1 for r in rows if r["kind"] == "example" and r["world"]),
            "not_answered": sum(1 for r in rows if r["kind"] == "not" and r["facts"]), "rows": rows}


def main() -> None:
    os.environ.setdefault("CUBBY_PLUGINS_LOCK", str(Path(tempfile.mkdtemp()) / "lock.json"))   # never the real lock
    data = Path(tempfile.mkdtemp())
    rep = {"startup": startup()}
    loader = loader_for(["units", "envcanada-alerts", "gutenberg-plays"], data)
    rep["units"] = units(loader)
    try:
        rep["envcanada_alerts"] = alerts(loader)
    except Exception as e:                                   # noqa: BLE001 -- offline: said, not hidden
        rep["envcanada_alerts"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    rep["gutenberg_plays"] = plays(loader)
    out = HERE / "logs" / f"plugins_checks_{time.strftime('%Y-%m-%d')}.json"
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v) for k, v in rep.items()},
                     indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
