"""board_timing -- order 0027's latency line: every /board/* endpoint under 200 ms at p95 on the local stack.

The server's own handler (``serve_api.make_handler``) with ``--board`` on, a stand-in brain (no model: the board reads
what the server holds, not what the model does) carrying ``--records`` kept ask records with thought graphs, and this
machine's real data: the order queue, the news archive (``CUBBY_NEWS_ARCHIVE``) and the run logs (``CUBBY_RUNS``).
Each endpoint is called ``--n`` times per audience; the first call (cold caches) is reported apart.

    python validation/board_timing.py [--n 50] [--records 200] [--serve 8790]
-> validation/logs/board_timing_<date>.json  (``--serve`` instead keeps the server up for a look at the page)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "standin")]

PATHS = ["/board/home", "/board/system", "/board/journey/r{last}", "/board/watch", "/board/worlds", "/board/forecast",
         "/board/mood", "/board/agents", "/board", "/board/static/app.js", "/board/assets/chat.webp"]


def stand_in(records: int):
    graph = {"nodes": [{"id": i, "kind": k} for i, k in enumerate(["question", "plan", "fetch", "gate", "hop", "fact", "vm", "answer"], 1)],
             "edges": []}
    hist = [{"id": f"r{i}", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": "Where was Ada Lovelace born?",
             "answer": "London", "verified": i % 7 != 0, "reply": "London." if i % 7 else None, "wall_s": 0.1, "graph": graph}
            for i in range(records)]
    loop = SimpleNamespace(history=hist, _forms=None, calls={"asked": records}, world=SimpleNamespace(), source=None, news=None)
    loop.ask = lambda text, **_kw: dict(hist[-1], question=text)            # --serve: the page's chat shows a kept record
    try:
        from question_forms import Forms
        loop._forms = Forms()
    except Exception:                                                     # noqa: BLE001 -- spans without roles
        pass
    return SimpleNamespace(board=True, ask_loop=loop, worlds={}, cortices=set(), emitter=SimpleNamespace(name="stand-in"),
                           chat=SimpleNamespace(chem=SimpleNamespace(to_dict=lambda: {})), events=[], harness=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--records", type=int, default=200)
    ap.add_argument("--serve", type=int, default=0)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--plugins", default=None, help="plugin folders to mount on the stand-in (order 0028)")
    a = ap.parse_args()
    import serve_api
    brain = stand_in(a.records)
    if a.plugins:
        from plugins_serve import mount
        mount(brain, a.plugins)
    httpd = serve_api.ThreadingHTTPServer(("127.0.0.1", a.serve), serve_api.make_handler(brain))
    if a.serve:
        print(f"board at http://127.0.0.1:{a.serve}/board (audit: ?token= the CUBBY_AUDIT_TOKEN)", flush=True)
        httpd.serve_forever()
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base, tok = f"http://127.0.0.1:{httpd.server_address[1]}", os.environ.get("CUBBY_AUDIT_TOKEN")
    rep = {"n": a.n, "records": a.records, "archive": bool(os.environ.get("CUBBY_NEWS_ARCHIVE")),
           "runs": bool(os.environ.get("CUBBY_RUNS")), "endpoints": {}}
    for who in ("user", "audit") if tok else ("user",):
        for p in PATHS:
            path = p.format(last=a.records - 1)
            ms = []
            for _ in range(a.n + 1):
                t0 = time.perf_counter()
                req = urllib.request.Request(base + path, headers={"X-Audit-Token": tok} if who == "audit" else {})
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
                ms.append(1000 * (time.perf_counter() - t0))
            warm = sorted(ms[1:])
            rep["endpoints"][f"{who} {p}"] = {"cold_ms": round(ms[0], 1), "p50_ms": round(warm[len(warm) // 2], 1),
                                              "p95_ms": round(warm[int(0.95 * (len(warm) - 1))], 1), "max_ms": round(warm[-1], 1)}
    rep["worst_p95_ms"] = max(v["p95_ms"] for v in rep["endpoints"].values())
    rep["worst_cold_ms"] = max(v["cold_ms"] for v in rep["endpoints"].values())
    httpd.shutdown()
    out = os.path.join(HERE, "logs", f"board_timing_{a.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "endpoints"}), "\n" +
          "\n".join(f"  {k:32s} {v}" for k, v in rep["endpoints"].items()))


if __name__ == "__main__":
    main()
