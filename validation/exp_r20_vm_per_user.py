"""One VM per user: what does isolation cost? (Nick, 2026-09-14: "we should also check resource usage
if one vm per user runs (separation of users requests)".)

Wired: STANDALONE (validation script; never imported by cubbyllm/).

The serving stack keeps ONE resident `cubelang run-proto` process (`CubelangSession`) and every question
goes through it. Per-user isolation means one such process per session: no user's frame, knowledge path
or program can reach another's, and a crash takes one user down, not the box. The question is what that
costs. Measured here, on this machine, with nothing modelled:

  * RSS per session, measured as the machine's used memory before and after N spawns (and per-process
    RSS where psutil can read it), so the answer is in users-per-GB;
  * the spawn time of the Nth session -- the cold cost a new user pays;
  * throughput: the same verify program run round-robin over N sessions from T threads, against the
    single-session baseline, so the answer is in questions-per-second-per-core and whether N sessions
    contend;
  * a leak check: RSS after the sessions have served `--per-session` programs each.

It does NOT measure the emitter: one GGUF in VRAM is shared by every user and is a separate budget
(the GPU lock in standin/emitter.py serialises it). This is the VM's half.

  python validation/exp_r20_vm_per_user.py --sessions 32 --threads 8 --per-session 40
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "standin", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROGRAM = """use vsa;

program CotChain implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, H1_CAPITAL, "paris";
        bind frame, H2_COUNTRY, "france";
        return recover(frame, H1_CAPITAL);
    }
}
"""


def mem_mb() -> tuple[float, float]:
    """(used MB, available MB) for the machine -- psutil when present, else the OS."""
    try:
        import psutil                                        # noqa: PLC0415
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1e6, vm.available / 1e6
    except Exception:                                        # noqa: BLE001
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return (m.ullTotalPhys - m.ullAvailPhys) / 1e6, m.ullAvailPhys / 1e6
        return 0.0, 0.0


def rss_of(pids: list[int]) -> float | None:
    try:
        import psutil                                        # noqa: PLC0415
        tot = 0.0
        for pid in pids:
            try:
                tot += psutil.Process(pid).memory_info().rss / 1e6
            except Exception:                                # noqa: BLE001
                pass
        return tot
    except Exception:                                        # noqa: BLE001
        return None


def main() -> None:
    from cubbyllm.bridges import cubelang_client as cc
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=32, help="concurrent users, one resident VM each")
    ap.add_argument("--threads", type=int, default=8, help="callers hitting them at once")
    ap.add_argument("--per-session", type=int, default=40, help="programs each session runs in the load phase")
    ap.add_argument("--baseline", type=int, default=200, help="programs the single-session baseline runs")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True); lines.append(s)

    used0, avail0 = mem_mb()
    log(f"machine: {used0:,.0f} MB used, {avail0:,.0f} MB available at start | sessions {a.sessions}, "
        f"threads {a.threads}, {a.per_session} programs each")

    # -- the baseline: one session, serially -------------------------------------------------
    s0 = cc.CubelangSession(exe=a.exe)
    t0 = time.perf_counter()
    for _ in range(a.baseline):
        s0.run(PROGRAM, fn="solve", args=["x"])
    base_s = time.perf_counter() - t0
    base_rate = a.baseline / base_s
    log(f"\nbaseline (1 session, serial): {a.baseline} programs in {base_s:.2f}s = {base_rate:,.0f}/s "
        f"({base_s / a.baseline * 1000:.2f} ms each)")
    s0.close()

    # -- N sessions: the spawn cost --------------------------------------------------------
    sessions, spawn_ms = [], []
    used_before, _ = mem_mb()
    for i in range(a.sessions):
        t = time.perf_counter()
        s = cc.CubelangSession(exe=a.exe)
        s.run(PROGRAM, fn="solve", args=["x"])               # first request: the process is warm after this
        spawn_ms.append((time.perf_counter() - t) * 1000)
        sessions.append(s)
    used_after, avail_after = mem_mb()
    pids = [getattr(getattr(s, "_proc", None), "pid", None) for s in sessions]
    pids = [p for p in pids if p]
    rss = rss_of(pids)
    grew = used_after - used_before
    # two different numbers, both true: summed RSS counts the executable's pages once PER PROCESS, and the
    # OS maps them once. Capacity is set by the MARGINAL cost -- what the machine actually gave up for N
    # more users -- so that is what users-per-GB is computed from; the RSS figure is reported beside it.
    per_mb = grew / a.sessions
    per_rss = (rss / len(pids)) if rss else None
    log(f"\n{a.sessions} resident sessions: spawn+first-call median {statistics.median(spawn_ms):.0f} ms "
        f"(min {min(spawn_ms):.0f}, max {max(spawn_ms):.0f})")
    log(f"  machine used {used_before:,.0f} -> {used_after:,.0f} MB (+{grew:,.0f} MB) | "
        + (f"summed process RSS {rss:,.0f} MB" if rss else "per-process RSS unavailable (no psutil)"))
    log(f"  ≈ {per_mb:,.2f} MB per user MARGINAL"
        + (f" (summed RSS says {per_rss:,.1f} MB each -- the executable's pages counted once per process, mapped once by the OS)" if per_rss else "")
        + f"  ->  {1000 / per_mb:,.0f} users per GB, {avail_after / per_mb:,.0f} more on this machine's free memory")

    # -- throughput under concurrency -------------------------------------------------------
    lock = threading.Lock(); nxt = [0]; lat: list[float] = []
    total = a.sessions * a.per_session

    def worker() -> None:
        while True:
            with lock:
                i = nxt[0]
                if i >= total:
                    return
                nxt[0] += 1
            s = sessions[i % len(sessions)]
            t = time.perf_counter()
            s.run(PROGRAM, fn="solve", args=["x"])           # one session is one user: never shared mid-call
            with lock:
                lat.append((time.perf_counter() - t) * 1000)

    # a session is single-threaded (one stdin/stdout pipe): give each thread its own slice of sessions
    def worker_sliced(k: int) -> None:
        mine = sessions[k::a.threads]
        for j in range(a.per_session * len(mine)):
            s = mine[j % len(mine)]
            t = time.perf_counter()
            s.run(PROGRAM, fn="solve", args=["x"])
            with lock:
                lat.append((time.perf_counter() - t) * 1000)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker_sliced, args=(k,)) for k in range(a.threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    load_s = time.perf_counter() - t0
    n = len(lat)
    lat.sort()
    rate = n / load_s
    log(f"\nload: {n:,} programs over {a.sessions} sessions from {a.threads} threads in {load_s:.2f}s = "
        f"{rate:,.0f}/s ({rate / max(1, a.threads):,.0f}/s per thread)")
    log(f"  latency ms: p50 {lat[n // 2]:.2f} · p90 {lat[int(n * 0.9)]:.2f} · p99 {lat[int(n * 0.99)]:.2f} · max {lat[-1]:.2f}")
    log(f"  vs the serial baseline: {rate / base_rate:.2f}x throughput on {a.threads} threads")

    used_end, avail_end = mem_mb()
    rss_end = rss_of(pids)
    log(f"\nafter the load: machine used {used_end:,.0f} MB"
        + (f" | summed RSS {rss_end:,.0f} MB ({(rss_end - rss) / max(1, len(pids)):+.2f} MB per session since warm)" if rss and rss_end else ""))
    for s in sessions:
        s.close()
    time.sleep(1.0)
    used_closed, _ = mem_mb()
    log(f"after close: machine used {used_closed:,.0f} MB (released {used_end - used_closed:,.0f} MB)")

    out = {"sessions": a.sessions, "threads": a.threads, "per_session": a.per_session,
           "baseline_programs": a.baseline, "baseline_s": round(base_s, 3), "baseline_rate_per_s": round(base_rate, 1),
           "spawn_ms": {"median": round(statistics.median(spawn_ms), 1), "min": round(min(spawn_ms), 1), "max": round(max(spawn_ms), 1)},
           "mem_mb": {"used_before": round(used_before, 1), "used_after": round(used_after, 1), "grew": round(grew, 1),
                      "summed_rss": round(rss, 1) if rss else None, "per_user_marginal": round(per_mb, 3),
                      "per_user_rss": round(per_rss, 2) if per_rss else None,
                      "users_per_gb": round(1000 / per_mb, 1), "used_after_load": round(used_end, 1),
                      "released_on_close": round(used_end - used_closed, 1)},
           "load": {"programs": n, "wall_s": round(load_s, 3), "rate_per_s": round(rate, 1),
                    "p50_ms": round(lat[n // 2], 3), "p90_ms": round(lat[int(n * 0.9)], 3),
                    "p99_ms": round(lat[int(n * 0.99)], 3), "max_ms": round(lat[-1], 3),
                    "speedup_vs_serial": round(rate / base_rate, 2)},
           "note": "one resident cubelang run-proto process per user. The emitter (one GGUF, GPU-locked) is shared "
                   "and is a separate budget; this is the VM's half of per-user isolation."}
    (LOGS / f"exp_r20_vm_per_user{a.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (LOGS / f"exp_r20_vm_per_user{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwrote exp_r20_vm_per_user{a.tag}.{{json,log}}")


if __name__ == "__main__":
    main()
