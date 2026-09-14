# One VM per user: what isolation costs — 14 September

Nick: *"we should also check resource usage if one vm per user runs (separation of users requests)."*
Measured, not modelled: `validation/exp_r20_vm_per_user.py`, logs in `validation/logs/exp_r20_vm_per_user.*`.

The serving stack keeps one resident `cubelang run-proto` process (`CubelangSession`) and every question
goes through it. Per-user isolation means one such process per session: no user's frame, knowledge path
or program can reach another's, and a crash takes one user down rather than the box.

## The numbers (256 sessions, 8 threads, this machine)

| | |
|---|---|
| memory per user, **marginal** | **1.69 MB** — the machine gave up 432 MB for 256 sessions |
| memory per user, summed RSS | 6.8 MB — the executable's pages counted once per process; the OS maps them once. Capacity follows the marginal figure |
| **users per GB** | **592** (and 12,617 more fit in this machine's free memory) |
| cold cost of a new user | **19 ms** median spawn + first call (min 17, max 30) |
| throughput | **5,211 programs/s** across 256 sessions from 8 threads — 3.62× the serial baseline of 1,438/s |
| latency | p50 1.47 ms · p90 2.10 · p99 2.77 · max 3.73 |
| leak, after 7,680 programs | +0.25 MB per session; **506 MB released on close** |

At 128 sessions the same shape: 1.9 MB marginal each, 19 ms spawn, 5,176/s, p99 2.83 ms. Nothing degrades
between 16 and 256 — spawn time is flat, per-session memory is flat, throughput is bounded by the thread
count and not by the number of sessions.

## What it means

Isolation is not the expensive part of serving. A VM per user costs under 2 MB and 19 ms, so a 16 GB box
holds on the order of **8,000 idle user sessions** and serves thousands of verifications a second from a
handful of threads. The budget is dominated by the *emitter*: one GGUF in VRAM, shared by every user and
serialised by the GPU lock in `standin/emitter.py` — that is the queue that decides concurrency, and it is
a separate measurement. The VM's half of per-user isolation is, in practice, free.

Two consequences worth holding on to. A session can be spawned **per request** rather than per user for
19 ms if statelessness is ever worth more than warmth. And the honest reading of "users per GB" is the
marginal number: summed RSS over-counts shared pages by 4× here, which is the difference between 147 and
592 users per GB — the kind of figure that quietly becomes a wrong capacity plan.

Not measured here: the emitter's VRAM and its lock under concurrency; the fact store (one wiki world of
552k facts is shared read-only across users today, and a per-user store would be a different budget);
and the process count against the OS's own limits, which nothing here approached.
