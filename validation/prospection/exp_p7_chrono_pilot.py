"""H-P7 — chrono init on the recurrent gates: does a long-time-constant
substrate SURVIVE training, and does it buy anything beyond the window?

WHY THIS IS THE LAST OPEN ROUTE. H-P3 measured the trained hybrid's recurrence
at tau ~ 0.5–2 steps per unit, shortening monotonically from init (2.8) through
step 500 (2.0), step 2000 (1.2–2.1) and ~1B tokens (0.5–1.8); the gate itself
caps tau at ~1000. H-P6 then closed the "make it with an objective" route: an
explicit 500-token prediction loss moved no recurrent time constant at all.
What is left is setting the spectrum at INIT — Tallec & Ollivier's chrono init,
tau log-uniform over decades — and asking whether gradient descent keeps it
when attention is available to do the long-range work instead.

Two matched arms (this wrapper runs exp_p6_multihorizon_pilot.py, the Group P
pilot-arm runner, with CB_MH_W=0 and CB_CHRONO=1; the baseline arm is the same
runner with CB_CHRONO=0):

    baseline   default init (proj_d bias 1.0 -> every unit tau ~ 2.8)
    chrono     tau log-uniform in [CB_CHRONO_TMIN, CB_CHRONO_TMAX] (1..500)

WHAT DECIDES IT, in order:
  1. held-out CE at matched tokens — chrono must be free (kill: worse beyond
     ~0.01 nats at 65M tokens).
  2. SURVIVAL: the gate spectrum at init vs after training, per layer. The
     init spectrum is set by construction (~22% of units with tau >= 100); the
     question is what fraction is still there at the end. Kill: the trained
     spectrum collapses onto the baseline's (p50 1–2, no unit >= 8) — training
     erodes chrono the way it eroded the default init, and the route is closed.
  3. BEYOND-WINDOW MEMORY: the impulse response — one substituted token's
     relative perturbation of the recurrent state, read at lags past the
     attention window (>512). Inside the window attention carries it; beyond,
     only the recurrence can. Baseline should be ~0 there; a surviving chrono
     spectrum shows up as a non-zero tail. This is the mechanism-level number.
  4. CAPABILITY: exp_needle_recall.py on both checkpoints at 256/512/1024/4096,
     read the way H-D4 reads it — TRAINED vs its own shuffled control. Chrono
     only matters if recall past the window rises above the floor.

If survival holds but recall does not move, the record says: the recurrence CAN
hold a long time constant but has nothing useful to put in it at this scale —
which is still a design fact, and points at the episodic store as the
beyond-window mechanism, as H-P3 concluded.

Env (on top of the runner's): CB_CHRONO (default 1 here) CB_CHRONO_TMIN
CB_CHRONO_TMAX CB_CHRONO_WSCALE (1.0 = bias-only, the paper's form) — and
everything exp_p6 / train_colab accept (CB_D CB_L CB_B CB_S CB_STEPS CB_LR
CB_WARMUP CB_CKPT CB_MH_OUT CB_JSONL_DIR CB_CACHE_MIRROR ...).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CB_CHRONO", "1")
os.environ.setdefault("CB_MH_W", "0")
os.environ.setdefault("CB_MH_TAG", "chrono" if os.environ["CB_CHRONO"] == "1" else "baseline")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp_p6_multihorizon_pilot as runner  # noqa: E402  (reads the env at import)

if __name__ == "__main__":
    runner.main()
