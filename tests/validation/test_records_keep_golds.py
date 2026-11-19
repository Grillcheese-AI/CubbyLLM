"""Order 0075 (Nick, 2026-11-19): a record of arm C or M keeps every gold of its question. Records before kept the first
8, while the run graded against all of them (0075 found it); existing records are not edited."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from system_eval.gpu import m_record  # noqa: E402
from system_eval.run_c import one  # noqa: E402

GOLDS = [f"g{i}" for i in range(12)]
ROW = {"id": "x:0", "bench": "webq", "question": "q", "golds": GOLDS, "seen": False}


def test_an_m_record_keeps_every_gold():
    r = m_record(ROW, "webq", "g11")
    assert r["golds"] == GOLDS and r["verdict"] == "right"


class _Loop:
    def ask(self, q):
        return {"answer": None, "verified": False, "reason": "no_plan"}


class _Seat:
    calls, last = 0, None


def test_a_c_record_keeps_every_gold():
    r = one(_Loop(), _Seat(), {"vm_calls": 0, "vm_s": 0.0}, ROW, set())
    assert r["golds"] == GOLDS and r["verdict"] == "refused"
