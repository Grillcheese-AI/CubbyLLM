import cubbyllm.core.probing as mod
from cubbyllm.core.probing import WrongContextProbe
from cubbyllm.core.protocols import Wiring


def test_probe_protocol_present():
    assert hasattr(WrongContextProbe, "probe")


def test_wiring_declared():
    assert mod.__wiring__ in (Wiring.WIRED, Wiring.STANDALONE)
