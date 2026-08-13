"""End-to-end smoke through the real cubelang VM. Skipped when the binary
is absent (CI without the Rust toolchain)."""
import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.reasoning import answer

try:
    cc.find_cubelang_exe()
    HAVE_VM = True
except cc.CubelangNotFound:
    HAVE_VM = False

pytestmark = pytest.mark.skipif(not HAVE_VM, reason="cubelang exe not found")

Q2 = "What is the continent of the country of citizenship of ada lovelace?"
F1 = "england is the country of citizenship of ada lovelace"
F2 = "europe is the continent of england"


def retriever(query, k):
    return [(0.9, F1)] if "ada" in query.lower() else [(0.9, F2)]


def run_fn(source, fn):
    return cc.run_program_proto(source, fn=fn)


def test_two_hop_chain_through_real_vm():
    r = answer(Q2, retriever, run_fn, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is True
    assert r.answer == "europe"
    assert all(h.similarity is not None and h.similarity >= 0.5 for h in r.trace)
