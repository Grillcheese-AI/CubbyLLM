"""Order 0076, steps 3 and 4 (cubby_serve/records/envelope.py, validation/verify_cheez.py), as 0064's addendum asks.
Keys are made in memory for each test and never written. Pinned:
- the same graph in another key order gives the same signature input;
- an untouched record: "signed", and "same" against the store at its revision;
- every spoiled copy is caught (a fact's value changed, a fact dropped, a fact added, the snapshot changed, the
  answer changed, one byte of the envelope, one byte of a part): never "signed" and "same" together;
- a body for tenant A does not open with tenant B's key; a member's key does not open the provenance part;
- a revealed value with its salt matches its commitment, another value does not;
- planted secret strings are not in the envelope; the verifier opens no connection; no key in a tracked file."""
from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("joserfc")
pytest.importorskip("pyld")
from joserfc.jwk import OctKey, OKPKey  # noqa: E402

from cubby_serve.records import envelope as E  # noqa: E402
from cubbyllm.records import cheez_writer as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
import verify_cheez as V  # noqa: E402
from test_cheez_writer import REC, RUN, ST  # noqa: E402

SECRET = "Plantedsecretvalue-7f3a"
PEM = __import__("re").compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----\s*\n[A-Za-z0-9+/=]{40,}")
JWK_D = __import__("re").compile(r'"d"\s*:\s*"[A-Za-z0-9_-]{40,}"')


def _keys():
    return E.Keys(OKPKey.generate_key("Ed25519"), "sign-1", OctKey.generate_key(256), "conv-a", OctKey.generate_key(256), "prov-a")


def _pub(k):
    return OKPKey.import_key(k.sign.as_dict(private=False))


def _store(st=ST):
    entity = {"found": True, "revision": st["revision"], "statements": [{"guid": st["guid"], "pid": int(st["pid"][1:]),
                                                                          "rank": st["rank"], "value": st["value"]}]}
    return lambda req: entity if req.get("id") == st["item"] else {"found": False}


def _record(rec=REC):
    g, why = W.write(rec, RUN)
    assert why is None
    return g


def _read(sealed, k, ask=_store()):
    return V.read(sealed, _pub(k), k.conversation, k.provenance, ask=ask, snapshot=RUN["snapshot"])


def test_canonical_form_ignores_key_order():
    g = _record()
    shuffled = {"@graph": [dict(reversed(list(n.items()))) for n in reversed(g["@graph"])]}
    assert E.canonical(g) == E.canonical(shuffled)


def test_an_untouched_record_is_signed_and_the_same():
    k = _keys()
    sealed = E.seal(_record(), k)
    assert V.signed(sealed, _pub(k)) == "signed" and _read(sealed, k) == {"says": "same"}


def _respoil(sealed, k, spoil):
    """The parts decrypted, spoiled and encrypted again with the data keys; the envelope kept (no signing key)."""
    from joserfc import jwe
    conv, prov = (V._open(sealed[p], key) for p, key in (("conversation", k.conversation), ("provenance", k.provenance)))
    spoil(conv["@graph"], prov["@graph"])
    out = dict(sealed)
    for p, part, key, kid in (("conversation", conv, k.conversation, k.conversation_id), ("provenance", prov, k.provenance, k.provenance_id)):
        out[p] = jwe.encrypt_compact({"alg": "dir", "enc": "A256GCM", "kid": kid}, json.dumps(part).encode(), key)
    return out


def _fv(graph):
    return next(n for n in graph if "cheez:FactVersion" in V._types(n))


SPOILS = {
    "a fact's value changed": lambda c, p: _fv(p).update({"rdf:object": {"@id": "wd:Q1"}}),
    "a fact dropped": lambda c, p: p.remove(_fv(p)),
    "a fact added": lambda c, p: p.append(dict(_fv(p), **{"@id": "urn:uuid:00000000-0000-4000-8000-0000000000aa"})),
    "the snapshot changed": lambda c, p: next(n for n in p if n.get("dcterms:type") == "cheez:StoreSnapshot").update({"cheez:hash": "f" * 64}),
    "the answer changed": lambda c, p: next(n for n in c if "cubby:Answer" in V._types(n)).update({"prov:value": "Euro"}),
}


@pytest.mark.parametrize("name", sorted(SPOILS))
def test_a_spoiled_copy_is_never_signed_and_the_same(name):
    k = _keys()
    bad = _respoil(E.seal(_record(), k), k, SPOILS[name])
    assert _read(bad, k)["says"] == "changed"


def test_a_record_signed_over_a_wrong_fact_or_snapshot_is_caught_by_the_store():
    k = _keys()
    wrong = copy.deepcopy(REC)
    wrong["cert"]["facts"][0]["worlds"][0]["statements"] = [dict(ST, value={"entity": "Q1"})]
    assert _read(E.seal(_record(wrong), k), k)["says"] == "missing"
    other = dict(RUN, snapshot="e" * 64)
    g, _ = W.write(REC, other)
    assert _read(E.seal(g, k), k)["says"] == "other snapshot"


@pytest.mark.parametrize("part", ["envelope", "conversation", "provenance"])
def test_one_changed_byte_is_caught(part):
    k = _keys()
    sealed = E.seal(_record(), k)
    tok = sealed[part]
    i = len(tok) // 2
    bad = dict(sealed, **{part: tok[:i] + ("A" if tok[i] != "A" else "B") + tok[i + 1:]})
    assert V.signed(bad, _pub(k)) != "signed" or _read(bad, k)["says"] != "same"


def test_tenants_and_parts_stay_apart():
    a, b = _keys(), _keys()
    sealed = E.seal(_record(), a)
    assert V.read_conversation(sealed, b.conversation) is None            # tenant B's key
    assert V.read_conversation(sealed, a.conversation) is not None        # a member of A reads the conversation
    assert V._open(sealed["provenance"], a.conversation) is None          # and not the provenance


def test_a_revealed_value_matches_its_commitment_and_no_other():
    k = _keys()
    sealed = E.seal(_record(), k)
    salt = V._open(sealed["provenance"], k.provenance)["secrets"]["salts"]["answer"]
    assert V.revealed(sealed, _pub(k), "answer", "Egyptian pound", salt)
    assert not V.revealed(sealed, _pub(k), "answer", "Euro", salt)


def test_planted_secrets_are_not_in_the_envelope(monkeypatch):
    rec = copy.deepcopy(REC)
    rec["answer"] = SECRET
    k = _keys()
    sealed = E.seal(_record(rec), k)
    from joserfc import jws
    payload = jws.extract_compact(sealed["envelope"].encode()).payload.decode()
    assert SECRET not in sealed["envelope"] and SECRET not in payload
    assert SECRET not in sealed["conversation"] and SECRET not in sealed["provenance"]


def test_the_verifier_opens_no_connection(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("a connection was opened")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    k = _keys()
    sealed = E.seal(_record(), k)
    assert _read(sealed, k) == {"says": "same"} and V.signed(sealed, _pub(k)) == "signed"


def test_no_key_material_in_a_tracked_file():
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    bad = []
    for f in files:
        p = ROOT / f
        if p.suffix in (".png", ".jpg", ".pt", ".safetensors", ".ipynb", ".gz") or not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if PEM.search(text) or JWK_D.search(text):                         # a key's body, not a pattern for one
            bad.append(f)
    assert bad == []
