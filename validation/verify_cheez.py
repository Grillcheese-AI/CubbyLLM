"""Order 0076, step 4: the verifier for a sealed CheezML record (`cubby_serve/records/envelope.py`). Offline: it opens
no connection, and reads only the home store, through its pipe (`wdstore serve`, never a socket).

- **anyone** (the public key): ``signed`` -> "signed" when the envelope's signature holds, else "changed";
- **a reader** (the public key and both parts' keys): ``read`` decrypts the parts and says
  - "changed": a part does not open, or the record it holds is not the one the envelope signed (its RDFC-1.0 digest);
  - "other snapshot": the record names a snapshot the store is not;
  - "missing": a fact version is not in the store at its revision (each one named);
  - "same": the signed record recomputes, and every fact version is in the store at its revision;
  a member's key opens the conversation part only: ``read_conversation`` gives it, and nothing about provenance;
- **an owner's reveal** (``revealed``): one value and its salt checked against the envelope's commitment.

A library for now (`tests/records/test_envelope.py`, step 7's timing): its command line comes with Nick's keys, which
are read from the key store, never from a file or an argument.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from cubby_serve.records.envelope import canonical, commit  # noqa: E402

__wiring__ = "STANDALONE"


def _public(sealed: dict, public_key) -> dict | None:
    from joserfc import jws
    try:
        return json.loads(jws.deserialize_compact(sealed["envelope"], public_key, algorithms=["Ed25519"]).payload)
    except Exception:                                    # noqa: BLE001 -- any failure to verify is "changed"
        return None


def signed(sealed: dict, public_key) -> str:
    return "signed" if _public(sealed, public_key) is not None else "changed"


def _open(token: str, key) -> dict | None:
    from joserfc import jwe
    try:
        return json.loads(jwe.decrypt_compact(token, key).plaintext)
    except Exception:                                    # noqa: BLE001
        return None


def read_conversation(sealed: dict, member_key) -> dict | None:
    return _open(sealed["conversation"], member_key)


def _types(n: dict) -> set:
    t = n.get("@type", [])
    return set(t if isinstance(t, list) else [t])


def in_store(fv: dict, source: dict, ask) -> bool:
    """The statement in the store at the record's revision: the item's revision, and a statement with that id,
    property, rank and value."""
    from cubbyllm.records.cheez_facts import RANK, value
    item = source["schema:identifier"]
    e = ask({"op": "get", "id": item, "nodes": True}) or {}
    if not e.get("found") or str(e.get("revision")) != source["schema:version"]:
        return False
    pid = fv["rdf:predicate"].split(":")[-1]
    return any(s.get("guid") == fv.get("schema:identifier") and f"P{s.get('pid')}" == pid and RANK.get(s.get("rank"))
               == fv["wikibase:rank"] and value(s.get("value")) == fv["rdf:object"] for s in e.get("statements", []))


def read(sealed: dict, public_key, conversation_key, provenance_key, ask=None, snapshot: str | None = None) -> dict:
    pub = _public(sealed, public_key)
    conv, prov = _open(sealed["conversation"], conversation_key), _open(sealed["provenance"], provenance_key)
    if pub is None or conv is None or prov is None:
        return {"says": "changed", "why": "the signature, or a part that does not open"}
    graph = conv["@graph"] + prov["@graph"]
    if hashlib.sha256(canonical({"@graph": graph}).encode("utf-8")).hexdigest() != pub["record_sha256"]:
        return {"says": "changed", "why": "the record is not the one the envelope signed"}
    snaps = [n["cheez:hash"] for n in graph if n.get("dcterms:type") == "cheez:StoreSnapshot"]
    if snapshot is not None and any(s != snapshot for s in snaps):
        return {"says": "other snapshot", "record": snaps, "store": snapshot}
    if ask is not None:
        by_id = {n["@id"]: n for n in graph}
        missing = [n["@id"] for n in graph if "cheez:FactVersion" in _types(n)
                   and not in_store(n, by_id[n["prov:hadPrimarySource"]], ask)]
        if missing:
            return {"says": "missing", "fact versions": missing}
    return {"says": "same"}


def revealed(sealed: dict, public_key, which: str, value: str, salt_b64: str, index: int = 0) -> bool:
    """An owner's reveal: ``which`` "answer" or "facts" (with its index), the value and its salt."""
    pub = _public(sealed, public_key)
    if pub is None:
        return False
    salt = base64.urlsafe_b64decode(salt_b64 + "=" * (-len(salt_b64) % 4))
    c = pub["commitments"][which]
    want = c if which == "answer" else (c[index] if index < len(c) else None)
    return want is not None and commit(value, salt) == want


__all__ = ["in_store", "read", "read_conversation", "revealed", "signed"]
