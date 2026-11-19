"""cubby_serve/records/envelope.py -- a CheezML record kept private and still checkable (order 0064's addendum of
2026-11-10, sections 2 and 3; order 0076, step 3).

**Sign, then encrypt.** The plain record is canonicalised (RDFC-1.0 over its graph, `pyld`, the CheezML context
served from the pinned copy only: no network), and its canonical N-Quads hashed. The host signs, with Ed25519 (JWS),
the public envelope: the record's ``urn:uuid``, its type, the recorded time, the key ids, the commitments and that
digest. The body is encrypted (JWE, AES-256-GCM) in two parts, each under its readers' key:
- **conversation:** the question, the answer or refusal and the turn (the tenant's members);
- **provenance:** everything else: the facts read, their sources and revisions, the program, the run, the checks
  (the audit view only).
**Commitments:** the answer value and each fact read get an HMAC-SHA-256 under a random 32-byte salt of the record's
own; the salts travel inside the provenance part, so an owner can reveal one value and its salt and anyone can check
it against the envelope. No plain hash of a value is written (a short answer could be guessed from its hash).

**Keys** are handed in (``Keys``): the signing key, and one data key per part, each with its id. They come from the
OS key store or a key service, made and kept by Nick; never from a file, a log, an argument or a commit. Nothing here
reads, writes, prints or makes a key: the tests make theirs in memory.

Wiring: STANDALONE (validation/verify_cheez.py and the tests; nothing on the serve path yet).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path

__wiring__ = "STANDALONE"

CONTEXT_URL = "https://w3id.org/cheezml/context.jsonld"
CONVERSATION = {"cheez:Question", "cubby:Answer", "cubby:Refusal", "cubby:Turn"}
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Keys:
    """The keys one record is sealed with (`joserfc.jwk` keys) and their ids."""
    sign: object
    sign_id: str
    conversation: object
    conversation_id: str
    provenance: object
    provenance_id: str


def _context() -> dict:
    return json.loads((ROOT / "vendor" / "cheezml" / "src" / "context.jsonld").read_text(encoding="utf-8"))


def _loader(url, options=None):
    if url != CONTEXT_URL:
        raise ValueError(f"no document is fetched: {url}")
    return {"contextUrl": None, "documentUrl": url, "document": _context()}


def canonical(record: dict) -> str:
    """The record's RDFC-1.0 canonical N-Quads (a JSON-LD key order or spacing changes nothing)."""
    from pyld import jsonld
    doc = dict(record, **{"@context": CONTEXT_URL})
    return jsonld.normalize(doc, {"algorithm": "URDNA2015", "format": "application/n-quads", "documentLoader": _loader})


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def commit(value: str, salt: bytes) -> str:
    return hmac.new(salt, str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def _types(n: dict) -> set:
    t = n.get("@type", [])
    return set(t if isinstance(t, list) else [t])


def _committed(graph: list) -> dict:
    """{"answer": its value | None, "facts": [each fact version's statement as one string]}."""
    ans = next((n.get("prov:value") for n in graph if "cubby:Answer" in _types(n)), None)
    facts = [json.dumps([n.get("rdf:subject"), n.get("rdf:predicate"), n.get("rdf:object"), n.get("schema:identifier")],
                        sort_keys=True) for n in graph if "cheez:FactVersion" in _types(n)]
    return {"answer": ans, "facts": facts}


def seal(record: dict, keys: Keys, rng=os.urandom) -> dict:
    """{"envelope": the signed public part (a JWS), "conversation": JWE, "provenance": JWE}."""
    from joserfc import jwe, jws
    graph = record["@graph"]
    head = next(n for n in graph if "cheez:Record" in _types(n))
    values = _committed(graph)
    salts = {"answer": rng(32), "facts": [rng(32) for _ in values["facts"]]}
    public = {"id": head["@id"], "type": sorted(_types(head)), "recorded": head.get("prov:generatedAtTime"),
              "kid": {"sign": keys.sign_id, "conversation": keys.conversation_id, "provenance": keys.provenance_id},
              "record_sha256": hashlib.sha256(canonical(record).encode("utf-8")).hexdigest(),
              "commitments": {"answer": commit(values["answer"], salts["answer"]) if values["answer"] is not None else None,
                              "facts": [commit(v, s) for v, s in zip(values["facts"], salts["facts"])]}}
    env = jws.serialize_compact({"alg": "Ed25519", "kid": keys.sign_id}, json.dumps(public, sort_keys=True), keys.sign,
                                algorithms=["Ed25519"])
    conv = [n for n in graph if _types(n) & CONVERSATION]
    prov = [n for n in graph if not _types(n) & CONVERSATION]
    secrets = {"salts": {"answer": _b64(salts["answer"]), "facts": [_b64(s) for s in salts["facts"]]}, "values": values}
    seal_part = lambda part, key, kid: jwe.encrypt_compact({"alg": "dir", "enc": "A256GCM", "kid": kid},  # noqa: E731
                                                           json.dumps(part, sort_keys=True).encode("utf-8"), key)
    return {"envelope": env,
            "conversation": seal_part({"@graph": conv}, keys.conversation, keys.conversation_id),
            "provenance": seal_part({"@graph": prov, "secrets": secrets}, keys.provenance, keys.provenance_id)}


__all__ = ["CONTEXT_URL", "Keys", "canonical", "commit", "seal"]
