"""Reader for the Teams JSON export - the structured one, not the copy-paste.

This shape is everything the text export was not: `author`, an ISO `timestamp` with
real precision, plain `text` beside `contentHtml`, plus `reactions`, `attachments`,
`replyTo`, `edited`, `deleted`, `system` and `isOwn`. No chrome to strip, no preview
lines duplicating every message, no two-shapes-in-one-file.

What it adds beyond the text export, and why each one matters for affect:

  GAP        seconds since the previous message. Six messages in ninety seconds is a
             different state from one reply five hours later, and nobody has to label
             it - the clock already did. This is the cheapest affect signal in the
             corpus and it is free.
  BURST      index within a run of messages from the same author with small gaps. A
             thought fired off in four fragments is one utterance under pressure; the
             fragmentation IS the signal, so the pieces are kept AND grouped.
  REACTIONS  a thumbs-up or a laugh from the other person is a response to affect,
             recorded by someone who was there.
  REPLY_TO   what a message is answering, which is the appraisal context that the
             monoamines do not carry (grok's point about disappointment being a
             trajectory, not a coordinate).

System messages, deleted messages, and empty text are dropped - a join notice is not
register. Everything else goes through the same scrub as the text path.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime

BURST_GAP = 120          # seconds; above this, a new burst starts


def _when(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_json(path: pathlib.Path) -> tuple[list[dict], dict]:
    """Returns (messages, meta). Messages carry gap/burst/reaction fields the text
    export could not provide."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    meta = blob.get("meta") or {}
    raw = blob.get("messages") or []

    out, prev_t, prev_author, burst = [], None, None, 0
    for m in raw:
        if m.get("system") or m.get("deleted"):
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        t = _when(m.get("timestamp") or "")
        gap = None
        if t and prev_t:
            gap = max(0.0, (t - prev_t).total_seconds())
        # A burst is same author, small gap. Resets on either change.
        if prev_author == m.get("author") and gap is not None and gap <= BURST_GAP:
            burst += 1
        else:
            burst = 0
        reactions = [r.get("type") or r.get("emoji") or r.get("name") or "?"
                     for r in (m.get("reactions") or []) if isinstance(r, dict)]
        out.append({
            "speaker": m.get("author") or "?",
            "when": m.get("timestamp"),
            "text": text,
            "gap_s": None if gap is None else round(gap, 1),
            "burst": burst,
            "reactions": reactions,
            "reply_to": (m.get("replyTo") or None),
            "edited": bool(m.get("edited")),
            "attachments": len(m.get("attachments") or []),
        })
        prev_t, prev_author = t, m.get("author")
    return out, meta
