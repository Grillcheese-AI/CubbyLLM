"""cubby_serve/board/chat_store.py -- encrypted persistence for Control Board chat turns.

Every user chat persisted by the serve stack goes through `cubbyllm.speech.vault` (never plaintext on disk).
Each turn is appended as one authenticated ciphertext line (fresh random nonce + HMAC-SHA256 via `vault.encrypt_str`),
so restarts and browser reloads restore both the Control Board thread and `CubbyChat`'s recent memory.

`clear()` retires the active file beside itself with a timestamp (never permanently deletes).

Wired: WIRED (`cubby_serve.cli`, `cubby_serve.post`, `cubby_serve.board.api`).
"""
from __future__ import annotations

import json
import pathlib
import threading
import time

from cubby_serve.keys import use_host_key
from cubbyllm.speech import audience, vault

__wiring__ = "WIRED"


class BoardChatStore:
    """Vault-encrypted append-only store of Control Board chat turns (`{ts, q, rec}`)."""

    def __init__(self, path: str | pathlib.Path, max_keep: int = 500) -> None:
        self.path = pathlib.Path(path)
        self.max_keep = max(1, int(max_keep))
        self._lock = threading.Lock()
        self._turns: list[dict] = []
        use_host_key()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        loaded: list[dict] = []
        for raw in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(vault.decrypt_str(line))
                if isinstance(item, dict) and "q" in item and isinstance(item.get("rec"), dict):
                    loaded.append(item)
            except (ValueError, RuntimeError):
                continue                                 # skip corrupt/foreign lines without breaking boot
        self._turns = loaded[-self.max_keep:]

    def append(self, question: str, rec: dict) -> dict:
        """Encrypt and append one board turn; safe to call from threaded HTTP handlers."""
        entry = {"ts": rec.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "q": str(question), "rec": json.loads(json.dumps(rec, default=str))}
        blob = vault.encrypt_str(json.dumps(entry, ensure_ascii=False))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(blob + "\n")
            self._turns.append(entry)
            if len(self._turns) > self.max_keep:
                self._turns = self._turns[-self.max_keep:]
        return entry

    def list(self, who: str = "user", limit: int = 100) -> list[dict]:
        """Recent turns formatted for `who` ('user' strips internal/provenance fields; 'audit' keeps them)."""
        with self._lock:
            items = list(self._turns[-max(1, int(limit)):])
        if who == "audit":
            return [{"ts": it["ts"], "q": it["q"], "rec": dict(it["rec"])} for it in items]
        return [{"ts": it["ts"], "q": it["q"], "rec": audience.for_user(it["rec"])} for it in items]

    def seed_chat(self, chat) -> int:
        """Restore recent casual chat turns into `chat.history` on boot so multi-turn memory survives restarts."""
        if chat is None or not hasattr(chat, "history"):
            return 0
        span = max(1, int(getattr(chat, "history_turns", 1) or 1))
        with self._lock:
            items = list(self._turns)
        restored = []
        for it in items:
            rec = it.get("rec") or {}
            if rec.get("kind") == "chat" and rec.get("reply"):
                restored.append({"user": it["q"], "reply": str(rec["reply"])})
        recent = restored[-span:]
        chat.history.extend(recent)
        return len(recent)

    def clear(self, chat=None) -> str | None:
        """Retire the active vault log aside (never delete) and start a fresh board conversation."""
        retired = None
        with self._lock:
            if self.path.is_file():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                dest = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
                i = 1
                while dest.exists():
                    dest = self.path.with_name(f"{self.path.stem}.{stamp}_{i}{self.path.suffix}")
                    i += 1
                self.path.rename(dest)
                retired = dest.name
            self._turns.clear()
        if chat is not None and hasattr(chat, "history"):
            chat.history.clear()
        return retired


__all__ = ["BoardChatStore"]
