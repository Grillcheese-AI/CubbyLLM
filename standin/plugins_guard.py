"""standin/plugins_guard.py -- a plugin's files pinned by Nick's approval, its code checked, its reach limited
(order 0028).

**This is review plus a pin, not a sandbox.** A plugin's Python runs in the server's process with the server's rights.
What this module does is make sure that what runs is what Nick read and approved, and that the code does not reach
for the network, processes or a shell on its own:

- ``approve(plugin)`` records the SHA-256 of every file in the plugin's folder (its adapter files too, when they live
  elsewhere: ``extra_files``) and the permissions approved, in a local, git-ignored lock file (``CUBBY_PLUGINS_LOCK``,
  default ``standin/data/out/plugins.lock.json``). ``status(plugin)`` is ``ok``, ``needs approval`` (never approved) or
  ``blocked: <file> changed`` / ``... added`` / ``... removed`` / ``permissions changed``. Nothing loads unless ``ok``.
- ``check(plugin)`` reads every ``.py`` file's syntax tree: an import of ``socket``, ``urllib``, ``http``, ``requests``,
  ``httpx``, ``subprocess``, ``ctypes``, ``multiprocessing`` or ``asyncio.subprocess``, or a call to ``os.system`` /
  ``os.popen`` / ``os.exec*`` / ``os.spawn*`` / ``__import__`` / ``importlib``, fails it, with the line.
- ``Ctx``: what the code gets instead. ``ctx.fetch(url)`` reaches only the hosts the plugin declared, through the news
  archive's polite fetcher (robots read and obeyed, ``Crawl-delay`` honoured, always ``CubbyLLM/0.1``); ``ctx.data_dir``
  is its only writable folder; ``ctx.log`` writes to the server's log.

Wired: WIRED (``plugins_load``; the CLI ``python -m standin.plugins`` approves and checks).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import time
import urllib.parse
from pathlib import Path

__wiring__ = "WIRED"

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORTS = {"socket", "urllib", "http", "requests", "httpx", "subprocess", "ctypes", "multiprocessing",
                     "asyncio.subprocess", "urllib3", "aiohttp", "ftplib", "smtplib", "telnetlib", "pty"}
FORBIDDEN_CALLS = {"os.system", "os.popen", "os.execv", "os.execve", "os.execl", "os.execlp", "os.execvp",
                   "os.spawnl", "os.spawnv", "os.spawnve", "os.startfile", "__import__", "importlib.import_module"}


def lock_path() -> Path:
    return Path(os.environ.get("CUBBY_PLUGINS_LOCK") or ROOT / "standin" / "data" / "out" / "plugins.lock.json")


def _read_lock() -> dict:
    p = lock_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _write_lock(d: dict) -> None:
    p = lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def extra_files(plugin) -> list[Path]:
    """Files outside the folder that the plugin loads (an adapter kept beside the models): pinned the same way."""
    out = []
    ad = plugin.extra.get("adapter") or {}
    d = os.environ.get(ad.get("dir_env") or "") if isinstance(ad, dict) else None
    if d and Path(d).is_dir():
        out += sorted(p for p in Path(d).iterdir() if p.is_file() and p.name.endswith(("_lora.json", "_lora.safetensors")))
    return out


def files(plugin) -> dict[str, str]:
    """Every file the plugin brings, by its path relative to the folder (an outside file by ``ext:<name>``) -> SHA-256."""
    out = {}
    for p in sorted(plugin.folder.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            out[p.relative_to(plugin.folder).as_posix()] = _sha(p)
    for p in extra_files(plugin):
        out[f"ext:{p.name}"] = _sha(p)
    return out


TEXT = (".py", ".md", ".json", ".txt", ".yaml", ".yml", ".csv", ".jsonl")


def _sha(p: Path) -> str:
    """A file's SHA-256; a text file's with its line endings as LF (a checkout that turns them to CRLF changes nothing
    a reviewer reads, and must not block the plugin). Binary files (an adapter's weights) byte for byte."""
    h = hashlib.sha256()
    if p.suffix.lower() in TEXT:
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def approve(plugin, by: str = "nick") -> dict:
    lock = _read_lock()
    lock[plugin.name] = {"files": files(plugin), "permissions": plugin.permissions, "version": plugin.version,
                         "approved_by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_lock(lock)
    return lock[plugin.name]


def revoke(plugin) -> None:
    lock = _read_lock()
    lock.pop(plugin.name, None)
    _write_lock(lock)


def status(plugin) -> str:
    rec = _read_lock().get(plugin.name)
    if rec is None:
        return "needs approval"
    now, then = files(plugin), rec.get("files") or {}
    for f in sorted(set(now) | set(then)):
        if f not in then:
            return f"blocked: {f} added"
        if f not in now:
            return f"blocked: {f} removed"
        if now[f] != then[f]:
            return f"blocked: {f} changed"
    if (rec.get("permissions") or {}) != (plugin.permissions or {}):
        return "blocked: permissions changed"
    return "ok"


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check(plugin) -> list[str]:
    """Every reason the plugin's code fails review: [] when it passes."""
    out = []
    for p in sorted(plugin.folder.rglob("*.py")):
        rel = p.relative_to(plugin.folder).as_posix()
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as e:
            out.append(f"{rel}:{e.lineno}: does not parse")
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS) or m in FORBIDDEN_CALLS:
                    out.append(f"{rel}:{node.lineno}: imports {m} (use ctx.fetch / ctx.data_dir)")
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name in FORBIDDEN_CALLS:
                    out.append(f"{rel}:{node.lineno}: calls {name}")
    return out


class Refused(PermissionError):
    pass


class Ctx:
    """What a plugin's code may use: its declared hosts through the polite fetcher, its own folder, the log."""

    def __init__(self, plugin, data_root, log=print, fetch=None) -> None:
        self.plugin, self.log = plugin, log
        self.data_dir = Path(data_root) / "plugins" / plugin.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._archive = None
        self._fetch = fetch                                    # tests hand in a fake; default: the news fetcher

    def fetch(self, url: str) -> tuple[int, bytes]:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if urllib.parse.urlsplit(url).scheme not in ("http", "https") or host not in self.plugin.network:
            raise Refused(f"{self.plugin.id} may not fetch {host or url!r}: declared {self.plugin.network}")
        if self._archive is None:
            from news_archive import Archive, _get
            self._archive = Archive(self.data_dir / "_fetch", fetch=self._fetch or _get)
        robots = self._archive.robots(host)
        if robots is not None:
            from news_source import UA
            from robots_rules import refuses
            if refuses(robots, UA, url):
                raise Refused(f"{host}'s robots.txt refuses {url}")
        return self._archive.polite(url)
