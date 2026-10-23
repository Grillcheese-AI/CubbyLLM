"""standin/plugins_guard.py -- a plugin's files pinned by Nick's approval, its code checked, its reach limited
(order 0028).

**This is review plus a pin, not a sandbox.** A plugin's Python runs in the server's process with the server's rights.
What this module does is make sure that what runs is what Nick read and approved, and that the code does not reach
for the network, processes or a shell on its own:

- ``approve(plugin)`` records the SHA-256 of every file in the plugin's folder (its adapter files too, when they live
  elsewhere: ``extra_files``) and the permissions approved, in a local, git-ignored lock file (``CUBBY_PLUGINS_LOCK``,
  default ``standin/data/out/plugins.lock.json``). ``status(plugin)`` is ``ok``, ``needs approval`` (never approved) or
  ``blocked: <file> changed`` / ``... added`` / ``... removed`` / ``permissions changed``. Nothing loads unless ``ok``.
- ``check(plugin)`` reads every ``.py`` file's syntax tree and fails it, with the line, on: an import of a network,
  process, shell or loader module (``socket``, ``urllib``, ``http``, ``requests``, ``subprocess``, ``ctypes``,
  ``multiprocessing``, ``importlib``, ``builtins``, ...); ``eval`` / ``exec`` / ``compile`` / ``__import__``; ``getattr`` /
  ``setattr`` / ``delattr`` with a name that is not a literal; ``sys.modules`` or ``__builtins__``; every ``os.exec*``,
  ``os.spawn*``, ``os.posix_spawn*``, ``os.system`` / ``os.popen`` / ``os.fork`` / ``os.kill``; deleting or moving
  (``shutil.rmtree`` / ``move``, ``os.remove`` / ``unlink`` / ``rmdir`` / ``rename`` / ``replace``, ``.unlink()`` /
  ``.rmdir()`` / ``.rename()``); and writing: ``open`` with a write mode (or a mode that is not a literal), ``.write_text``
  / ``.write_bytes``. Code writes through ``ctx.save`` (its own folder) instead. (Cowork's review probe, 2026-10-22: the
  first scan caught 2 of 12 simple escapes.)
- ``Ctx``: what the code gets instead. ``ctx.fetch(url)`` reaches only the hosts the plugin declared, through the news
  archive's polite fetcher (robots read and obeyed, ``Crawl-delay`` honoured, always ``CubbyLLM/0.1``); ``ctx.data_dir``
  is its only writable folder; ``ctx.log`` writes to the server's log.

Wired: WIRED (``plugins_load``; the CLI ``python -m cubbyllm.plugins`` approves and checks).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import time
import urllib.parse
from pathlib import Path

from cubbyllm.local_paths import data_dir

__wiring__ = "WIRED"

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORTS = {"socket", "urllib", "http", "requests", "httpx", "subprocess", "ctypes", "multiprocessing",
                     "asyncio.subprocess", "urllib3", "aiohttp", "ftplib", "smtplib", "telnetlib", "pty", "importlib",
                     "builtins", "pickle", "marshal", "runpy", "code", "codeop", "webbrowser", "winreg", "signal"}
FORBIDDEN_CALLS = {"os.system", "os.popen", "os.startfile", "os.fork", "os.forkpty", "os.kill", "os.killpg",
                   "__import__", "importlib.import_module", "eval", "exec", "compile", "breakpoint",
                   "shutil.rmtree", "shutil.move", "os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.rename",
                   "os.renames", "os.replace", "os.truncate", "os.chmod", "os.chown"}
FORBIDDEN_PREFIXES = ("os.exec", "os.spawn", "os.posix_spawn")
FORBIDDEN_METHODS = {"unlink", "rmdir", "rename", "write_text", "write_bytes", "symlink_to", "hardlink_to", "chmod"}
DYNAMIC_ATTR = {"getattr", "setattr", "delattr"}


def lock_path() -> Path:
    return Path(os.environ.get("CUBBY_PLUGINS_LOCK") or data_dir("plugins.lock.json"))


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


def _approved_dir(plugin) -> Path:
    return lock_path().parent / "plugins_approved" / plugin.name


def review(plugin, limit: int = 20_000) -> dict:
    """What Approve shows before it acts: each file new, changed, same or removed since the last approval -- a new or
    changed text file with its text (a changed one as a diff against the copy kept at approval), a binary one by size
    and hash -- and ``digest``, the hash of exactly this state, which ``approve(expect=...)`` checks."""
    import difflib
    then = (_read_lock().get(plugin.name) or {}).get("files") or {}
    now, out = files(plugin), []
    for f in sorted(set(now) | set(then)):
        row = {"file": f, "sha256": now.get(f)}
        if f not in now:
            row["status"] = "removed"
        elif f not in then:
            row["status"] = "new"
        else:
            row["status"] = "changed" if now[f] != then[f] else "same"
        path = plugin.folder / f if not f.startswith("ext:") else None
        if row["status"] in ("new", "changed") and path is not None and path.suffix.lower() in TEXT:
            text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
            old = _approved_dir(plugin) / f
            if row["status"] == "changed" and old.is_file():
                row["diff"] = "".join(difflib.unified_diff(old.read_text(encoding="utf-8").splitlines(True),
                                                           text.splitlines(True), f"approved/{f}", f"now/{f}"))[:limit]
            else:
                row["text"] = text[:limit]
        elif row["status"] in ("new", "changed"):
            row["bytes"] = (path.stat().st_size if path is not None else None)
        out.append(row)
    digest = hashlib.sha256(json.dumps([now, plugin.permissions], sort_keys=True).encode()).hexdigest()
    return {"name": plugin.name, "permissions": plugin.permissions, "files": out, "digest": digest,
            "check": check(plugin)}


def approve(plugin, by: str = "nick", expect: str | None = None) -> dict:
    """Pin the plugin as it is now. ``expect``: the digest of the review that was shown -- refused if the files changed
    since (what is approved is what was read)."""
    import shutil as _shutil
    if expect is not None and expect != review(plugin)["digest"]:
        raise Refused(f"{plugin.id} changed since its review was shown: show it again before approving")
    keep = _approved_dir(plugin)
    _shutil.rmtree(keep, ignore_errors=True)
    for p in plugin.folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT and "__pycache__" not in p.parts:
            dst = keep / p.relative_to(plugin.folder)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"), encoding="utf-8")
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
                if (any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS) or m in FORBIDDEN_CALLS
                        or m.startswith(FORBIDDEN_PREFIXES)):
                    out.append(f"{rel}:{node.lineno}: imports {m} (use ctx.fetch / ctx.save)")
            if isinstance(node, ast.Attribute) and _dotted(node) == "sys.modules":
                out.append(f"{rel}:{node.lineno}: reaches sys.modules")
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                out.append(f"{rel}:{node.lineno}: reaches __builtins__")
            if isinstance(node, ast.Call):
                out += [f"{rel}:{node.lineno}: {why}" for why in _call(node)]
    return out


def _call(node: ast.Call) -> list[str]:
    name = _dotted(node.func)
    if name in FORBIDDEN_CALLS or name.startswith(FORBIDDEN_PREFIXES):
        return [f"calls {name}"]
    if name in DYNAMIC_ATTR and (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant)):
        return [f"{name} with a name that is not a literal"]
    if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_METHODS:
        return [f"calls .{node.func.attr}() (write through ctx.save)"]
    if name == "open" or (isinstance(node.func, ast.Attribute) and node.func.attr == "open"):
        mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
        if mode is not None and not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
            return ["open with a mode that is not a literal"]
        if mode is not None and set(mode.value) & set("wax+"):
            return [f"open for writing ({mode.value!r}: write through ctx.save)"]
    return []


class Refused(PermissionError):
    pass


class Ctx:
    """What a plugin's code may use: its declared hosts through the polite fetcher, its own folder, the log."""

    def __init__(self, plugin, data_root, log=print, fetch=None) -> None:
        self.plugin, self.log = plugin, log
        self.data_dir = Path(data_root) / "plugins" / plugin.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._archive = None
        self._fetch = fetch                                    # the server's (cubby_serve.sources.news.get); tests: a fake

    def _inside(self, name: str) -> Path:
        p = (self.data_dir / name).resolve()
        if self.data_dir.resolve() not in p.parents:
            raise Refused(f"{self.plugin.id} may write only in its own folder, not {name!r}")
        return p

    def save(self, name: str, data: bytes) -> Path:
        """Write ``data`` to ``name`` inside the plugin's folder (the only place it writes)."""
        p = self._inside(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def load(self, name: str) -> bytes | None:
        p = self._inside(name)
        return p.read_bytes() if p.is_file() else None

    def fetch(self, url: str) -> tuple[int, bytes]:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if urllib.parse.urlsplit(url).scheme not in ("http", "https") or host not in self.plugin.network:
            raise Refused(f"{self.plugin.id} may not fetch {host or url!r}: declared {self.plugin.network}")
        if self._archive is None:
            from ..sources.news.archive import Archive
            self._archive = Archive(self.data_dir / "_fetch", fetch=self._fetch)   # the server's fetch (order 0031)
        robots = self._archive.robots(host)
        if robots is not None:
            from ..sources.news.source import UA
            from ..sources.robots import refuses
            if refuses(robots, UA, url):
                raise Refused(f"{host}'s robots.txt refuses {url}")
        return self._archive.polite(url)
