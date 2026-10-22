"""standin/plugins_index.py -- what plugins exist, read from their front matter alone (order 0028).

A plugin is a folder with a ``PLUGIN.md``: YAML front matter between ``---`` lines, then plain documentation.
``discover(dirs)`` reads the front matter of every ``<dir>/<plugin>/PLUGIN.md`` -- the first lines of one file per
plugin, nothing else: no code is imported, no data or adapter opened -- and keeps it as a ``Plugin``. That index is
the only thing held in memory until a question needs a plugin (``plugins_load``).

The front matter (every key but ``name``, ``description`` and ``kinds`` has a default):

    name: units                       # unique; the plugin is plugin:<name>
    version: 0.1.0
    description: Converts quantities between units of length, mass, temperature, time and volume.
    examples: ["how many km is 5 miles?"]   # questions that must wake it (the routing test)
    not: ["what is the capital of Canada?"] # questions that must not
    kinds: [tool]                     # any of KINDS
    audience: public                  # or internal (order 0016: never in a public response)
    licence: MIT
    requires: {plugins: [], base: null}     # base: the 450M export's hash, for an adapter
    permissions: {network: [], write: false, gpu: false}
    entry: {tool: tool.py:Units}

A folder whose front matter cannot be read, or that breaks a rule above, is left out with its reason in
``Index.errors``; it never stops the others.

Wired: WIRED (``plugins_load``, ``serve_api.py --plugins``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__wiring__ = "WIRED"

KINDS = ("adapter", "world", "source", "tool", "panel")
AUDIENCES = ("public", "internal")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


@dataclass
class Plugin:
    name: str
    description: str
    kinds: list
    folder: Path
    version: str = "0.0.0"
    examples: list = field(default_factory=list)
    not_: list = field(default_factory=list)
    audience: str = "public"
    licence: str = ""
    requires: dict = field(default_factory=dict)
    permissions: dict = field(default_factory=dict)
    entry: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)              # any other key, kept for the kind that reads it

    @property
    def id(self) -> str:
        return f"plugin:{self.name}"

    @property
    def network(self) -> list:
        return [h.lower() for h in (self.permissions or {}).get("network") or []]


class Index:
    def __init__(self) -> None:
        self.plugins: dict[str, Plugin] = {}
        self.errors: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.plugins)

    def __getitem__(self, name: str) -> Plugin:
        return self.plugins[name]

    def __contains__(self, name: str) -> bool:
        return name in self.plugins

    def names(self) -> list[str]:
        return sorted(self.plugins)


def front_matter(path: Path) -> dict:
    """The YAML between the first two ``---`` lines of ``path``, reading no further than the closing line."""
    import yaml
    lines = []
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() != "---":
            raise ValueError("PLUGIN.md must start with a --- line")
        for line in f:
            if line.strip() == "---":
                break
            lines.append(line)
        else:
            raise ValueError("no closing --- line")
    data = yaml.safe_load("".join(lines)) or {}
    if not isinstance(data, dict):
        raise ValueError("the front matter is not a mapping")
    return data


def parse(path: Path) -> Plugin:
    d = front_matter(path)
    name = str(d.get("name") or "")
    if not _NAME.match(name):
        raise ValueError(f"name {name!r}: lowercase letters, digits and dashes")
    if not str(d.get("description") or "").strip():
        raise ValueError("no description: the router reads it")
    kinds = list(d.get("kinds") or [])
    bad = [k for k in kinds if k not in KINDS]
    if not kinds or bad:
        raise ValueError(f"kinds {kinds}: each one of {KINDS}")
    audience = str(d.get("audience") or "public")
    if audience not in AUDIENCES:
        raise ValueError(f"audience {audience!r}: public or internal")
    known = {"name", "version", "description", "examples", "not", "kinds", "audience", "licence", "requires",
             "permissions", "entry"}
    return Plugin(name=name, description=str(d["description"]).strip(), kinds=kinds, folder=path.parent,
                  version=str(d.get("version") or "0.0.0"), examples=[str(x) for x in d.get("examples") or []],
                  not_=[str(x) for x in d.get("not") or []], audience=audience, licence=str(d.get("licence") or ""),
                  requires=dict(d.get("requires") or {}), permissions=dict(d.get("permissions") or {}),
                  entry=dict(d.get("entry") or {}), extra={k: v for k, v in d.items() if k not in known})


def discover(dirs) -> Index:
    """Every plugin under ``dirs`` (a list, or a comma-separated string): ``<dir>/<plugin>/PLUGIN.md``."""
    idx = Index()
    if isinstance(dirs, str):
        dirs = [d for d in dirs.split(",") if d.strip()]
    for d in dirs or []:
        root = Path(d).expanduser()
        if not root.is_dir():
            idx.errors[str(d)] = "not a folder"
            continue
        for md in sorted(root.glob("*/PLUGIN.md")):
            try:
                p = parse(md)
            except Exception as e:                         # noqa: BLE001 -- one bad folder never stops the rest
                idx.errors[md.parent.name] = f"{type(e).__name__}: {str(e)[:160]}"
                continue
            if p.name in idx.plugins:
                idx.errors[md.parent.name] = f"name {p.name!r} already taken by {idx.plugins[p.name].folder.name}"
                continue
            idx.plugins[p.name] = p
    return idx
