"""standin/plugins.py -- the plugins' command line (order 0028).

    python -m standin.plugins list    [--dirs plugins]        every plugin, its kinds and its status
    python -m standin.plugins check   <name> [--dirs ...]     the code review (imports, calls) and the pin
    python -m standin.plugins approve <name> [--dirs ...]     Nick's approval: every file's SHA-256 and the permissions
    python -m standin.plugins new     <name> --kinds tool,... a folder to start from (PLUGIN.md and the entry stubs)

``--dirs`` defaults to ``CUBBY_PLUGINS`` or the repo's ``plugins/``. Approval writes the local, git-ignored lock
(``plugins_guard.lock_path``); a file changed afterwards blocks the plugin until it is approved again.

Wired: STANDALONE (a command; the server reads the same lock through ``plugins_load``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

__wiring__ = "STANDALONE"

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import plugins_guard as guard  # noqa: E402
from plugins_index import KINDS, discover  # noqa: E402

STUBS = {
    "tool": ("tool.py", 'class Tool:\n    schema = {"type": "object", "properties": {}, "required": []}\n\n'
                        '    def __init__(self, ctx):\n        self.ctx = ctx\n\n    def run(self, args: dict) -> dict:\n'
                        '        return {}\n'),
    "source": ("source.py", "class Source:\n    def __init__(self, ctx):\n        self.ctx = ctx\n\n"
                            "    def covers(self, question: str) -> float:\n        return 0.0\n\n"
                            "    def answer(self, question: str) -> list[str]:\n        return []\n"),
    "panel": ("panel.json", json.dumps({"title": "", "chart": "list", "call": "source.panel"}, indent=1) + "\n"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m standin.plugins", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["list", "check", "approve", "new"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--dirs", default=os.environ.get("CUBBY_PLUGINS") or str(HERE.parent / "plugins"))
    ap.add_argument("--kinds", default="tool")
    a = ap.parse_args(argv)
    if a.cmd == "new":
        kinds = [k for k in a.kinds.split(",") if k]
        if not a.name or any(k not in KINDS for k in kinds):
            ap.error(f"new <name> --kinds any of {KINDS}")
        d = Path(a.dirs.split(",")[0]) / a.name
        d.mkdir(parents=True, exist_ok=False)
        entry = {k: (f"{STUBS[k][0]}:{STUBS[k][1].split()[1].rstrip(':')}" if k in CODE else STUBS[k][0])
                 for k in kinds if k in STUBS}
        (d / "PLUGIN.md").write_text("---\n" + f"name: {a.name}\nversion: 0.1.0\ndescription: \nexamples: []\nnot: []\n"
                                     f"kinds: [{', '.join(kinds)}]\naudience: public\nlicence: \n"
                                     "requires: {plugins: [], base: null}\npermissions: {network: [], write: false, gpu: false}\n"
                                     f"entry: {json.dumps(entry)}\n---\n\n# {a.name}\n", encoding="utf-8")
        for k in kinds:
            if k in STUBS:
                (d / STUBS[k][0]).write_text(STUBS[k][1], encoding="utf-8")
        print(f"{d}: write the description and ten examples and ten not-lines first, then the code")
        return 0
    idx = discover(a.dirs)
    if a.cmd == "list":
        for name in idx.names():
            p = idx[name]
            print(f"{name:24s} {','.join(p.kinds):18s} {p.audience:8s} {guard.status(p)}")
        for name, why in sorted(idx.errors.items()):
            print(f"{name:24s} (not read) {why}")
        return 0
    if a.name not in idx:
        print(f"no plugin {a.name!r} under {a.dirs}" + (f": {idx.errors[a.name]}" if a.name in idx.errors else ""))
        return 2
    p = idx[a.name]
    bad = guard.check(p)
    for b in bad:
        print("FAIL", b)
    if a.cmd == "check":
        print(f"{p.id}: code {'fails review' if bad else 'passes review'}; pin: {guard.status(p)}")
        return 1 if bad else 0
    if bad:
        print(f"{p.id}: not approved -- its code fails review")
        return 1
    rec = guard.approve(p)
    print(f"{p.id} approved: {len(rec['files'])} files pinned, permissions {json.dumps(p.permissions)}")
    return 0


CODE = ("tool", "source")

if __name__ == "__main__":
    sys.exit(main())
