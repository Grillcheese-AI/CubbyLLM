"""Order 0033 (Cowork's answer 6 in 0031's review): one data root. Every data place the packages keep resolves under
``cubbyllm.local_paths.data_dir()`` -- ``$CUBBY_DATA``, else today's place ``standin/data/out`` -- and no module spells
the old place in its code; the mover takes what is there to a new root and never overwrites.
Run: python -m pytest tests/test_data_root.py
"""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm import local_paths  # noqa: E402

# each data place a module keeps, as the module names it when it loads
PLACES = """
import json
from cubbyllm.sources import web, wikidata_aliases
from cubbyllm.sources.news import feeds
from cubbyllm.speech import audience
from cubbyllm.reasoning import lexicon
from cubbyllm.worlds import hdc_store, hdc_world, hygiene, wikikg
from cubbyllm.plugins import guard
from cubby_serve import keys, ledger
from cubby_serve.cli import parser
a = parser().parse_args([])
print(json.dumps([str(p) for p in (web.CACHE, wikidata_aliases.CACHE, wikidata_aliases.PROPERTY_TABLE, feeds.CACHE,
                                   audience.TOKEN_FILE, lexicon.DEFAULT, hdc_store.DEFAULT, hdc_world.OUT, hygiene.OUT,
                                   wikikg.DATA_DIR, guard.lock_path(), keys.HOST_KEY_FILE, ledger.DEFAULT_DB,
                                   a.history, a.learned, a.skills)]))
"""


def test_the_root_is_todays_place_unless_cubby_data_names_another(monkeypatch, tmp_path):
    monkeypatch.delenv("CUBBY_DATA", raising=False)
    monkeypatch.setattr(local_paths, "_cache", {})                   # not this machine's paths.local.json
    assert local_paths.data_dir() == ROOT / "standin" / "data" / "out"
    monkeypatch.setenv("CUBBY_DATA", str(tmp_path))
    assert local_paths.data_dir("web_cache") == tmp_path / "web_cache"


def test_every_data_place_the_packages_keep_is_under_the_root(tmp_path):
    env = {**os.environ, "CUBBY_DATA": str(tmp_path), "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", PLACES], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    places = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(places) == 16
    assert all(Path(p).resolve().is_relative_to(tmp_path.resolve()) for p in places), places


def _code_strings(path: Path) -> list[tuple[int, str]]:
    """The string tokens of a module's code: docstrings left out."""
    src = path.read_text(encoding="utf-8")
    docs = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docs |= set(range(body[0].lineno, body[0].end_lineno + 1))
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in (tokenize.NL, tokenize.NEWLINE,
                                                                                              tokenize.COMMENT)]
    return [(t.start[0], t.string) for t in toks if t.start[0] not in docs]


def test_no_module_names_the_old_place_in_its_code():
    bad = []
    for pkg in ("cubbyllm", "cubby_serve"):
        for f in sorted((ROOT / pkg).rglob("*.py")):
            if f == ROOT / "cubbyllm" / "local_paths.py":
                continue                                             # the root's own default, `DATA_DEFAULT`
            toks = _code_strings(f)
            for i, (line, s) in enumerate(toks):
                joined = s.strip("\"'rbf").replace("\\\\", "/").replace("\\", "/")
                spelt = "standin/data/out" in joined and " " not in joined   # a path; a help text may name the default
                pieces = s.strip("\"'") == "data" and i + 2 < len(toks) and toks[i + 1][1] in ("/", ",") \
                    and toks[i + 2][1].strip("\"'") == "out"
                if spelt or pieces:
                    bad.append(f"{f.relative_to(ROOT)}:{line}")
    assert not bad, f"data places outside data_dir(): {bad}"


def test_the_mover_moves_what_is_there_and_never_overwrites(tmp_path):
    spec = importlib.util.spec_from_file_location("move_data_root", ROOT / "tools" / "move_data_root.py")
    mover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mover)
    src, dst = tmp_path / "old", tmp_path / "new"
    (src / "web_cache").mkdir(parents=True)
    (src / "web_cache" / "page.json").write_text("{}", encoding="utf-8")
    (src / "host.key").write_text("old", encoding="utf-8")
    dst.mkdir()
    (dst / "host.key").write_text("new", encoding="utf-8")
    go, kept = mover.move(src, dst)
    assert [e.name for e in go] == ["web_cache"] and [e.name for e in kept] == ["host.key"]
    assert (dst / "web_cache" / "page.json").exists() and not (src / "web_cache").exists()
    assert (dst / "host.key").read_text(encoding="utf-8") == "new" and (src / "host.key").read_text(encoding="utf-8") == "old"
