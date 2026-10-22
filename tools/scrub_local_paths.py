"""tools/scrub_local_paths.py -- rewrite local paths in committed text files to the names ``cubbyllm.local_paths``
resolves (order 0030).

A known root becomes its name: a user's GitHub checkout of this repo ``$CUBBY_REPO``, the folder of sibling repos
``$CUBBY_GITHUB``, a user folder ``~``; the trained models ``$CUBBY_MODELS``, the synced Drive ``$CUBBY_DRIVE``, the
training data ``$CUBBY_TRAINING_DATA``, the datasets ``$CUBBY_DATASETS``, the worlds ``$CUBBY_WORLDS``; the platform's
``$ProgramFiles``, ``$WINDIR``, ``$TEMP``; any other drive root ``$CUBBY_DATA_ROOT``. PowerShell files get
``$env:NAME``, cmd files ``%NAME%``. In Python, a string that is code (not a docstring) is also wrapped in
``_lp(...)`` (``cubbyllm.local_paths.expand``) and the import added, so a default keeps resolving on the machine that
has the folder; docstrings and comments are rewritten as text. A Python file that no longer parses is left unwritten
and listed.

    python tools/scrub_local_paths.py [--write] [paths ...]     (default: every committed text file; a dry run)

Wired: STANDALONE (a maintenance command).
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

__wiring__ = "STANDALONE"

REPO = Path(__file__).resolve().parents[1]
TEXT = (".py", ".md", ".json", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".js", ".html", ".css", ".sh", ".ps1",
        ".cmd", ".bat", ".jsonl", ".csv", ".tsv", ".ipynb", ".log", ".svg")
S = r"[\\/]{1,2}"                                          # a separator, single or JSON-escaped
D = r"(?<![A-Za-z0-9\\$%])[A-Za-z]:" + S                   # a drive root
USER = r"[^\\/\s\"'`]+"
RULES = [  # (pattern, name); a name of None means the home folder
    (D + rf"Users{S}{USER}{S}Documents{S}GitHub{S}CubbyLLM(?![\w-])", "CUBBY_REPO"),
    (D + rf"Users{S}{USER}{S}Documents{S}GitHub(?![\w-])", "CUBBY_GITHUB"),
    (D + rf"Users{S}{USER}(?![\w-])", None),
    (r"/c/Users/" + USER + r"/Documents/GitHub/CubbyLLM(?![\w-])", "CUBBY_REPO"),
    (r"/c/Users/" + USER + r"/Documents/GitHub(?![\w-])", "CUBBY_GITHUB"),
    (r"(?<![\w.])/(?:c/)?Users/" + USER + r"(?![\w-])", None),
    (D + r"CUBBY-TRAINED-MODELS(?![\w-])", "CUBBY_MODELS"),
    (D + r"My Drive(?![\w-])", "CUBBY_DRIVE"),
    (D + r"grillcheese_training_data(?![\w-])", "CUBBY_TRAINING_DATA"),
    (D + r"datasets(?![\w-])", "CUBBY_DATASETS"),
    (D + r"cubby_worlds(?![\w-])", "CUBBY_WORLDS"),
    (D + r"Program Files(?: \(x86\))?(?![\w-])", "ProgramFiles"),
    (D + r"Windows(?![\w-])", "WINDIR"),
    (D + r"tmp(?![\w-])", "TEMP"),
    (r"(?<![A-Za-z0-9\\$%])[A-Za-z]:(?:/|\\\\|\\(?![ntr\"\\u]))(?=[A-Za-z0-9_.(~-])", "CUBBY_DATA_ROOT/"),
]
RULES = [(re.compile(p), n) for p, n in RULES]
VENV = re.compile(r"\.venv-dml\b")


def name(n: str | None, ext: str) -> str:
    slash = n.endswith("/") if n else False
    n = n.rstrip("/") if n else None
    if ext == ".ps1":
        out = "$env:USERPROFILE" if n is None else f"$env:{n}"
    elif ext in (".cmd", ".bat"):
        out = "%USERPROFILE%" if n is None else f"%{n}%"
    else:
        out = "~" if n is None else f"${n}"
    return out + ("/" if slash else "")


def rewrite(text: str, ext: str) -> str:
    for rx, n in RULES:
        text = rx.sub(lambda m, n=n: name(n, ext), text)
    return VENV.sub("<venv>", text)


_ANY = re.compile("|".join(f"(?:{rx.pattern})" for rx, _n in RULES) + f"|{VENV.pattern}")


_CANDIDATE = re.compile(r"(?<![A-Za-z0-9\\$%])[A-Za-z]:[\\/]|Users[\\/]|venv-dml")   # cheap; the rules run around it


def hit(text: str) -> bool:
    """Does ``text`` hold a local path (any rule's pattern, or the venv's name)? Each cheap candidate is checked
    against the rules in a window around it (a path's start lies at most a user folder before ``Users``)."""
    for m in _CANDIDATE.finditer(text):
        lo = max(0, m.start() - 40)
        if _ANY.search(text, lo, m.end() + 120):
            return True
    return False


def _docstrings(tree) -> set[int]:
    lines = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            lines.add(body[0].lineno)
    return lines


def python(src: str, depth: int) -> str:
    docs = _docstrings(ast.parse(src))
    toks, wrapped = list(tokenize.generate_tokens(io.StringIO(src).readline)), False
    lines = src.splitlines(keepends=True)
    edits = []                                            # (start, end, new) per token
    runs, run = [], []                                    # adjacent strings (implicit concatenation) wrap as one
    for t in toks:
        if t.type == tokenize.STRING:
            run.append(t)
        elif t.type not in (tokenize.NL, tokenize.COMMENT) and run:
            runs.append(run)
            run = []
        if t.type == tokenize.COMMENT and hit(t.string):
            edits.append((t.start, t.end, rewrite(t.string, ".py")))
    runs += [run] if run else []
    for run in runs:
        if not any(hit(t.string) for t in run):
            continue
        path = any(rx.search(t.string) for t in run for rx, _n in RULES)    # a folder, not only the venv's name
        code = path and run[0].start[0] not in docs and all(t.start[0] == t.end[0] for t in run)
        for i, t in enumerate(run):
            new = rewrite(t.string, ".py")
            if code:
                new = ("_lp(" if i == 0 else "") + new + (")" if i == len(run) - 1 else "")
                wrapped = True
            if new != t.string:
                edits.append((t.start, t.end, new))
    for (r0, c0), (r1, c1), new in sorted(edits, reverse=True):
        if r0 == r1:
            lines[r0 - 1] = lines[r0 - 1][:c0] + new + lines[r0 - 1][c1:]
        else:                                             # a multi-line string (a docstring): text only
            block = "".join(lines[r0 - 1:r1])
            start = sum(len(l) for l in lines[:r0 - 1])
            whole = "".join(lines)
            a, b = start + c0, sum(len(l) for l in lines[:r1 - 1]) + c1
            whole = whole[:a] + new + whole[b:]
            lines = whole.splitlines(keepends=True)
            del block
    out = "".join(lines)
    if wrapped and "from cubbyllm.local_paths import expand as _lp" not in out:
        imp = ("import pathlib as _pathlib  # noqa: E402\nimport sys as _sys  # noqa: E402\n"
               f"_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[{depth}]))      # the repo root, for:\n"
               "from cubbyllm.local_paths import expand as _lp  # noqa: E402 -- local folders by name (order 0030)\n")
        tree = ast.parse(out)
        at = 0
        for st in tree.body:
            if (isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and st is tree.body[0]) or \
                    (isinstance(st, ast.ImportFrom) and st.module == "__future__"):
                at = st.end_lineno
        ls = out.splitlines(keepends=True)
        out = "".join(ls[:at]) + ("\n" if at else "") + imp + "".join(ls[at:])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    files = a.paths or subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.splitlines()
    changed, broken = 0, []
    for f in files:
        p = REPO / f
        if p.suffix.lower() not in TEXT or not p.is_file() or p.resolve() == Path(__file__).resolve():
            continue
        src = p.read_text(encoding="utf-8", errors="surrogateescape")
        if not hit(src):
            continue
        try:
            out = python(src, len(Path(f).parts) - 1) if p.suffix == ".py" else rewrite(src, p.suffix.lower())
            if p.suffix == ".py":
                ast.parse(out)
        except (SyntaxError, tokenize.TokenError, ValueError) as e:
            broken.append(f"{f}: {type(e).__name__}: {e}")
            continue
        changed += 1
        print(("wrote " if a.write else "would write ") + f)
        if a.write:
            p.write_text(out, encoding="utf-8", errors="surrogateescape", newline="")
    for b in broken:
        print("LEFT (fix by hand):", b)
    print(f"{changed} files {'rewritten' if a.write else 'to rewrite'}, {len(broken)} left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
