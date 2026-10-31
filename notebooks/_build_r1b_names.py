"""Generate r1b_names.ipynb: order 0046's R1b names extract and the EN / FR Wikipedia redirect tables, streamed on Colab
to Drive (Nick's yes, 2026-10-31; docs/RETRIEVAL.md decisions 5 and 6). The notebook is thin; the logic is
validation/retrieval_sets/names_extract.py and validation/retrieval_sets/wiki_redirects.py. No GPU. Re-run after
editing:
    python notebooks/_build_r1b_names.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "r1b_names.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# R1b: every item's EN and FR names, and the Wikipedia redirects (order 0046)

**What this fetches, and only this** (the claims are never read):
- Wikidata's labels (`en`, `fr`, `mul`), aliases (`en`, `fr`) and enwiki / frwiki titles, from the `wikidata-pq` tables
  on the Hugging Face Hub (CC0), release **20260928** (Wikidata's weekly dump of 2026-09-28). The tables are that dump
  split by language and site, so only 8 files are read: **1.36 GB, 119.4M names, an extract of ~1.5 GB**.
- The EN and FR Wikipedia `page` and `redirect` tables (dumps.wikimedia.org, gzipped SQL, **~3.1 GB streamed**), kept
  as article-space redirect pairs only.

Nothing raw is stored: each file is streamed and only the extract lands on Drive, the master copy
(`cubbyllm/r1b_names/`). **CPU only:** pick a CPU runtime (no GPU units spent).

**The stop:** cell 3 prints the plan (bytes per file, the extract's size) and writes it to Drive. Cell 4 runs only
when `GO = True`, after Nick reads the plan. (The plan above was measured from home on 2026-10-31; this cell measures
it again where it runs.)
"""),
    code(r"""
!git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM 2>/dev/null || git -C /content/CubbyLLM pull -q
!pip -q install -U pyarrow huggingface_hub
from google.colab import drive; drive.mount('/content/drive')
import os, subprocess, sys
REPO = '/content/CubbyLLM'
OUT = '/content/drive/MyDrive/cubbyllm/r1b_names'
os.makedirs(OUT, exist_ok=True)
for f in ('validation/retrieval_sets/names_extract.py', 'validation/retrieval_sets/wiki_redirects.py'):
    assert os.path.exists(f'{REPO}/{f}'), f'push {f} first'


def run(*args):
    # stream the script's output as it comes (a captured run shows nothing until it ends)
    p = subprocess.Popen([sys.executable, '-u', *args], cwd=REPO, text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'{args[0]} failed ({p.returncode})'
"""),
    md(r"""
## 1. The plan (reads the files' footers and first row groups, and the dumps' headers)
"""),
    code(r"""
run('validation/retrieval_sets/names_extract.py', 'plan', '--out', OUT)
run('validation/retrieval_sets/wiki_redirects.py', 'plan', '--out', OUT)
"""),
    md(r"""
## 2. The extract (after Nick's go on the plan)

Resumable: a file already on Drive is skipped, so re-running after a disconnect continues where it stopped.
"""),
    code(r"""
GO = False          # Nick: set True after reading the plan above
assert GO, 'read the plan first, then set GO = True'
run('validation/retrieval_sets/names_extract.py', 'names', '--out', OUT)
run('validation/retrieval_sets/wiki_redirects.py', 'pairs', '--out', OUT)
!du -sh "$OUT"/* 2>/dev/null
"""),
    md(r"""
## 3. Done

The extract is on Drive (`cubbyllm/r1b_names/names_20260928/`, `redirects_enwiki.parquet`, `redirects_frwiki.parquet`,
the manifests). **Disconnect the runtime.** Claude Code builds the VM segment from it at home (names and aliases to
items, `cubelang knowledge build`), copies it to the SSD, and scores R1b against the lines written in the order.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
