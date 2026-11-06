"""Generate wikidata_dump.ipynb: order 0055 step 1. Nick runs it, signed in to the Google account whose Drive holds
the dump (the 5 TB one): the dump downloads once into that Drive, then one pass counts it and draws the 1% sample.
No GPU. Re-run after editing:
    python notebooks/_build_wikidata_dump.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "wikidata_dump.ipynb"


def _src(text):
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


CELLS = [
    md(r"""
# 0055 step 1: the Wikidata dumps, downloaded once and counted

**Part A** is Wikimedia's JSON dump (done 2026-11-06; its download check is run again after a Drive lag stopped it).
**Part B** is 0055's source dump since Cowork's addendum: the WikiKGQA 2026 challenge's own Wikidata (Turtle, 121 GB,
CC0, Zenodo record 22661413; Nick's yes). Run the first cell, then A1 (the check only: every part is already in), then
B1 and B2.

**Runtime:** CPU, high RAM (no GPU). **Sign in to the account whose Drive has the room** (the 5 TB one): the dump
(about 104 GB, bz2) is kept there as the source copy, in 1 GB parts.

1. **Download:** two connections at a time (Wikimedia's limit), in parts. If the runtime drops, run the cell again: it
   resumes at the first missing part. At the end the parts' sha1 is checked against Wikimedia's (a part the Drive
   shows late after its rename is waited for; if the check still stops on a missing part, run the cell again).
2. **Count:** one pass over the parts, never unpacked to disk: the counts (`counts.json`) and the 1% sample
   (`sample_1pct.jsonl.gz`), written to the Drive. If the runtime drops here, the pass starts over (the download is
   kept). The pass hashes the parts too (checked against the download's `manifest.json` when there is one) and stops if
   a part is missing or corrupt.

Keep this tab open while it runs. **Disconnect when the count prints its summary.** Then tell Claude Code.
"""),
    code(r"""
import os, shutil, subprocess, sys
if not os.path.exists('/content/CubbyLLM'):
    !git clone -q https://github.com/Grillcheese-AI/CubbyLLM.git /content/CubbyLLM
else:
    !cd /content/CubbyLLM && git pull -q --ff-only
!apt-get -qq install -y lbzip2 > /dev/null
!pip -q install orjson
from google.colab import drive; drive.mount('/content/drive')
REPO = '/content/CubbyLLM'
HOME = '/content/drive/MyDrive/wikidata'                 # a folder at the Drive's root (Nick)
PARTS, COUNT = f'{HOME}/dump_parts', f'{HOME}/count'
os.makedirs(PARTS, exist_ok=True); os.makedirs(COUNT, exist_ok=True)
free = shutil.disk_usage('/content/drive').free / 1e9
print(f'Drive free: {free:.0f} GB; cores: {os.cpu_count()}; lbzip2: {bool(shutil.which("lbzip2"))}')
assert free > 250, 'this Drive is too small: sign in to the account with the room'

def step(script, *args):
    cmd = [sys.executable, '-u', f'validation/fact_store/{script}', *args]
    with open(f'{HOME}/{script}.log', 'a', encoding='utf-8', buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=''); f.write(line)
        assert p.wait() == 0, f'{script} failed ({p.returncode})'
"""),
    md(r"""
## A1. Wikimedia's JSON: download (resumable: run again if the runtime drops)
"""),
    code(r"""
step('dump_fetch.py', '--out', PARTS)
print(open(f'{PARTS}/manifest.json').read())
"""),
    md(r"""
## A2. Wikimedia's JSON: count, and the 1% sample (done: skip)
"""),
    code(r"""
step('dump_count.py', '--parts', PARTS, '--out', COUNT, '--workers', str(os.cpu_count()))
"""),
    md(r"""
## B1. The WikiKGQA dump (Turtle): download, checked against Zenodo's md5 (resumable)
"""),
    code(r"""
KPARTS, KCOUNT = f'{HOME}/kgqa_parts', f'{HOME}/kgqa_count'
os.makedirs(KPARTS, exist_ok=True); os.makedirs(KCOUNT, exist_ok=True)
step('dump_fetch.py', '--out', KPARTS, '--zenodo', '22661413:wikikgqa-wikidata-all-dump.ttl.bz2')
print(open(f'{KPARTS}/manifest.json').read())
"""),
    md(r"""
## B2. The WikiKGQA dump: count, the 1% sample (the same ids as A2's), its date and its redirects
"""),
    code(r"""
step('ttl_count.py', '--parts', KPARTS, '--out', KCOUNT, '--workers', str(os.cpu_count()))
"""),
    md(r"""
## Done

**Disconnect the runtime.** At this Drive's root: `wikidata/count/` and `wikidata/dump_parts/` (Part A),
`wikidata/kgqa_count/` and `wikidata/kgqa_parts/` (Part B). Then tell Claude Code.
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
