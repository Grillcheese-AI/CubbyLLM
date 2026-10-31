"""names_extract -- order 0046, R1b: the names extract (Nick's yes, 2026-10-31; ``docs/RETRIEVAL.md`` decisions 5 and 6).

Every Wikidata item's EN and FR names, and nothing else: its labels (``en``, ``fr`` and ``mul``, the default label that
holds in every language -- 10.3M items have one and no ``en`` label), its aliases (``en``, ``fr``) and its enwiki and
frwiki titles. Never its claims. Read from the ``wikidata-pq`` tables on the Hugging Face Hub
(``permutans/wikidata-labels``, ``-aliases``, ``-links``; CC0), pinned to the release built from Wikidata's weekly dump
of 2026-09-28. Those tables are the dump split by language and by site, so only the EN, FR and ``mul`` files are read:
about 1.4 GB, where the same columns of the 959 GB dump copy (``philippesaade/wikidata``, the order's first reading)
hold every language in each row. Only the ``id`` and the name column of each file are read.

Two stages, as the order asks:
- ``plan``: each file's size, rows and row groups from its footer, the bytes the run will read, and the extract's size
  projected from the first row group of each file. Nothing is written but the plan. Nick says go on it.
- ``names``: one parquet per source on the output folder (Drive, the master copy): ``id``, ``name``, ``kind``
  (label / alias / title), ``lang`` (en / fr / mul). No raw file is kept: each is read column by column over HTTP.

    python validation/retrieval_sets/names_extract.py plan  --out <folder>
    python validation/retrieval_sets/names_extract.py names --out <folder>

Wired: STANDALONE (a data builder; the notebook ``notebooks/r1b_names.ipynb`` runs it on Colab).
"""
from __future__ import annotations

import argparse
import json
import os
import time

__wiring__ = "STANDALONE"

REV = "20260928"
# (repo, folder, the column holding the name, kind, lang)
SOURCES = [("permutans/wikidata-labels", "en", "value", "label", "en"),
           ("permutans/wikidata-labels", "fr", "value", "label", "fr"),
           ("permutans/wikidata-labels", "mul", "value", "label", "mul"),
           ("permutans/wikidata-aliases", "en", "value", "alias", "en"),
           ("permutans/wikidata-aliases", "fr", "value", "alias", "fr"),
           ("permutans/wikidata-links", "enwiki", "title", "title", "en"),
           ("permutans/wikidata-links", "frwiki", "title", "title", "fr")]


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def files(fs, rev: str = REV) -> list[dict]:
    """Every parquet file the run reads, with its source's columns."""
    out = []
    for repo, folder, col, kind, lang in SOURCES:
        base = f"datasets/{repo}@{rev}/{folder}"
        for e in sorted(fs.ls(base, detail=True), key=lambda e: e["name"]):
            if e["name"].endswith(".parquet"):
                out.append({"path": e["name"], "size": int(e.get("size") or 0), "repo": repo, "folder": folder,
                            "col": col, "kind": kind, "lang": lang})
    return out


def footer(open_fn, f: dict) -> dict:
    """Rows, row groups, and the compressed bytes of the two columns read (the rest is never fetched)."""
    import pyarrow.parquet as pq
    with open_fn(f["path"]) as h:
        m = pq.ParquetFile(h).metadata
        names = [m.schema.column(i).name for i in range(m.num_columns)]
        want = {names.index("id"), names.index(f["col"])}
        read = sum(m.row_group(g).column(c).total_compressed_size for g in range(m.num_row_groups) for c in want)
        return {"rows": m.num_rows, "row_groups": m.num_row_groups, "bytes_read": read}


def table(open_fn, f: dict, row_groups=None):
    """The file's names as an arrow table: id, name, kind, lang."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    with open_fn(f["path"]) as h:
        pf = pq.ParquetFile(h)
        groups = range(pf.metadata.num_row_groups) if row_groups is None else row_groups
        t = pf.read_row_groups(list(groups), columns=["id", f["col"]])
    n = t.num_rows
    return pa.table({"id": t.column("id"), "name": t.column(f["col"]),
                     "kind": pa.array([f["kind"]] * n, pa.string()), "lang": pa.array([f["lang"]] * n, pa.string())})


def plan(open_fn, fl: list[dict]) -> dict:
    """Bytes to read, rows, and the extract's projected size (the first row group of each file, written as parquet)."""
    import io
    import pyarrow.parquet as pq
    rows = []
    for f in fl:
        ft = footer(open_fn, f)
        t = table(open_fn, f, row_groups=[0])
        buf = io.BytesIO()
        pq.write_table(t, buf, compression="zstd")
        per_row = buf.tell() / max(t.num_rows, 1)
        rows.append(dict(f, **ft, projected_bytes=round(per_row * ft["rows"])))
    return {"rev": REV, "files": rows, "files_n": len(rows), "source_bytes": sum(r["size"] for r in rows),
            "bytes_read": sum(r["bytes_read"] for r in rows), "names": sum(r["rows"] for r in rows),
            "projected_extract_bytes": sum(r["projected_bytes"] for r in rows)}


def extract(open_fn, fl: list[dict], out: str) -> dict:
    """One parquet per source file under `out`/names_<rev>/; a file already written is skipped (resumable)."""
    import pyarrow.parquet as pq
    d = os.path.join(out, f"names_{REV}")
    os.makedirs(d, exist_ok=True)
    done = []
    for f in fl:
        dst = os.path.join(d, f"{f['kind']}_{f['lang']}_{os.path.basename(f['path'])}")
        if not os.path.exists(dst):
            t0 = time.time()
            t = table(open_fn, f)
            pq.write_table(t, dst + ".part", compression="zstd")
            os.replace(dst + ".part", dst)
            print(f"{dst}: {t.num_rows:,} names in {time.time() - t0:.0f} s", flush=True)
        done.append({"source": f["path"], "out": os.path.basename(dst), "bytes": os.path.getsize(dst)})
    man = {"rev": REV, "sources": [s[:2] for s in SOURCES], "files": done, "bytes": sum(x["bytes"] for x in done)}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"), indent=1)
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "names"))
    ap.add_argument("--out", required=True, help="the output folder (Drive, on Colab)")
    a = ap.parse_args()
    fs = _fs()
    fl = files(fs)
    opener = lambda p: fs.open(p, "rb")             # noqa: E731
    if a.stage == "plan":
        p = plan(opener, fl)
        os.makedirs(a.out, exist_ok=True)
        json.dump(p, open(os.path.join(a.out, f"names_plan_{REV}.json"), "w", encoding="utf-8"), indent=1)
        for r in p["files"]:
            print(f"{r['repo'].split('/')[1]:>17}/{r['folder']:<7} {r['size'] / 1e6:8.1f} MB  {r['rows']:>12,} rows  "
                  f"read {r['bytes_read'] / 1e6:8.1f} MB  -> ~{r['projected_bytes'] / 1e6:7.1f} MB")
        print(f"{p['files_n']} files, {p['source_bytes'] / 1e9:.2f} GB on the Hub, {p['bytes_read'] / 1e9:.2f} GB read, "
              f"{p['names']:,} names, extract ~{p['projected_extract_bytes'] / 1e9:.2f} GB")
    else:
        m = extract(opener, fl, a.out)
        print(f"{len(m['files'])} files, {m['bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
