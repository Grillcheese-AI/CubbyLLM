"""A library of ebooks -> plain text the history-graph reader takes (build_history_graph.py --src), one .txt per
book, for every book that is not text yet. Local only: no model, no network.

    python standin/data/convert_books.py --library <library root> --have <text folder> [--have <another>]
        --out <text output folder> --scratch <a folder on a drive with room> [--workers 6] [--archives]

What counts as a book: .epub, .azw3/.azw/.mobi (if the `mobi` package is installed), .htmlz, .pdf -- loose in the
library's folders and, with --archives, inside its .rar archives (listed and extracted one at a time with 7-Zip into
--scratch, deleted once converted). One copy per title: the same book in several formats is converted once, from the
best text source (epub, then the Kindle formats, then html, then pdf), and a title already present as text under
any --have folder is skipped. A pdf whose text layer is too thin (a scan: fewer than 400 characters a page) is not
converted and is listed in the manifest as needing OCR. Resumable: a book whose .txt exists is not redone.
"""
from __future__ import annotations

import argparse, collections, concurrent.futures as cf, html.parser, json, os, pathlib, re, shutil, subprocess, sys, time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

PREF = {"epub": 0, "azw3": 1, "azw": 2, "mobi": 3, "htmlz": 4, "pdf": 5}
SEVEN_ZIP = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
PDFTOTEXT = shutil.which("pdftotext") or r"C:\Program Files\Git\mingw64\bin\pdftotext.exe"
MIN_CHARS_PER_PAGE = 400


def title_key(name: str) -> str:
    s = re.sub(r"\.(pdf|epub|azw3|azw|mobi|htmlz|djvu|doc|azw4|txt)$", "", name, flags=re.I)
    s = re.sub(r"\.(pdf|epub)$", "", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]|\([^)]*(retail|scribd|v\d\.\d|epub|pdf)[^)]*\)|\(\d\)", " ", s, flags=re.I)
    return " ".join(re.sub(r"[^\w]+", " ", s.lower()).split())[:90]


class _Text(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head") and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(markup: str) -> str:
    p = _Text()
    p.feed(markup)
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text).strip()


def epub_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        order = []
        try:
            container = z.read("META-INF/container.xml").decode("utf-8", "replace")
            opf_path = re.search(r'full-path="([^"]+)"', container).group(1)
            opf = z.read(opf_path).decode("utf-8", "replace")
            base = os.path.dirname(opf_path)
            items = {m.group(1): m.group(2) for m in re.finditer(r'<item\b[^>]*\bid="([^"]+)"[^>]*\bhref="([^"]+)"', opf)}
            items.update({m.group(2): m.group(1) for m in re.finditer(r'<item\b[^>]*\bhref="([^"]+)"[^>]*\bid="([^"]+)"', opf)})
            for m in re.finditer(r'<itemref\b[^>]*\bidref="([^"]+)"', opf):
                href = items.get(m.group(1))
                if href:
                    order.append(os.path.normpath(os.path.join(base, href)).replace("\\", "/"))
        except Exception:
            pass
        if not order:
            order = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        parts = []
        for n in order:
            n = re.sub(r"%20", " ", n.split("#")[0])
            if n in names:
                parts.append(html_text(z.read(n).decode("utf-8", "replace")))
        return "\n\n".join(p for p in parts if p)


def htmlz_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        n = next((x for x in z.namelist() if x.lower().endswith((".html", ".htm"))), None)
        return html_text(z.read(n).decode("utf-8", "replace")) if n else ""


def kindle_text(path: str) -> str:
    import mobi                                                   # optional: pip install mobi
    tmp, out = mobi.extract(path)
    try:
        if out.lower().endswith(".epub"):
            return epub_text(out)
        return html_text(open(out, encoding="utf-8", errors="replace").read())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_COMMON = set("the of and to in a is that was for as with by on it his from at which be were are this had "
              "de la le et les des du en un une der die und das den von zu".split())


def readable(text: str) -> float:
    """The share of words that are the commonest words of English, French or German: ~0.3 in prose, ~0 in a
    text layer whose letters are scrambled."""
    words = re.findall(r"[a-zà-ÿ]+", text[:60000].lower())
    return sum(w in _COMMON for w in words) / max(1, len(words))


def shift(text: str, k: int) -> str:
    return "".join(chr(ord(c) + k) if not c.isspace() and 32 < ord(c) + k < 0x250 else c for c in text)


def pdf_text(path: str) -> tuple[str, str | None]:
    """-> (text, None) or ('', why)."""
    r = subprocess.run([PDFTOTEXT, "-enc", "UTF-8", path, "-"], capture_output=True, timeout=900)
    text = r.stdout.decode("utf-8", "replace")
    pages = text.count("\f") or 1                                 # pdftotext ends every page with a form feed
    if len(text.replace("\f", "").strip()) / pages < MIN_CHARS_PER_PAGE:
        return "", "scan_needs_ocr"
    text = text.replace("\f", "\n\n")
    if readable(text) < 0.06:                                     # a font whose letters map to the wrong codes
        for k in (29, -29, 3, -3, 1, -1):                         # ('WKH' for 'the': every glyph off by 29)
            if readable(shift(text[:60000], k)) > 0.12:
                text = shift(text, k)
                break
        else:
            return "", "garbled_text_layer"
    # pdftotext breaks every line: join lines inside a paragraph, keep blank-line paragraph breaks
    text = re.sub(r"(?<![\n.!?:;\"”’])\n(?!\n)", " ", text)
    text = re.sub(r"-\s+(?=[a-z])", "", text)              # 'histo- ry' -> 'history' (a word split at a line end)
    return text, None


def convert(path: str) -> tuple[str, str | None]:
    e = path.rsplit(".", 1)[-1].lower()
    try:
        if e == "epub":
            return epub_text(path), None
        if e == "htmlz":
            return htmlz_text(path), None
        if e in ("azw3", "azw", "mobi"):
            try:
                return kindle_text(path), None
            except ImportError:
                return "", "kindle_needs_mobi_package"
        if e == "pdf":
            return pdf_text(path)
    except Exception as ex:                                      # noqa: BLE001 -- a broken file is listed, not fatal
        return "", f"error:{type(ex).__name__}"
    return "", "format"


def archive_members(archive: str) -> list[tuple[str, int]]:
    out = subprocess.run([SEVEN_ZIP, "l", "-ba", "-slt", archive], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300).stdout
    members = []
    for block in out.split("\n\n"):
        m = re.search(r"^Path = (.+)$", block, re.M)
        s = re.search(r"^Size = (\d+)$", block, re.M)
        if m and "Attributes = D" not in block:
            members.append((m.group(1).strip(), int(s.group(1)) if s else 0))
    return members


def inventory(library: str, have_dirs: list[str], archives: bool):
    """-> {title key: (pref, ext, source)}; source is ('file', path) or ('archive', archive path, member path)."""
    have = set()
    for d in have_dirs:
        for root, _, files in os.walk(d):
            have.update(title_key(f) for f in files if f.lower().endswith(".txt"))
    books: dict[str, tuple] = {}

    def offer(name, src):
        e = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if e not in PREF:
            return
        k = title_key(os.path.basename(name))
        if k in have or not k:
            return
        if k not in books or PREF[e] < books[k][0]:
            books[k] = (PREF[e], e, src)

    rars = []
    for root, _, files in os.walk(library):
        for f in files:
            p = os.path.join(root, f)
            if f.lower().endswith(".rar"):
                rars.append(p)
            else:
                offer(f, ("file", p))
    if archives:
        for a in rars:
            for member, _ in archive_members(a):
                offer(member, ("archive", a, member))
    return books, len(have)


def out_path(out: str, library: str, src: tuple) -> pathlib.Path:
    if src[0] == "file":
        rel = os.path.relpath(src[1], library)
    else:
        rel = os.path.join(os.path.splitext(os.path.relpath(src[1], library))[0], src[2])
    return pathlib.Path(out) / (os.path.splitext(rel)[0] + ".txt")


def write_book(args):
    path, dest = args
    text, why = convert(path)
    if why:
        return why, 0
    words = len(text.split())
    if words < 2000:
        return "too_short", words
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return "converted", words


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True)
    ap.add_argument("--have", action="append", default=[], help="a folder of books already text; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True, help="where archive members are unpacked, one archive at a time")
    ap.add_argument("--archives", action="store_true", help="also the books inside the library's .rar archives")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="a trial: at most this many books")
    args = ap.parse_args()
    t0 = time.time()
    books, n_have = inventory(args.library, args.have + [args.out], args.archives)
    items = sorted(books.items(), key=lambda kv: kv[1][2][1])
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} books to convert ({n_have:,} titles already text) ({time.time() - t0:.0f}s)", flush=True)
    stats, words, ocr = collections.Counter(), 0, []
    loose = [(src[1], out_path(args.out, args.library, src)) for _, (_, _, src) in items if src[0] == "file"]
    by_archive = collections.defaultdict(list)
    for _, (_, _, src) in items:
        if src[0] == "archive":
            by_archive[src[1]].append(src[2])

    def account(res, rel):
        nonlocal words
        why, w = res
        stats[why] += 1
        words += w
        if why == "scan_needs_ocr":
            ocr.append(rel)

    with cf.ProcessPoolExecutor(args.workers) as pool:
        todo = [(p, d) for p, d in loose if not d.exists()]
        stats["already_done"] += len(loose) - len(todo)
        for (p, d), res in zip(todo, pool.map(write_book, todo)):
            account(res, os.path.relpath(p, args.library))
            if sum(stats.values()) % 200 == 0:
                print(f"  {dict(stats)}, {words / 1e6:.0f}M words, {time.time() - t0:.0f}s", flush=True)
        for i, (archive, members) in enumerate(by_archive.items()):
            dests = [(m, out_path(args.out, args.library, ("archive", archive, m))) for m in members]
            dests = [(m, d) for m, d in dests if not d.exists()]
            if not dests:
                continue
            tmp = pathlib.Path(args.scratch) / f"a{i:04d}"
            tmp.mkdir(parents=True, exist_ok=True)
            listfile = tmp / "_members.txt"
            listfile.write_text("\n".join(m for m, _ in dests), encoding="utf-8")
            subprocess.run([SEVEN_ZIP, "x", archive, f"-o{tmp}", "-y", f"@{listfile}"], capture_output=True, timeout=7200)
            jobs = [(str(tmp / m), d) for m, d in dests if (tmp / m).exists()]
            stats["archive_member_missing"] += len(dests) - len(jobs)
            for (p, d), res in zip(jobs, pool.map(write_book, jobs)):
                account(res, os.path.relpath(archive, args.library) + " :: " + os.path.relpath(p, tmp))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  archive {i + 1}/{len(by_archive)}: {dict(stats)}, {words / 1e6:.0f}M words, {time.time() - t0:.0f}s", flush=True)
    manifest = {"library": os.path.basename(os.path.normpath(args.library)), "books": len(items), "stats": dict(stats),
                "words": words, "needs_ocr": ocr, "wall_s": round(time.time() - t0)}
    (ROOT / "validation" / "logs" / "convert_books.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("books", "stats", "words", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
