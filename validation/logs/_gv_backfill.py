"""Global Voices history backfill (order 0024, Nick's yes 2026-10-22).
Saves each raw feed page (?paged=N) gzipped, once; resumable (existing pages are skipped).
Honours the site's robots Crawl-delay (10 s) and Retry-After. Identifies as CubbyLLM/0.1.
usage: python _gv_backfill.py <feed url> <out dir> [delay s] [max page]"""
import sys, os, time, gzip, datetime, urllib.request, urllib.error, xml.etree.ElementTree as ET
base, out = sys.argv[1], sys.argv[2]
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
maxp = int(sys.argv[4]) if len(sys.argv) > 4 else 10**6
UA = {"User-Agent": "CubbyLLM/0.1", "Accept-Encoding": "identity"}
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "_progress.log"), "a", encoding="utf-8")
def say(m):
    log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {m}\n"); log.flush()
say(f"start {base} delay={delay}")
n = 1
while n <= maxp:
    fn = os.path.join(out, f"page_{n:05d}.xml.gz")
    if os.path.exists(fn):
        n += 1; continue
    tries, items, body = 0, None, None
    while True:
        try:
            r = urllib.request.urlopen(urllib.request.Request(f"{base}?paged={n}", headers=UA), timeout=60)
            body = r.read()
            if body[:2] == b"\x1f\x8b": body = gzip.decompress(body)
            items = ET.fromstring(body).findall("./channel/item")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404: items = []; break
            tries += 1
            ra = e.headers.get("Retry-After")
            wait = int(ra) if ra and ra.isdigit() else min(3600, 60 * 2 ** tries)
            say(f"page {n} HTTP {e.code}; retry {tries} in {wait}s")
            if tries > 8: say("giving up"); sys.exit(1)
            time.sleep(wait)
        except Exception as e:
            tries += 1
            say(f"page {n} {type(e).__name__} {str(e)[:80]}; retry {tries}")
            if tries > 8: say("giving up"); sys.exit(1)
            time.sleep(60 * tries)
    if not items:
        say(f"end at page {n} (no items)"); break
    with gzip.open(fn + ".tmp", "wb") as f: f.write(body)
    os.replace(fn + ".tmp", fn)
    if n % 50 == 0 or n <= 2: say(f"page {n} ok ({len(items)} items, oldest {items[-1].findtext('pubDate')})")
    n += 1
    time.sleep(delay)
say("stopped" if n > maxp else "done")
