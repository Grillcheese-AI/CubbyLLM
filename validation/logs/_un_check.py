import urllib.request, gzip, re
UA = {"User-Agent": "CubbyLLM/0.1", "Accept-Encoding": "identity"}
for u in ["https://news.un.org/feed/subscribe/en/news/all/rss.xml", "https://news.un.org/feed/subscribe/fr/news/all/rss.xml"]:
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read()
    enc = r.headers.get("Content-Encoding")
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b); enc = (enc or "") + " (gzip bytes)"
    print(u, "| status", r.status, "| type", r.headers.get("Content-Type"), "| encoding", enc, "| bytes", len(b))
    print("   first bytes:", b[:80])
    t = b.decode("utf-8", "replace")
    print("   <item> count:", len(re.findall(r"<item[\s>]", t)), "| <entry> count:", len(re.findall(r"<entry[\s>]", t)))
    m = re.search(r"<pubDate>([^<]+)</pubDate>", t); print("   first pubDate:", m.group(1) if m else "-")
