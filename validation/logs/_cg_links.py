import urllib.request, re, html, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
r = get("https://www.crisisgroup.org/rss-0")
t = r.read().decode("utf-8", "replace")
print("page status", r.status, "chars", len(t))
links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', t, re.S)
seen, out = set(), []
for h, txt in links:
    hl = h.lower()
    if "rss" in hl or "feed" in hl or hl.endswith(".xml"):
        u = h if h.startswith("http") else "https://www.crisisgroup.org" + h
        label = re.sub(r"<[^>]+>", "", html.unescape(txt)).strip()
        if u not in seen and not u.rstrip("/").endswith("/rss-0"):
            seen.add(u); out.append((u, label))
print("candidate links", len(out))
for u, label in out:
    try:
        x = get(u).read()
        root = ET.fromstring(x)
        items = root.findall("./channel/item") or root.findall("{http://www.w3.org/2005/Atom}entry")
        newest = items[0].findtext("pubDate") if items else "-"
        print(f"FEED  {len(items):3d} items  newest {newest}  | {label[:50]} | {u}")
    except Exception as e:
        print(f"NOT A FEED ({type(e).__name__})  | {label[:50]} | {u}")
