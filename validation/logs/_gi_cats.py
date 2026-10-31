import urllib.request, re, html, gzip
UA = {"User-Agent": "CubbyLLM/0.1"}
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    return b.decode("utf-8", "replace")
home = get("https://www.globalissues.org/")
issues = []
for h, txt in re.findall(r'<a[^>]+href="([^"]*/issue/[^"]+)"[^>]*>(.*?)</a>', home, re.S):
    u = h if h.startswith("http") else "https://www.globalissues.org" + h
    name = re.sub(r"<[^>]+>", "", html.unescape(txt)).strip()
    if name and u not in [x[0] for x in issues]:
        issues.append((u, name))
print("issue links on the home page:", len(issues))
for u, n in issues[:5]: print("  ", n, "|", u)
# alternate feed links on the first issue page
if issues:
    page = get(issues[0][0])
    alts = re.findall(r'<link[^>]+rel="alternate"[^>]*>', page)
    print("alternate links on", issues[0][1], ":", alts[:5])
    feeds = sorted(set(re.findall(r'href="([^"]*(?:feed|rss|atom)[^"]*)"', page)))
    print("feed-like hrefs:", feeds[:12])
