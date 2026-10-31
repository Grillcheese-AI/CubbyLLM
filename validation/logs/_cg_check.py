import urllib.request, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
for name, u in [("crisiswatch", "https://www.crisisgroup.org/rss/crisiswatch"), ("global-139", "https://www.crisisgroup.org/rss/139"),
                ("africa-1", "https://www.crisisgroup.org/rss/1"), ("mena-81", "https://www.crisisgroup.org/rss/81"),
                ("ukraine-72", "https://www.crisisgroup.org/rss/72")]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
        root = ET.fromstring(r.read()); items = root.findall("./channel/item")
        links = [i.findtext("link") for i in items]
        print(f"{name:12s} {r.status} items={len(items):3d} newest={items[0].findtext('pubDate') if items else '-'}")
    except Exception as e:
        print(f"{name:12s} FAIL {type(e).__name__} {str(e)[:80]}")
