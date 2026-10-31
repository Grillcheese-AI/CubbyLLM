import sys, urllib.request, gzip, xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
for u in sys.argv[1:]:
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "CubbyLLM/0.1"}), timeout=30)
    b = r.read(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    items = ET.fromstring(b).findall("./channel/item")
    ds = []
    for i in items:
        try: ds.append(parsedate_to_datetime(i.findtext("pubDate")))
        except Exception: pass
    ds.sort()
    print(u, "| items", len(items), "| dated", len(ds), "| oldest", ds[0] if ds else "-", "| newest", ds[-1] if ds else "-", "| sorted newest-first:", ds and parsedate_to_datetime(items[0].findtext("pubDate")) == ds[-1])
