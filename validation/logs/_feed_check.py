import sys, urllib.request, urllib.robotparser as rp, xml.etree.ElementTree as ET
from urllib.parse import urlsplit
UA = {"User-Agent": "CubbyLLM/0.1"}
AI = {"anthropic-ai", "claudebot", "claude-web", "claude", "gptbot", "ccbot", "google-extended", "chatgpt-user",
      "perplexitybot", "perplexity-user", "cohere-ai", "bytespider", "oai-searchbot"}
for u in sys.argv[1:]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
        root = ET.fromstring(r.read()); items = root.findall("./channel/item")
        print(f"FEED {r.status} items={len(items)} newest={items[0].findtext('pubDate') if items else '-'} title={root.findtext('./channel/title')} | {u}")
    except Exception as e:
        print(f"FEED FAIL {type(e).__name__} {str(e)[:80]} | {u}")
    site = "{0.scheme}://{0.netloc}".format(urlsplit(u))
    try:
        t = urllib.request.urlopen(urllib.request.Request(site + "/robots.txt", headers=UA), timeout=20).read().decode("utf-8", "replace")
        groups, cur = [], None
        for raw in t.splitlines():
            l = raw.split("#")[0].strip()
            if not l: continue
            k, _, v = l.partition(":"); k = k.strip().lower(); v = v.strip()
            if k == "user-agent":
                if cur is None or cur["rules"]: cur = {"agents": [], "rules": []}; groups.append(cur)
                cur["agents"].append(v)
            elif k in ("allow", "disallow") and cur is not None: cur["rules"].append((k, v))
        blocked = sorted({a for g in groups if ("disallow", "/") in g["rules"] for a in g["agents"] if a.lower() in AI})
        p = rp.RobotFileParser(); p.parse(t.splitlines())
        print(f"     robots: AI agents blocked: {blocked or 'none'}; CubbyLLM may fetch: {p.can_fetch('CubbyLLM/0.1', u)}")
    except Exception as e:
        print(f"     robots: {type(e).__name__}")
