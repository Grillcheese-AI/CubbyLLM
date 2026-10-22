"""standin/robots_rules.py -- what a publisher's robots.txt says, read by group (RFC 9309).

- `disallowed_all(txt)`: the agents shut out entirely (`Disallow: /`). Consecutive `User-agent` lines share the rules
  that follow them: the first parser (in `news_source.py`) read one name a block, so a group of thirty-five names over
  one `Disallow: /` gave the rule to the last only, and La Presse, Le Devoir and the Guardian read as naming no AI
  agent (2026-10-22).
- `AI_AGENTS`: the crawler products publishers name when they mean "not for building AI systems".
- `refuses(txt, agent, url)`: does this robots.txt refuse `agent` the path of `url` (our own name, or `*`)? A refusal
  addressed to us is never worked around, and the poller never changes its name (`CubbyLLM/0.1`) to get past one.
- `crawl_delay(txt, agent)`: the seconds it asks between requests (Global Voices: 10); the poller waits that long.

Wired: WIRED (`news_source.NewsSource`, `news_archive`).
"""
from __future__ import annotations

import urllib.parse
import urllib.robotparser

__wiring__ = "WIRED"

AI_AGENTS = frozenset("""gptbot chatgpt-user oai-searchbot perplexitybot ccbot anthropic-ai
    claude-web claudebot cohere-ai deepseek deepseekbot google-extended applebot-extended
    bytespider meta-externalagent amazonbot ai2bot omgili""".split())


def disallowed_all(robots: str) -> set[str]:
    """The agents (lowercase) a robots.txt shuts out entirely, read by group."""
    out, agents, ruled = set(), [], False
    for raw in robots.splitlines():
        line = raw.split("#", 1)[0].strip()
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "user-agent":
            if ruled:                                    # a rule ended the last group: this name starts a new one
                agents, ruled = [], False
            agents.append(val.lower())
        elif key in ("disallow", "allow", "crawl-delay"):
            ruled = True
            if key == "disallow" and val == "/":
                out.update(agents)
    return out


def ai_named(robots: str) -> list[str]:
    """The AI agents a robots.txt shuts out entirely, sorted."""
    return sorted(disallowed_all(robots) & AI_AGENTS)


def crawl_delay(robots: str, agent: str) -> float:
    """The seconds a robots.txt asks ``agent`` (or ``*``) to wait between requests; 0 when it asks none."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots.splitlines())
    d = rp.crawl_delay(agent.split("/", 1)[0])
    return float(d) if d else 0.0


def refuses(robots: str, agent: str, url: str) -> bool:
    """True when the rules refuse `agent` (its product token, e.g. `CubbyLLM`) the path of `url`."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots.splitlines())
    parts = urllib.parse.urlsplit(url)
    return not rp.can_fetch(agent.split("/", 1)[0], parts.path + ("?" + parts.query if parts.query else "") or "/")
