"""cubbyllm/sources/news/policy.py -- serve versus train: the AI crawlers a publisher's front door disallows, the
hosts read, a feed's licence, what may be trained on. Split out of source.py (order 0031) as a mixin of NewsSource,
verbatim.

Wired: WIRED (NewsSource).
"""
from __future__ import annotations

import urllib.parse
from cubbyllm.reasoning.planner import Triple, normalize as _normalize   # noqa: E402
from ..robots import AI_AGENTS, ai_named  # noqa: E402

__wiring__ = "WIRED"


class NewsPolicy:
    """NewsSource's serve-versus-train policy (a mixin; order 0031)."""

    # -- what the publisher says about this use ------------------------------------------
    def ai_optout(self, feed_id: str) -> list[str] | None:
        """The AI crawler products this feed's host tells to go away entirely, read from its
        own robots.txt -- or None when there is no robots.txt to read. An empty list means the
        host names none of them.

        This is NOT a permission check for us: none of those directives is addressed to
        CubbyLLM, and every host measured allows the feed path under `User-agent: *`. It is the
        publisher's stated position on having their words used to build an AI, recorded next to
        the words so the choice is visible per fact instead of buried in a config. Measured
        2026-09-14: BBC, NYT, CBC, The Verge, TechCrunch and HackerNoon all name Anthropic's
        crawler explicitly; Global News, WIRED, arXiv and Reddit name none."""
        host = urllib.parse.urlsplit(self.feeds.get(feed_id, "")).netloc
        if not host:
            return None
        # The FEED host understates the publisher. feeds.bbci.co.uk names no AI crawler while
        # www.bbc.co.uk names fifteen; rss.nytimes.com serves no robots.txt at all while
        # www.nytimes.com names fourteen. A feed subdomain is delivery infrastructure -- the
        # position is stated at the front door, and reading only the side door is a way of not
        # hearing it (measured 2026-09-14).
        saw_policy = False
        for h in self._hosts(host):
            names = self._robots_for(h)
            if names:
                return names                                 # the first host that names any of them
            saw_policy = saw_policy or names is not None
        # [] and None are DIFFERENT answers and collapsing them would be the lie: [] is "they
        # published a policy and it names none of these", None is "there was nothing to read".
        return [] if saw_policy else None

    @classmethod
    def _hosts(cls, host: str) -> list[str]:
        """The feed host, then the publisher's front door: `rss.nytimes.com` -> also
        `www.nytimes.com`. Two-label TLDs (co.uk, com.au) keep three labels."""
        out = [host]
        parts = host.split(".")
        for suffix, door in cls.FRONT_DOOR.items():
            if host == suffix or host.endswith("." + suffix):
                out.append(door)
                return out
        if len(parts) < 2:                                   # a bare hostname has no front door
            return out
        two = parts[-2] in ("co", "com", "net", "org", "gov", "ac") and len(parts[-1]) == 2
        keep = 3 if two else 2
        if len(parts) > keep:
            out.append("www." + ".".join(parts[-keep:]))
        return out

    def _robots_for(self, host: str) -> list[str] | None:
        if host in self._robots:
            return self._robots[host]
        try:
            txt = self.fetch(f"https://{host}/robots.txt").decode("utf-8", "replace")
        except Exception:                                    # noqa: BLE001 -- no robots.txt is not a yes and not a no
            self._robots[host] = None
            return None
        self._robots[host] = ai_named(txt)
        return self._robots[host]

    def policy(self) -> dict[str, list[str] | None]:
        """`{feed_id: [the AI crawlers its host disallows]}` for every configured feed."""
        return {fid: self.ai_optout(fid) for fid in self.feeds}

    def licence_of(self, feed_id: str) -> str:
        """`OPEN` when the publisher names no AI crawler, `SERVE_ONLY` when they name any,
        `UNKNOWN` when there is no robots.txt to read. UNKNOWN is treated as SERVE_ONLY by
        `trainable` -- silence is not consent, and the cost of being wrong is asymmetric."""
        names = self.ai_optout(feed_id)
        return self.UNKNOWN if names is None else (self.SERVE_ONLY if names else self.OPEN)

    def trainable(self, feed_id: str) -> bool:
        """May this feed's headlines enter a dataset that becomes model weights?"""
        return self.licence_of(feed_id) == self.OPEN

    def training_facts(self, entity: str) -> list[Triple]:
        """`facts()`, minus every publisher who asked not to be used to build an AI. This is
        the ONLY entry point a dataset builder may call; `facts()` is for serving."""
        keep = {f for f in self.feeds if self.trainable(f)}
        out = []
        for t in self.facts(entity):
            fact = _normalize(f"{t.obj} is the {t.rel} of {t.subj}")
            if self.provenance.get(fact) in keep:
                out.append(t)
        return out
