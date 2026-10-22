"""cubbyllm/sources -- where a fact the stores lack comes from: Wikidata, the web, the news (order 0031: moved here from
standin/). The fetching itself leaves for cubby_serve/sources/ in the split commits (Nick, 2026-10-22: cubbyllm/ stays
offline); what stays here takes a fetch it is given.

Wired: WIRED (the ask loop's sources).
"""
__wiring__ = "WIRED"
