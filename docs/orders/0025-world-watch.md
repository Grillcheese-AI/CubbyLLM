# 0025 — World watch: official warnings, tension signals and their change points
status: proposed
written: 2026-10-22 by Cowork · approved: — · hypothesis: H-K12 (proposed: the world's state from event streams),
with H-F2 (temporal worlds) · **after 0024 (the archive) and 0020 (the change-point toolkit)**

## Why
Nick (2026-10-22): analyse tensions, disaster warnings and patterns in the news, to predict the state of the world
from current and past events, and to see what follows if one branch or one choice becomes reality. That is two
orders. This one builds what is measured: the warnings officials publish, and signals of tension with the dates
they change. Order 0026 builds forecasts and the branching tree on top of it.

Checked from Nick's machine the same day: GDACS (the UN / EU global disaster alerts) answers with 283 current
alerts; USGS's earthquake feeds answer (21 quakes of magnitude 4.5+ in the past day); Environment Canada's alert
datamart answers at `dd.weather.gc.ca/today/alerts/cap/` (its older paths are gone); WHO's news feed answers (109
items). GDELT's 15-minute event files (order 0024) carry, for every event, a CAMEO code, a quad class (verbal or
material, cooperation or conflict), a Goldstein score (-10 to +10, conflict to cooperation) and a tone.

- **What it brings:** Cubby relays the official warnings that concern a place, and says where tension between two
  countries is rising, since when and how sharply, against each pair's own normal.
- **Advantage over conventional models:** a chat model knows of a disaster only if it was in its training data or
  a search finds it. Here warnings come from the agencies that issue them, and "tension is rising" is a measured
  change against a baseline, with its date and its evidence in the audit view.
- **In plain words:** Cubby tells you about official warnings where you care, and notices when relations between
  countries start to sour, before it is a headline.

## Build
- **Warnings** (`standin/warnings_source.py`): GDACS (alerts with their green / orange / red level, the hazard, the
  area), USGS (quakes, GeoJSON), Environment Canada (CAP alerts by region; Lévis first), WHO (news; outbreak notices
  if a feed exists). Each kept as an official statement with its issuer, level, area, time and expiry. Cubby
  relays them dated and attributed in its own words ("an official alert for ... is in effect until ..."), never as
  its own prediction, and never in place of the agency's: the reply points to the official channel.
- **Signals** (`validation/world_signals.py`, then a world): from GDELT's events, per country and per country pair
  and day, the event count by quad class, the mention-weighted mean Goldstein score and the tone; each against its
  own rolling 90-day baseline (a z-score). Daily, stored with the evidence (event ids).
- **Change points:** order 0020's toolkit (CUSUM, BOCPD, PELT) on the signals. A **tension shift** is reported only
  when the change probability, the effect size and the persistence all pass, as in 0020, so one bad day is not a
  shift. It is said as "measured", dated, and the audit view shows the events.
- **Ground truth for later:** UCDP's georeferenced event data (organized violence, monthly candidate releases) through
  its API: a token is needed, requested by email by Nick (`x-ucdp-access-token`, 5,000 requests a day).

## Acceptance (pre-registered)
- **Warnings:** 7 days of polling: per source, alerts kept, 0 duplicates; 20 questions ("any warning for Lévis?",
  "what alerts are out for <country>?"): **0 wrong spoken**, every answer dated and attributed to its issuer, and an
  expired alert never said as current.
- **Signals:** one year of GDELT history (2025) built offline; the signals reproduce known shifts: on 10 dated events
  chosen beforehand by Cowork and Nick from 2025 (an outbreak of fighting, a ceasefire, a diplomatic break), a tension
  shift is detected within 7 days of the date on >= 7. On 10 quiet country pairs chosen the same way, false shifts
  are counted and reported.
- **Replies:** no tension shift said without its date, its measure and "measured from news reports"; 0 named outlets.
- Off = unchanged; full `pytest tests` and `pytest standin/tests`.

## Constraints
- Rules in `CLAUDE.md` hold (flag-gated, < 300 lines a file, `__wiring__`, tests pin the claim).
- No external model. A machine-coded event is a reported claim; a signal is a measure of reporting, not of the world.
- **ACLED is not used:** its terms forbid substitute or competing products, and need a corporate licence. ReliefWeb's
  API now refuses requests without an approved app name (HTTP 403): only if Nick asks for access.

## Out of scope
- Forecasts, scenarios, the branching tree: order 0026.
- Warnings issued by Cubby itself.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
