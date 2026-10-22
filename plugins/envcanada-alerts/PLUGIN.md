---
name: envcanada-alerts
version: 0.1.0
description: Weather warnings, watches and alerts in effect now for places in Canada, live from the national weather service.
examples:
  - is there a storm warning in Lévis?
  - any weather alerts in Québec today?
  - are there weather warnings for Montreal right now?
  - is a snowfall warning in effect for Ottawa?
  - is there a heat warning in Toronto?
  - any freezing rain alerts near Quebec City?
  - is a wind warning in effect in Halifax?
  - are there active weather watches in Winnipeg?
  - is there a tornado warning in Saskatchewan?
  - any rainfall warnings for Vancouver this week?
not:
  - what is the capital of Canada?
  - how many kilometres is 5 miles?
  - who wrote Romeo and Juliet?
  - what was the weather like in 1900?
  - when did Quebec join Canada?
  - what is the population of Toronto?
  - who is the mayor of Montreal?
  - what happened in Ottawa yesterday?
  - is Lévis a city in Quebec?
  - how far is Halifax from Toronto?
kinds: [source, panel]
audience: public
licence: OGL-Canada
requires: {plugins: [], base: null}
permissions: {network: [dd.weather.gc.ca], write: true, gpu: false}
entry: {source: "source.py:Alerts", panel: panel.json}
---

# envcanada-alerts

The weather warnings, watches and statements in effect in Canada, from the national weather service's Datamart (its
CAP alerts, under the Open Government Licence - Canada). It reads only `dd.weather.gc.ca`, through `ctx.fetch` (robots
read, `CubbyLLM/0.1`), at most every half hour over the last twelve hours of files, and keeps each alert file once in
its own folder.

- `covers(question)`: 0.8 when the question is about weather alerts (warning, watch, storm, snowfall, avertissement,
  verglas, ...), else 0.
- `answer(question)`: for each place the question names that has an alert in effect, "A yellow warning - wind is in
  effect for Avalon Peninsula Southeast until 2026-10-23 05:25 UTC." Nothing for a place it cannot find: an empty answer
  is a refusal, never "no warning".
- The panel: the alerts in effect, counted by headline, and the first ten.
