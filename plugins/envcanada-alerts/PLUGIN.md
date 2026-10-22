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
  - is there a weather warning for Calgary?
  - are any storm watches out for Nova Scotia?
  - is Environment Canada warning about fog in Lévis?
  - is there a frost advisory tonight in Ontario?
  - any blizzard warnings in Nunavut?
  - should I worry about a weather alert in Gatineau today?
  - is there an extreme cold warning in Edmonton?
  - any special weather statement for Quebec?
  - is a thunderstorm warning in effect near Sherbrooke?
  - are there any active alerts for my area in New Brunswick?
  - is there a heat advisory in Montreal this afternoon?
  - what weather warnings are in effect in Canada right now?
  - is there an air quality warning in Kelowna?
  - y a-t-il un avertissement de tempête à Lévis ?
  - des alertes météo à Québec aujourd'hui ?
  - est-ce qu'il y a un avertissement de neige à Montréal ?
  - y a-t-il une veille d'orage violent en Estrie ?
  - un avertissement de pluie verglaçante est-il en vigueur à Gatineau ?
  - quelles alertes météo sont en vigueur au Québec ?
  - est-ce qu'Environnement Canada a émis un avertissement de vent à Rimouski ?
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
  - what causes a tornado?
  - how much snow did Montreal get in 1998?
  - what is the average temperature in Calgary in July?
  - when was Environment Canada founded?
  - how do I prepare an emergency kit?
  - what is the difference between a watch and a warning?
  - who is the premier of Quebec?
  - how far is Lévis from Quebec City?
  - what is the history of the ice storm in 1998?
  - what time does the sun set in Halifax?
  - is Gatineau in Quebec or Ontario?
  - what are the biggest cities in Canada?
  - what is the climate like in Vancouver?
  - how many provinces does Canada have?
  - quelle est la population de Lévis ?
  - qu'est-ce qui cause la pluie verglaçante ?
  - quand a eu lieu la crise du verglas ?
  - quelle est la capitale de la Nouvelle-Écosse ?
  - comment se préparer à une tempête ?
  - quel temps faisait-il à Québec en 1900 ?
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
