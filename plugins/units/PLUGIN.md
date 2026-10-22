---
name: units
version: 0.1.0
description: Converts a quantity from one unit to another, for length and distance, mass and weight, temperature, time, volume and speed.
examples:
  - how many kilometres is 5 miles?
  - convert 100 fahrenheit to celsius
  - what is 3 feet in metres?
  - how many pounds are in 2 kilograms?
  - convert 60 miles per hour to km/h
  - how many litres is a gallon?
  - what is 20 degrees celsius in fahrenheit?
  - how many seconds are in 3 hours?
  - convert 500 grams to ounces
  - how many inches is 30 centimetres?
not:
  - what is the capital of Canada?
  - who wrote Hamlet?
  - how many people live in Montreal?
  - when was the Eiffel Tower built?
  - is there a storm warning in Lévis?
  - how tall is Mount Everest?
  - who is the prime minister of Canada?
  - what is the speed of light?
  - how many moons does Jupiter have?
  - what happened in Ottawa yesterday?
kinds: [tool]
audience: public
licence: MIT
requires: {plugins: [], base: null}
permissions: {network: [], write: false, gpu: false}
entry: {tool: "tool.py:Units"}
---

# units

The smallest code plugin: a unit conversion tool with no network and nothing written. `run({"value": 5, "from":
"mi", "to": "km"})` returns `{"value": 8.04672, "unit": "km"}`, or `{"error": ...}` for units of different kinds or
units it does not know. Exact factors (SI definitions), temperatures by their offsets.
