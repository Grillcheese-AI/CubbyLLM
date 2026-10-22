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
  - what's 10 km in miles?
  - how many metres are in a mile?
  - convert 2 pounds to grams
  - how much is a litre in gallons?
  - my oven says 350 F, what is that in celsius?
  - how many minutes are in 2 days?
  - how tall is 6 feet in centimetres?
  - how many ounces in a kilogram?
  - 120 km/h is how many mph?
  - how many yards is 100 metres?
  - what is 0 kelvin in celsius?
  - how many hours is 10000 seconds?
  - a recipe wants 250 ml, how many litres is that?
  - is 5 kilograms more than 10 pounds?
  - change 1 tonne into pounds
  - combien de kilomètres font 10 miles ?
  - convertis 30 degrés celsius en fahrenheit
  - combien de livres pèse 5 kilos ?
  - combien de pieds dans 2 mètres ?
  - 100 km/h, ça fait combien en miles à l'heure ?
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
  - how far is Paris from London?
  - what is the boiling point of water?
  - how much does an elephant weigh?
  - what time is it in Tokyo?
  - how long is the Nile river?
  - how fast can a cheetah run?
  - what is the coldest temperature ever recorded?
  - who invented the metric system?
  - why does the US use miles?
  - what will the temperature be tomorrow in Montreal?
  - how many calories are in a banana?
  - how old is the universe?
  - how many countries use fahrenheit?
  - what is a kilowatt-hour?
  - quelle est la capitale du Canada ?
  - quelle distance entre Montréal et Québec ?
  - combien d'habitants à Paris ?
  - qui a inventé le système métrique ?
  - quelle température fera-t-il demain ?
  - combien pèse la Lune ?
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
