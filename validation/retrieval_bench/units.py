"""units -- quantities compared after unit conversion, for scoring only (order 0047, Cowork's question 2; 0046's harness).

"96 m" and "315 feet" are the same height: a spoken quantity and a gold answer are ``near`` when both read as a number
with a unit of the same dimension (or the gold has no unit) and they agree within ``TOL`` (0.5%, relative). A bare gold
number is compared with the spoken number as stated ("180 cm" against "180.5"). Never used on the serving path.

Wired: STANDALONE (the retrieval bench's scorer).
"""
from __future__ import annotations

import re

__wiring__ = "STANDALONE"

TOL = 0.005
# unit word -> (dimension, factor to the dimension's base unit)
UNITS = {}
for words, dim, f in [
    ("m metre metres meter meters", "length", 1.0), ("cm centimetre centimetres centimeter centimeters", "length", 0.01),
    ("mm millimetre millimetres millimeter millimeters", "length", 0.001), ("km kilometre kilometres kilometer kilometers", "length", 1000.0),
    ("ft foot feet", "length", 0.3048), ("in inch inches", "length", 0.0254), ("mi mile miles", "length", 1609.344),
    ("yd yard yards", "length", 0.9144),
    ("kg kilogram kilograms kilo kilos", "mass", 1.0), ("g gram grams", "mass", 0.001), ("t tonne tonnes", "mass", 1000.0),
    ("lb lbs pound pounds", "mass", 0.45359237), ("oz ounce ounces", "mass", 0.028349523125),
    ("km2 km² sq_km square_kilometre square_kilometres square_kilometer square_kilometers", "area", 1e6),
    ("m2 m² square_metre square_metres square_meter square_meters", "area", 1.0), ("ha hectare hectares", "area", 1e4),
    ("sq_mi square_mile square_miles mi2 mi²", "area", 2589988.110336), ("acre acres", "area", 4046.8564224),
    ("s second seconds", "time", 1.0), ("min minute minutes", "time", 60.0), ("h hour hours", "time", 3600.0),
    ("°c degree_celsius degrees_celsius", "celsius", 1.0),
    ("km/h kmh kilometres_per_hour kilometers_per_hour", "speed", 1 / 3.6), ("mph miles_per_hour", "speed", 0.44704),
]:
    for w in words.split():
        UNITS[w.replace("_", " ")] = (dim, f)
_NUM = re.compile(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(.*?)\s*$")


def read(text: str) -> tuple[float, str | None, float] | None:
    """(the number, its dimension or None, the factor to base) for "96 m", "315 feet", "180.5"; None for anything else."""
    m = _NUM.match(str(text or "").lower().replace("square ", "square_").replace("per hour", "per_hour"))
    if not m:
        return None
    try:
        x = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = m.group(2).strip().rstrip(".").replace("_", " ")
    if not unit:
        return x, None, 1.0
    if unit not in UNITS:
        return None
    dim, f = UNITS[unit]
    return x, dim, f


def near(said: str, gold: str, tol: float = TOL) -> bool:
    a, b = read(said), read(gold)
    if a is None or b is None or (a[1] is None and b[1] is None):
        return False
    if b[1] is None or a[1] is None:                          # a bare gold number: the spoken number as stated
        x, y = a[0], b[0]
    elif a[1] != b[1]:
        return False
    else:
        x, y = a[0] * a[2], b[0] * b[2]
    return y != 0 and abs(x - y) / abs(y) <= tol


__all__ = ["near", "read"]
