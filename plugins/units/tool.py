"""units -- a plugin tool (order 0028): a quantity from one unit to another. No network, nothing written."""

# each unit: (kind, factor to the kind's base unit); temperatures are handled by their offsets
UNITS = {
    "m": ("length", 1.0), "metre": ("length", 1.0), "meter": ("length", 1.0), "metres": ("length", 1.0),
    "km": ("length", 1000.0), "kilometre": ("length", 1000.0), "kilometres": ("length", 1000.0),
    "cm": ("length", 0.01), "centimetre": ("length", 0.01), "centimetres": ("length", 0.01),
    "mm": ("length", 0.001), "in": ("length", 0.0254), "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    "ft": ("length", 0.3048), "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "yd": ("length", 0.9144), "yard": ("length", 0.9144), "mi": ("length", 1609.344), "mile": ("length", 1609.344),
    "miles": ("length", 1609.344),
    "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "kilograms": ("mass", 1.0), "g": ("mass", 0.001),
    "gram": ("mass", 0.001), "grams": ("mass", 0.001), "lb": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237), "oz": ("mass", 0.028349523125), "ounce": ("mass", 0.028349523125),
    "ounces": ("mass", 0.028349523125), "t": ("mass", 1000.0), "tonne": ("mass", 1000.0),
    "s": ("time", 1.0), "second": ("time", 1.0), "seconds": ("time", 1.0), "min": ("time", 60.0),
    "minute": ("time", 60.0), "minutes": ("time", 60.0), "h": ("time", 3600.0), "hour": ("time", 3600.0),
    "hours": ("time", 3600.0), "day": ("time", 86400.0), "days": ("time", 86400.0),
    "l": ("volume", 1.0), "litre": ("volume", 1.0), "liter": ("volume", 1.0), "litres": ("volume", 1.0),
    "ml": ("volume", 0.001), "gal": ("volume", 3.785411784), "gallon": ("volume", 3.785411784),
    "gallons": ("volume", 3.785411784),
    "m/s": ("speed", 1.0), "km/h": ("speed", 1 / 3.6), "kph": ("speed", 1 / 3.6), "mph": ("speed", 0.44704),
}
TEMPS = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}


def _to_kelvin(v, u):
    return {"c": v + 273.15, "f": (v - 32) * 5 / 9 + 273.15, "k": v}[u]


def _from_kelvin(v, u):
    return {"c": v - 273.15, "f": (v - 273.15) * 9 / 5 + 32, "k": v}[u]


class Units:
    schema = {"type": "object", "properties": {"value": {"type": "number"}, "from": {"type": "string"},
                                               "to": {"type": "string"}}, "required": ["value", "from", "to"]}

    def __init__(self, ctx) -> None:
        self.ctx = ctx

    def run(self, args: dict) -> dict:
        v, a, b = float(args["value"]), str(args["from"]).strip().lower(), str(args["to"]).strip().lower()
        if a in TEMPS and b in TEMPS:
            return {"value": round(_from_kelvin(_to_kelvin(v, TEMPS[a]), TEMPS[b]), 6), "unit": b}
        if a not in UNITS or b not in UNITS:
            return {"error": f"unknown unit: {a if a not in UNITS else b}"}
        (ka, fa), (kb, fb) = UNITS[a], UNITS[b]
        if ka != kb:
            return {"error": f"{a} is a {ka}, {b} is a {kb}"}
        return {"value": round(v * fa / fb, 6), "unit": b}
