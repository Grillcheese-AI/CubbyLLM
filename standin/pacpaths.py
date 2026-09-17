"""pacpaths - where the cubbyverse assets live. One definition, several readers.

Wired: WIRED (the pac world and its frontend).

Split out of `pacman.py` so the roots are declared ONCE. Two modules need them
for unrelated reasons - `feeling` loads plutchik.json, `pacui_threejs` loads the
page, the faces, the music - and a constant that lives in two places is how
this project already shipped a monoamine table that was wrong in both copies
identically. Nothing here decides anything; it is six paths.
"""
from __future__ import annotations

import os
import pathlib
__wiring__ = "WIRED"


_CV = pathlib.Path(r"C:\Users\grill\Documents\GitHub\cubbyverse")


PACMAN_3D = _CV / "examples" / "pacman_3d.py"


PACMAN_LIVE = _CV / "examples" / "pacman_live.py"        # the exact frontend is lifted from here


ASSETS_DIR = _CV / "examples" / "assets"                 # cubby's face textures


PLUTCHIK_JSON = _CV / "cubbyverse" / "core" / "emotion" / "plutchik.json"


REPLAY_OUT = pathlib.Path(r"C:/tmp/cubbyman_pacman_3d.html")



MUSIC_PATH = pathlib.Path(os.environ.get("CB_PAC_MUSIC") or
                          (pathlib.Path(__file__).resolve().parent / "data" / "music" / "neon_pixel_dash.mp3"))
