"""Render one journey through Cubby's brain as a video: the same zones, arcs and cards as the live
page (dashboard/brain3d.html, at /brain), from a journey that page saved (its Save button).

    blender -b --factory-startup -P tools/brain_render.py -- JOURNEY.json [--out DIR] [--fps 30] [--preview]

The page is the live instrument; this is the film of one of its journeys, at 1920x1080: the brain's
point-cloud shell, each zone a glowing core, the signal riding an arc zone to zone, the zone flaring
in the step's tone (amber running, teal certified, red refused, violet a question framed) and a card
saying what happened. Steps keep their order and at least STEP seconds apart so each can be read.
Writes <out>/frames and <out>/<journey>.mp4 (ffmpeg).
"""

import pathlib as _pathlib  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))      # the repo root, for:
from cubbyllm.local_paths import expand as _lp  # noqa: E402 -- local folders by name (order 0030)
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import bpy
from mathutils import Vector

TONE = {"hot": "#ffb648", "yes": "#45d3c0", "no": "#ff6b6b", "wonder": "#a594ff"}
BG, STEP, FLY, HOLD = "#070b16", 0.75, 0.56, 3.4         # seconds: between steps, a flight, a card on screen
FONT = Path(_lp(r"$WINDIR\Fonts\segoeui.ttf"))
FONT_B = Path(_lp(r"$WINDIR\Fonts\segoeuib.ttf"))


def arg(name, default=None):
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else (name in a) if default is None else default


def B(p):                                                   # the page's y-up space -> Blender's z-up
    return Vector((p[0], -p[2], p[1]))


def rgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def mulberry(a):                                            # the page's seeded generator, bit for bit
    s = [a & 0xFFFFFFFF]

    def r():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = (((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF)
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


# ---------------------------------------------------------------------------- materials
def glow_mat(name, color, strength, halo=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out, em = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeEmission")
    em.name = "em"
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    if halo:                                                # bright in the middle, gone at the rim
        lw, inv, pw = nt.nodes.new("ShaderNodeLayerWeight"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
        inv.operation, pw.operation = "SUBTRACT", "POWER"
        inv.inputs[0].default_value, pw.inputs[1].default_value = 1.0, 4.0
        nt.links.new(lw.outputs["Facing"], inv.inputs[1])
        nt.links.new(inv.outputs[0], pw.inputs[0])
        tr, mix = nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
        soft = nt.nodes.new("ShaderNodeMath")
        soft.operation, soft.inputs[1].default_value = "MULTIPLY", 0.55
        nt.links.new(pw.outputs[0], soft.inputs[0])
        nt.links.new(soft.outputs[0], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])         # (dithered: depth-tested like the rest)
    else:
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def sphere(name, at, r, mat, segs=24):
    me = bpy.data.meshes.new(name)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=segs // 2, radius=r)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.location = at
    return o


# ---------------------------------------------------------------------------- the brain
def shell():
    """The page's point cloud: two folded hemispheres, a cerebellum and a stem."""
    rnd, pts, cols = mulberry(7), [], []

    def put(x, y, z, light):
        pts.append(B((x, y, z)))
        cols.append((0.25 * light, 0.32 * light, 0.95 * light, 1.0))
    for i in range(16000):
        u, a = rnd() * 2 - 1, rnd() * math.pi * 2
        s = math.sqrt(1 - u * u)
        dx, dy, dz = s * math.cos(a), u, s * math.sin(a)
        side = 1 if i % 2 else -1
        if dz * side < -0.15:
            dz = -dz
        fold = 1 + 0.05 * math.sin(9 * dx + 4 * dy) * math.cos(7 * dy - 3 * dz) + 0.025 * math.sin(23 * dx * dy)
        x, y, z = dx * 5.3 * fold, dy * 3.5 * fold + 1.2, dz * 3.2 * fold + side * 0.25
        if y < -1.1:
            y = -1.1 + (y + 1.1) * 0.25
        rnd()
        put(x, y, z, 0.42 + 0.25 * rnd())
    for i in range(2600):
        u, a = rnd() * 2 - 1, rnd() * math.pi * 2
        s, f = math.sqrt(1 - u * u), 1 + 0.06 * math.sin(30 * u)
        put(3.6 + s * math.cos(a) * 1.8 * f, -1.8 + u * 1.0 * f, s * math.sin(a) * 2.4 * f, 0.45 + 0.2 * rnd())
    for i in range(900):
        h, a = rnd(), rnd() * math.pi * 2
        r = 0.55 - 0.15 * h
        put(1.8 - h * 0.6 + math.cos(a) * r, -1.4 - h * 2.6, math.sin(a) * r, 0.4 + 0.2 * rnd())
    me = bpy.data.meshes.new("shell")
    me.from_pydata(pts, [], [])
    att = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    att.data.foreach_set("color", [c for col in cols for c in col])
    o = bpy.data.objects.new("Shell", me)
    bpy.context.scene.collection.objects.link(o)
    ng = bpy.data.node_groups.new("dots", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    mp, sm = ng.nodes.new("GeometryNodeMeshToPoints"), ng.nodes.new("GeometryNodeSetMaterial")
    mp.inputs["Radius"].default_value = 0.018
    m = glow_mat("dots", (1, 1, 1), 0.8)
    at = m.node_tree.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "col"
    m.node_tree.links.new(at.outputs["Color"], m.node_tree.nodes["em"].inputs["Color"])
    sm.inputs["Material"].default_value = m
    ng.links.new(gi.outputs[0], mp.inputs["Mesh"])
    ng.links.new(mp.outputs[0], sm.inputs["Geometry"])
    ng.links.new(sm.outputs[0], go.inputs[0])
    o.modifiers.new("dots", "NODES").node_group = ng


def arc(A, Bp):
    """The page's arc between two zones (page space), as a function of t."""
    A, Bp = Vector(A), Vector(Bp)
    mid = (A + Bp) * 0.5
    out = Vector((mid.x, 0.0, mid.z))
    out = out.normalized() if out.length > 1e-6 else Vector((1, 0, 0))
    ctrl = mid + out * 1.1 + Vector((0, 1.1, 0))
    return lambda t: B((1 - t) ** 2 * A + 2 * (1 - t) * t * ctrl + t * t * Bp)


def line(name, f, mat, width=0.01, n=40):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions, cu.bevel_depth = "3D", width
    sp = cu.splines.new("POLY")
    sp.points.add(n)
    for i, p in enumerate(sp.points):
        p.co = (*f(i / n), 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)


# ---------------------------------------------------------------------------- words on screen
def text(name, body, size, color, at, font, parent=None, align="LEFT"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body, cu.size, cu.align_x, cu.align_y = body, size, align, "TOP"
    if font.is_file():
        cu.font = bpy.data.fonts.load(str(font), check_existing=True)
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    cu.materials.append(glow_mat(name, color, 1.0))
    o.location, o.parent = at, parent
    return o


def keys(o, path, frames_values, interp="LINEAR"):
    for f, v in frames_values:
        setattr(o, path, v)
        o.keyframe_insert(path, frame=f)
    if o.animation_data and o.animation_data.action:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(o.animation_data.action, o.animation_data.action_slot)
        for fc in bag.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = interp


# ---------------------------------------------------------------------------- the film
def main():
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = Path(arg("--out", str(path.with_suffix("")))).resolve()
    fps = int(arg("--fps", "30"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine, sc.render.fps = "BLENDER_EEVEE", fps
    sc.eevee.taa_render_samples = 32
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.view_settings.view_transform = "Standard"                 # the page's sRGB colours, as they are
    sc.world = bpy.data.worlds.new("bg")
    sc.world.use_nodes = True
    next(n for n in sc.world.node_tree.nodes if n.type == "BACKGROUND").inputs[0].default_value = (*rgb(BG), 1.0)
    zones = {z["id"]: z for z in doc["zones"]}
    shell()
    dim = glow_mat("tract", rgb("#5a6db0"), 0.35)
    for a, b in doc.get("tracts", []):
        line(f"tract {a}-{b}", arc(zones[a]["p"], zones[b]["p"]), dim, 0.008)
    # when each step lands: in order, never closer than STEP, the first after a breath
    steps, t = doc["steps"], 1.2
    for i, s in enumerate(steps):
        t = max(t, 1.2 + s["at"] / 1000 + FLY) if i == 0 else max(t + STEP, 1.2 + s["at"] / 1000 + FLY)
        s["land"] = t
    end = steps[-1]["land"] + HOLD + 0.8
    F = lambda sec: int(round(sec * fps))
    sc.frame_start, sc.frame_end = 0, F(end)
    # zones: a core and a halo that flare on arrival, tinted by the step's tone
    for z in zones.values():
        z["at"], z["col"] = B(z["p"]), rgb(z["c"])
        z["core"] = sphere("core " + z["id"], z["at"], 0.15, glow_mat("core " + z["id"], z["col"], 3.0))
        z["halo"] = sphere("halo " + z["id"], z["at"], 0.5, glow_mat("halo " + z["id"], z["col"], 1.0, halo=True))
        z["hits"] = [(s["land"], s["tone"]) for s in steps if s["zone"] == z["id"]]
    for z in zones.values():                                     # flare levels, frame by frame (0.55 a second)
        em = z["halo"].active_material.node_tree.nodes["em"]
        for f in range(0, F(end) + 1, 2):
            sec, level, tone = f / fps, 0.0, None
            for land, tn in z["hits"]:
                if land <= sec:
                    level, tone = max(0.0, 1 - (sec - land) * 0.55), tn
            col = Vector(z["col"]).lerp(Vector(rgb(TONE[tone])), 0.65 * level) if tone else Vector(z["col"])
            em.inputs["Strength"].default_value = 0.8 + 6.0 * level
            em.inputs["Color"].default_value = (*col, 1.0)
            em.inputs["Strength"].keyframe_insert("default_value", frame=f)
            em.inputs["Color"].keyframe_insert("default_value", frame=f)
            z["halo"].scale = (1 + 1.6 * level,) * 3
            z["halo"].keyframe_insert("scale", frame=f)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    cam.data.lens = 31                                           # the page's 36-degree vertical view
    sc.camera = cam
    for z in zones.values():                                     # their names, under them, facing the camera
        tag = text("name " + z["id"], z["name"].upper(), 0.27, rgb("#aab6cf"), z["at"] + Vector((0, 0, -0.32)), FONT_B,
                   align="CENTER")
        tag.constraints.new("COPY_ROTATION").target = cam
    # the signal: a bright head and a fading tail along the arc, landing on the step's time
    for i, s in enumerate(steps):
        if not s.get("from") or s["from"] == s["zone"] or s["from"] not in zones:
            continue
        f_arc, col = arc(zones[s["from"]]["p"], zones[s["zone"]]["p"]), rgb(TONE.get(s["tone"], TONE["hot"]))
        for k in range(6):
            o = sphere(f"pulse {i}.{k}", (0, 0, 0), 0.11 - 0.014 * k, glow_mat(f"pulse {i}.{k}", col, 14 - 2 * k), 12)
            frames = []
            for f in range(F(s["land"] - FLY) - 1, F(s["land"] + 0.3) + 1):
                u = (f / fps - (s["land"] - FLY)) / FLY - k * 0.05
                vis = 0 <= u <= 1 and f / fps <= s["land"] + 0.25
                tt = min(1, max(0, u))
                frames.append((f, f_arc(1 - (1 - tt) ** 2), 1.0 if vis else 0.0))
            for f, p, v in frames:
                o.location, o.scale = p, (v, v, v)
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("scale", frame=f)
    # the camera: the page's slow orbit, leaning toward whatever just fired
    focus, yaw = Vector((0, 0, 1.0)), -0.5
    for f in range(0, F(end) + 1):
        sec = f / fps
        live = [s for s in steps if s["land"] <= sec < s["land"] + 6]
        want = B(Vector(zones[live[-1]["zone"]]["p"]) * 0.3 + Vector((0, 0.8, 0))) if live else Vector((0, 0, 1.0))
        focus = focus.lerp(want, 1 - math.exp(-1.5 / fps))
        yaw += 0.045 / fps
        d, pitch = 21.0, 0.22
        p3 = Vector((d * math.cos(pitch) * math.sin(yaw), d * math.sin(pitch), d * math.cos(pitch) * math.cos(yaw)))
        cam.location = focus + B(p3)
        cam.rotation_euler = (focus - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    # the words: a title, and a card for every step at the lower left, in the tone's colour
    heard = next((s["text"] for s in steps if s["kind"] in ("question", "user", "heard")), "")
    text("brand", "CubbyLLM · live brain", 0.016, rgb("#dfe7f5"), (-0.405, 0.228, -1.0), FONT_B, cam)
    if heard:                                                 # (at 1 unit, the frame is 0.86 wide and 0.48 high)
        text("asked", "“" + heard + "”", 0.022, rgb("#ffb648"), (-0.405, 0.205, -1.0), FONT_B, cam)
    for i, s in enumerate(steps):
        zname = zones[s["zone"]]["name"]
        body = "\n".join(textwrap.wrap(s["text"] or "—", 52)[:3])
        head = text(f"card {i}", f"{s['kind'].upper()}  ·  {zname}", 0.0115, rgb(TONE.get(s["tone"], TONE["hot"])),
                    (-0.405, -0.135, -1.0), FONT_B, cam)
        text(f"card {i} body", body, 0.0145, rgb("#dfe7f5"), (0, -0.017, 0), FONT, head)
        lines = body.count("\n") + 1                            # a dark glass behind it: the brain glows, the words stay legible
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        pane = bpy.context.active_object
        pane.name, pane.parent = f"card {i} glass", head
        pane.location, pane.scale = (0.19, -0.015 - 0.009 * lines, -0.002), (0.41, 0.036 + 0.0185 * lines, 1)
        pane.data.materials.append(glow_mat(f"glass {i}", rgb("#0d1322"), 1.0))
        on, off = s["land"], (steps[i + 1]["land"] if i + 1 < len(steps) else end)
        keys(head, "scale", [(0, (0, 0, 0)), (F(on) - 1, (0, 0, 0)), (F(on), (1, 1, 1)),
                             (F(off) - 1, (1, 1, 1)), (F(off), (0, 0, 0))], "CONSTANT")
    sc.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")), copy=True)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if arg("--preview"):                                      # a few stills at half size
        sc.render.resolution_percentage = 50
        for s in steps[::max(1, len(steps) // 6)]:
            sc.frame_set(F(s["land"] + 0.3))
            sc.render.filepath = str(frames / f"preview_{sc.frame_current:04d}.png")
            bpy.ops.render.render(write_still=True)
        return
    sc.render.filepath = str(frames / "f_")
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                    str(frames / "f_%04d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out.with_suffix(".mp4"))], check=True)
    print(f"BRAIN done -> {out.with_suffix('.mp4')}")


main()
