
import bpy, os, time
sc = bpy.data.scenes["CubbyEmblems"]
bpy.context.window_manager  # noqa
OUT = r"C:\Users\grill\Documents\GitHub\CubbyLLM\dashboard\assets\sections\anim"
r = sc.render
r.resolution_x = r.resolution_y = 360
sc.cycles.samples = 32
sc.cycles.use_denoising = True
sc.frame_start, sc.frame_end = 1, 72
log = open(os.path.join(OUT, "_render.log"), "a")
for name in ["em_watch", "em_mood", "em_chat", "em_worlds", "em_forecast", "em_system", "em_agents"]:
    for c in sc.collection.children: c.hide_render = (c.name != name)
    d = os.path.join(OUT, name[3:]); os.makedirs(d, exist_ok=True)
    t = time.time()
    for f in range(1, 73):
        p = os.path.join(d, "f_%03d.png" % f)
        if os.path.exists(p): continue
        sc.frame_set(f); r.filepath = p
        bpy.ops.render.render(write_still=True, scene=sc.name)
    log.write("%s done in %.0f s\n" % (name, time.time() - t)); log.flush()
log.write("ALL DONE\n"); log.close()
