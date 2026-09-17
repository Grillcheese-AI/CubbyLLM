"""pacui_threejs - ONE renderer for the pac world. Swap this file, keep the rest.

Wired: WIRED (the /pac frontend). Depends on the world for a replay; the world
depends on nothing here.

Owner: *"separate logic from the frontend too so if we want to port it on a
more sophisticated 3d engine than threejs it will be easy."* This is that file.
Every three.js string in the project now lives in it: the lifted page, the
patch list applied to it, the console panel, the sound panel, the asset and
music loaders, and the replay writer.

THE CONTRACT A RENDERER CONSUMES is three dicts the agent already produces and
which contain no markup, no element ids and no engine anywhere:

    init_payload()   the level: walls, hazards, pellets, power, mines, start
    resp()           the frame: where he is, what he ate, what the ghosts did
    affect()         how he is, plus what he holds about whoever is watching

A new engine implements those three and nothing else. It does not need to know
that this file exists, and this file does not need to know that it does.

WHAT IS STILL UGLY, said plainly rather than hidden: the page is a third-party
file read off disk and modified by a list of string replacements, and
`load_frontend` fails loudly when a patch stops matching. That is the right
behaviour for a lifted page nobody here owns, and it is also the strongest
argument for replacing it with an engine whose page we write.
"""
from __future__ import annotations

import json
import pathlib
import re

from pacpaths import (ASSETS_DIR, MUSIC_PATH, PACMAN_3D, PACMAN_LIVE,  # noqa: E402
                      REPLAY_OUT)
from pacworld import PacVerse  # noqa: E402
__wiring__ = "WIRED"



# the lifted page's strings we change: endpoint prefixes (so it can live under
# /pac next to the console), the subtitle/foot (say what actually drives him),
# and the two HUD rows we back with different real numbers than the original
_FRONTEND_PATCHES = [
    ("fetch('/state')", "fetch('/pac/state')"),
    # Keep what he says on screen long enough to READ it (owner, 2026-09-15).
    # 1.6s was set when the line was a template and you already knew what it
    # would say; now the model writes it and it is worth reading, so the banner
    # holds for a floor plus reading time, capped. `clearTimeout` is the part
    # that matters: at these durations two sentences in quick succession left
    # the first one's timer to hide the SECOND one early.
    ("setTimeout(()=>{$('banner').style.display='none';},1600);",
     "clearTimeout(window._sayT); window._sayT=setTimeout("
     "()=>{$('banner').style.display='none';},"
     "Math.min(15000, 3400+60*((s.says||'').length)));"),
    ('function build(D){', 'function build(D){ if(window.cbLevelStart) cbLevelStart(D);'),   # the level-start jingle
    ("  if(s.lives!=null) $('lives').innerHTML=",
     "  if(window.cbSfx) cbSfx(s);\n  if(s.lives!=null) $('lives').innerHTML="),   # the sound layer reads every frame
    ("fetch('/init')", "fetch('/pac/init')"),
    ("fetch('/next')", "fetch('/pac/next')"),
    ("'/assets/", "'/pac/assets/"),
    ('<div id="sub">live MoWM + VSA planner · CVL value</div>',
     '<div id="sub">live CubbyLLM stand-in brain · every move VM-guarded · learns as he explores</div>'),
    ("isometric · drag to orbit · CVL picks the target (evades ghosts) · BFS plans + discovers JUMP · "
     "mSA ghosts flank · fear is learned",
     "isometric · drag to orbit · explorer brain: the VM offers the exits and guards the choice · "
     "walls learned from refused moves · joins are his own programs · fear is learned"),
    ('R-STDP learned: <b id="wd">dopa +0</b> · <b id="wc">cort 0</b> · <b id="wt">treat 0</b>',
     'world model: <b id="wd">facts 0</b> · <b id="wc">derived 0</b> · <b id="wt">walls 0</b>'),
    ("$('wd').textContent='dopa '+sg(s.w_dopa); $('wc').textContent='cort '+sg(s.w_cort); "
     "$('wt').textContent='treat '+sg(s.w_treat);",
     "$('wd').textContent='facts '+s.w_dopa; $('wc').textContent='derived '+s.w_cort; "
     "$('wt').textContent='walls '+s.w_treat;"),
    ('covers tracks (learned): <b id="laylow">0.3</b>', 'refused moves (walls learned): <b id="laylow">0</b>'),
    ("$('laylow').textContent=s.lay_low.toFixed(2)", "$('laylow').textContent=String(s.lay_low)"),
    ("INTERVAL=320", "INTERVAL=600"),                   # our VM-guarded steps are slower than the planner's
    # the letter/word mechanic is gone: the speech line shows his THOUGHT, the flash says so
    ("if(s.word){ const prog=(s.collected||'').split('').join(' '); $('speaktxt').textContent = s.says ? "
     "('💬 says: \"'+s.says+'\"') : ('letters: '+(prog||'—')); }",
     "$('speaktxt').textContent = s.thought ? ('💭 '+s.thought) : '';"),
    ("flash('💬 CUBBY SAYS: \"'+s.says+'\"','#7fe0ff')", "flash('💭 '+s.says,'#7fe0ff')"),
    # traps (owner, 2026-09-02): a flashing red vortex per mine, drawn in the page's own scene
    ("let scene,cam,rnd,ctr,composer,vintagePass,agent,group,pellets,trailDots=[]",
     "let scene,cam,rnd,ctr,composer,vintagePass,agent,group,pellets,mines=new Map(),trailDots=[]"),
    ("function drawCompass(angle,intensity,name,color,msg){",
     "function syncMines(list){ const want=new Set((list||[]).map(c=>c.join(','))); "
     "for(const [k,m] of mines){ if(!want.has(k)){ group.remove(m); mines.delete(k); } } "
     "for(const c of (list||[])){ const k=c.join(','); if(mines.has(k)) continue; "
     "const m=new THREE.Mesh(new THREE.TorusGeometry(0.30,0.07,10,28), new THREE.MeshStandardMaterial({color:0xff2d2d,emissive:0xff1a1a,emissiveIntensity:1.0,metalness:0.3,roughness:0.4})); "
     "m.position.set(...W3(...c)); m.rotation.x=Math.PI/2; m.userData.vortex=true; group.add(m); mines.set(k,m); } }\n"
     "function drawCompass(angle,intensity,name,color,msg){"),
    ("pellets.set(p.join(','),m);}", "pellets.set(p.join(','),m);}\n  syncMines(D.mines);"),
    ("if(s.eaten){const k=s.eaten.join(',');", "syncMines(s.mines); if(s.eaten){const k=s.eaten.join(',');"),
    ("'  ·  pellets left: '+s.remaining", "'  ·  pellets left: '+s.remaining+'  ·  ◎ traps: '+(s.mines_left??0)"),
    ("<b>● pellet</b> · <i>◆ hazard</i>", "<b>● pellet</b> · <b style=\"color:#ff3b3b\">◎ trap</b> · <i>◆ hazard</i>"),
    ("pellets.forEach(m=>{ if(m.userData.star) m.rotation.z=now/600; });",
     "pellets.forEach(m=>{ if(m.userData.star) m.rotation.z=now/600; }); "
     "mines.forEach(m=>{ m.rotation.z=now/220; const sc=1+0.22*Math.sin(now/130); m.scale.set(sc,sc,1); "
     "m.material.emissiveIntensity=0.6+0.9*Math.abs(Math.sin(now/160)); });"),
]



# the console panel (owner, 2026-09-02): the brain's event feed (GET /events)
# under the big stage number, so the game and what Cubby does are one screen.
# Injected before </body>; everything is ours, nothing of the lifted page moves.
_CONSOLE_PANEL = r"""
<style>
  #cubbycon{position:fixed;right:3vw;top:71vh;bottom:12px;width:min(36vw,480px);z-index:12;background:rgba(8,12,20,.86);
    border:1px solid #2a3a55;border-radius:12px;box-shadow:0 8px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;
    font:11.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#c9d1d9}
  #cubbycon .hd{padding:7px 12px;border-bottom:1px solid #2a3a55;color:#8b949e;display:flex;gap:10px;align-items:center}
  #cubbycon .hd b{color:#58a6ff}#cubbycon .hd a{color:#58a6ff;text-decoration:none;margin-left:auto}
  #cubbycon .hd label{cursor:pointer}
  #cubbylog{flex:1;overflow-y:auto;padding:8px 12px;white-space:pre-wrap;word-break:break-word}
  #cubbylog .ev{margin:1px 0}#cubbylog .k{display:inline-block;min-width:88px}
  .k-user{color:#d29922}.k-sense{color:#bc8cff}.k-route{color:#58a6ff}.k-walk{color:#3fb950}.k-emit{color:#8b949e}
  .k-vm{color:#3fb950}.k-gate{color:#d29922}.k-learn{color:#3fb950}.k-speak{color:#c9d1d9}.k-explore{color:#58a6ff}
  .k-probe{color:#f85149}.k-derive{color:#bc8cff}.k-eat{color:#ffe66d}.k-caught{color:#f85149}.k-flee{color:#ff9f43}
  .k-plan{color:#8b949e}.k-program{color:#bc8cff}.k-forge{color:#bc8cff}.k-superpower_move{color:#27d3ff}
  .k-level_up{color:#2ce6a8}.k-out_of_time{color:#ff7043}.k-live_error{color:#f85149}.k-ghost_eaten{color:#2ce6a8}
  .k-says{color:#ffe66d}.k-retire{color:#8b949e}.k-restart{color:#ff7043}
  .k-thought{color:#ffe66d;font-style:italic}.k-modify{color:#bc8cff}
</style>
<div id="cubbycon"><div class="hd"><b>cubby</b> console · what he does, as he does it
  <label><input id="cubbyauto" type="checkbox" checked> follow</label><a href="/" target="_blank">full console ↗</a></div>
  <div id="cubbylog"></div></div>
<script>
(function(){
  const log=document.getElementById('cubbylog'), auto=document.getElementById('cubbyauto');
  let since=0; const H=['dopamine','serotonin','cortisol','oxytocin','noradrenaline'];
  const esc=s=>{const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;};
  function fmt(e){const d={...e};delete d.i;delete d.t;delete d.kind;
    if(e.kind==='sense')return `emotion=${d.emotion} `+H.map(h=>`${h.slice(0,4)}=${(d.state?.[h]??0).toFixed(2)}`).join(' ');
    if(e.kind==='walk')return `${d.verified?'VERIFIED':(d.reason||'walked')} answer=${d.answer??'-'} facts=[${(d.facts||[]).join(' | ')}]`;
    if(e.kind==='speak')return `(${d.turn_kind}, ${d.register||'-'}) "${d.reply}"`;
    if(e.kind==='program')return `${d.name} (${d.pattern||d.why}) — ${d.why}: ${(d.reasoning&&d.reasoning.verdict)||''}`;
    if(e.kind==='forge')return `${d.name} ${d.ok?'CERTIFIED':'REJECTED'} got=${d.got} expected=${d.expected}`;
    if(e.kind==='explore')return `${d.from||''} —${d.chosen}→ ${d.place} new=${d.new}${d.probed?' probed='+d.probed:''}`;
    if(e.kind==='thought')return `“${d.text}”${d.verbalized?'':' (host)'}`;
    return Object.entries(d).filter(([k])=>k!=='program'&&k!=='reasoning').map(([k,v])=>`${k}=${typeof v==='object'?JSON.stringify(v):v}`).join(' ');}
  async function poll(){
    try{const r=await fetch(`/events?since=${since}`);const j=await r.json();
      for(const e of j.events){log.insertAdjacentHTML('beforeend',`<div class="ev"><span class="k k-${e.kind}">${esc(e.kind)}</span> ${esc(fmt(e))}</div>`);}
      while(log.children.length>400)log.removeChild(log.firstChild);
      since=j.next; if(auto.checked&&j.events.length)log.scrollTop=log.scrollHeight;
    }catch(err){}
    setTimeout(poll,700);
  }
  poll();
})();
</script>
"""


# the sound layer (2026-09-02): the owner's soundtrack loops quietly under
# 8-bit oscillator effects for the game's events (a step ticks, a pellet is
# the two-note waka, a star an arpeggio, a trap a noise burst, a catch a
# falling saw…). Browsers need a gesture before audio: the toggle is that
# gesture; the choice and the music volume are remembered per browser.
_SOUND_PANEL = r"""
<audio id="bgm" loop preload="auto" src="/pac/music"></audio>
<audio id="jingle" preload="auto" src="/pac/sfx/game_start"></audio>
<div id="sndbox" style="position:fixed;left:14px;bottom:14px;z-index:60;display:flex;gap:8px;align-items:center;
  background:rgba(8,10,24,.78);border:1px solid rgba(120,140,255,.35);border-radius:10px;padding:6px 10px;
  font:600 12px system-ui,sans-serif;color:#dfe6ff;backdrop-filter:blur(6px)">
  <button id="sndbtn" style="background:#22264a;color:#dfe6ff;border:1px solid #5a63b0;border-radius:6px;padding:3px 9px;cursor:pointer;font:inherit">♪ sound off</button>
  <label style="display:flex;align-items:center;gap:5px;opacity:.9">music <input id="bgmvol" type="range" min="0" max="60" value="18" style="width:70px"></label>
</div>
<script>
(function(){
  const bgm=document.getElementById('bgm'), btn=document.getElementById('sndbtn'), vol=document.getElementById('bgmvol');
  let ac=null, on=false;
  try{ on=localStorage.getItem('cb_sound')==='1'; const v=localStorage.getItem('cb_bgm'); if(v!=null) vol.value=v; }catch(e){}
  bgm.volume=vol.value/100;                                   // the soundtrack sits UNDER the effects
  function ctx(){ if(!ac){ const AC=window.AudioContext||window.webkitAudioContext; if(AC) ac=new AC(); }
    if(ac&&ac.state==='suspended') ac.resume(); return ac; }
  function paint(){ btn.textContent=on?'♪ sound on':'♪ sound off'; btn.style.background=on?'#2f8a5a':'#22264a'; }
  function start(){ ctx(); bgm.play().catch(()=>{}); }
  function stop(){ bgm.pause(); }
  btn.onclick=function(){ on=!on; try{ localStorage.setItem('cb_sound',on?'1':'0'); }catch(e){} paint(); if(on){ start(); SFX.pellet(); } else stop(); };
  vol.oninput=function(){ bgm.volume=vol.value/100; try{ localStorage.setItem('cb_bgm',vol.value); }catch(e){} };
  if(on){ const kick=()=>{ start(); window.removeEventListener('pointerdown',kick); window.removeEventListener('keydown',kick); };
    window.addEventListener('pointerdown',kick); window.addEventListener('keydown',kick); }
  paint();
  // 8-bit voice: square/triangle/saw oscillators with short envelopes
  function tone(f0, ms, type, vol, f1){ const a=ctx(); if(!a||!on) return; const t=a.currentTime, o=a.createOscillator(), g=a.createGain();
    o.type=type||'square'; o.frequency.setValueAtTime(f0,t); if(f1) o.frequency.exponentialRampToValueAtTime(f1,t+ms/1000);
    g.gain.setValueAtTime(vol||0.1,t); g.gain.exponentialRampToValueAtTime(0.0001,t+ms/1000);
    o.connect(g); g.connect(a.destination); o.start(t); o.stop(t+ms/1000+0.02); }
  function seq(notes, step, type, vol){ notes.forEach((f,i)=>setTimeout(()=>tone(f, step*1.7, type, vol), i*step)); }
  function noise(ms, vol){ const a=ctx(); if(!a||!on) return; const n=Math.floor(a.sampleRate*ms/1000), b=a.createBuffer(1,n,a.sampleRate), d=b.getChannelData(0);
    for(let i=0;i<n;i++) d[i]=(Math.random()*2-1)*(1-i/n); const src=a.createBufferSource(); src.buffer=b; const g=a.createGain(); g.gain.value=vol||0.16;
    src.connect(g); g.connect(a.destination); src.start(); }
  const SFX={
    move:      ()=>tone(170,45,'triangle',0.05),
    jump:      ()=>tone(280,140,'square',0.09,1100),
    rest:      ()=>tone(330,260,'sine',0.05,495),
    pellet:    ()=>{ tone(880,60,'square',0.11); setTimeout(()=>tone(1320,80,'square',0.10),60); },
    star:      ()=>seq([523,659,784,1047,1319],70,'square',0.11),
    ghost:     ()=>seq([1047,1319,1568,2093],55,'square',0.11),
    caught:    ()=>tone(540,650,'sawtooth',0.13,55),
    gameover:  ()=>seq([392,349,311,262,196],170,'sawtooth',0.12),
    trapped:   ()=>{ noise(200,0.2); tone(130,240,'square',0.12,40); },
    mine:      ()=>tone(250,100,'square',0.08,125),
    level:     ()=>seq([523,659,784,1047,784,1047,1319],95,'square',0.12),
    fail:      ()=>seq([440,415,392,370],150,'triangle',0.10),
    superpower:()=>seq([659,880,1175,1568],65,'square',0.10),
  };
  // level start: the jingle plays over the ducked soundtrack (owner's game-start sample)
  const jingle=document.getElementById('jingle'); jingle.volume=0.55;
  jingle.addEventListener('ended', ()=>{ bgm.volume=vol.value/100; });
  window.cbLevelStart=function(D){ if(!on) return; try{ bgm.volume=(vol.value/100)*0.3; jingle.currentTime=0; jingle.play().catch(()=>{ bgm.volume=vol.value/100; }); }catch(e){} };
  let last={fr:0, to:null, rest:false};
  window.cbSfx=function(s){
    const fr=s.frightened||0, star=!!s.eaten && fr>last.fr, moved=s.to && (!last.to || s.to[0]!==last.to[0]||s.to[1]!==last.to[1]||s.to[2]!==last.to[2]);
    if(on){
      if(s.caught) SFX[s.caught==='gameover'?'gameover':'caught']();
      else if(s.trapped>0) SFX.trapped();
      else if(s.ate_ghost) SFX.ghost();
      else if(s.beaten) SFX.level();
      else if(s.failed) SFX.fail();
      else if(star) SFX.star();
      else if(s.learned) SFX.superpower();
      else if(s.eaten) SFX.pellet();
      else if(s.mined) SFX.mine();
      else if(s.move==='rest'){ if(!last.rest) SFX.rest(); }
      else if(s.move && s.move.indexOf('jump_')===0) SFX.jump();
      else if(moved) SFX.move();
    }
    last={fr:fr, to:s.to, rest:s.move==='rest'};
  };
})();
</script>
"""



def load_frontend(source: pathlib.Path = PACMAN_LIVE) -> tuple[str | None, list[str]]:
    """pacman_live.py's EXACT page, extracted at serve time (never imported),
    with the patches above applied and our console panel injected before
    </body>. -> (html, patches that did not apply)."""
    try:
        src = source.read_text(encoding="utf-8")
    except OSError:
        return None, ["source missing"]
    m = re.search(r'FRONTEND = r"""(.*?)"""\s*\n', src, re.S)
    if not m:
        return None, ["FRONTEND block not found"]
    html, missed = m.group(1), []
    for old, new in _FRONTEND_PATCHES:
        if old in html:
            html = html.replace(old, new)
        else:
            missed.append(old[:40])
    if "</body>" in html:
        html = html.replace("</body>", _CONSOLE_PANEL + _SOUND_PANEL + "</body>", 1)
    else:
        html += _CONSOLE_PANEL + _SOUND_PANEL
    return html, missed



def load_music(path: pathlib.Path = MUSIC_PATH) -> bytes | None:
    """The background soundtrack (the owner's "Neon Pixel Dash", 2026-09-02;
    override with CB_PAC_MUSIC). None when absent -> the page stays silent
    but the 8-bit effects still play."""
    try:
        return path.read_bytes() if path.suffix.lower() in (".mp3", ".ogg", ".wav") and path.is_file() else None
    except OSError:
        return None



def load_sfx(name: str, base: pathlib.Path = MUSIC_PATH.parent) -> bytes | None:
    """A sampled effect from the music dir: /pac/sfx/<name> -> <name>.mp3
    (name-safe). Today: `game_start` (the level-start jingle)."""
    if not re.fullmatch(r"[a-z0-9_]{1,40}", name or ""):
        return None
    try:
        fp = base / f"{name}.mp3"
        return fp.read_bytes() if fp.is_file() else None
    except OSError:
        return None



def load_asset(rel: str, base: pathlib.Path = ASSETS_DIR) -> bytes | None:
    """A face texture from cubbyverse's assets dir (png only, path-safe)."""
    try:
        fp = (base / rel.split("?")[0]).resolve()
        if fp.suffix.lower() == ".png" and base.resolve() in fp.parents and fp.is_file():
            return fp.read_bytes()
    except OSError:
        pass
    return None



def render_replay(env: PacVerse, traj: list[dict],
                  out: pathlib.Path = REPLAY_OUT, source: pathlib.Path = PACMAN_3D) -> pathlib.Path | None:
    """His real run through the SAME three.js page pacman_3d.py ships —
    extracted from the file's `_HTML` template at render time, never
    imported. None when the cubbyverse checkout is not on this machine."""
    try:
        src = source.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'_HTML = r"""(.*)"""', src, re.S)
    if not m:
        return None
    html = m.group(1)
    html = html.replace("3D Pac-Man — learned by the Mixture of World Models",
                        "3D Pac-Man — explored live by cubby-man")
    html = html.replace("agent learned 6 move operators, then VSA-planned the shortest "
                        "wall-avoiding path to every pellet",
                        "cubby-man learned this maze move by move — walls from VM refusals, "
                        "every fact in his own world model")
    html = html.replace("VSA planner pathing to each pellet…",
                        "exploring — every move VM-guarded…")
    data = {"w": env.w, "h": env.h, "d": env.d,
            "start": list(env.coords(env.start)),
            "walls": [list(w) for w in sorted(env.walls)],
            "pellets": [list(p) for p in env.pellets],
            "total": len(env.pellets), "traj": traj}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*DATA*/", json.dumps(data)), encoding="utf-8")
    return out
