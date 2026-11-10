// The Knowledge section's graph (order 0055's store): entities as rounded squares (a class icon, the id, a label pill, a
// HOT / COLD mark), statements as arrows with their property in a box. A path's chain is laid left to right and drawn
// cyan; its context (one hop more) dashed. Other nodes settle by a small force simulation. Drag a node to move it,
// drag the background to pan, wheel or the buttons to zoom, Fit to see all; a click selects, a double-click expands.
// Plain canvas, no library.
import { reduced } from './api.js';
import { ICONS, kind } from './store_views.js';

const C = { box: '#13233A', line: 'rgba(148,163,184,.35)', path: '#38BDF8', ctx: 'rgba(148,163,184,.45)', text: '#E6EDF5',
  sub: '#8FA0B4', sel: '#2DD4BF', hot: '#10B981', cold: '#94A3B8', pill: 'rgba(10,17,29,.92)' };
const S = 22, GAP = 230;   // half a node's side; the spacing of a path's chain
const icons = Object.fromEntries(Object.entries(ICONS).map(([k, d]) => [k, new Path2D(d)]));

function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

export function graphView(canvas, { onSelect, onExpand } = {}) {
  const ctx = canvas.getContext('2d');
  const nodes = new Map(), edges = new Map();
  let view = { x: 0, y: 0, k: 1 }, selected = null, centerId = null, heat = 0, raf = 0, drag = null, hover = null;

  function size() {
    const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, r.width * d); canvas.height = Math.max(1, r.height * d);
    ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  const W = () => canvas.getBoundingClientRect().width, H = () => canvas.getBoundingClientRect().height;
  const toWorld = (px, py) => ({ x: (px - W() / 2 - view.x) / view.k, y: (py - H() / 2 - view.y) / view.k });

  // g: {center, nodes:[{id,label,types,cold,on_path}], edges:[{from,to,pid,label,on_path}]}; chain: the path's ids in order
  function setData(g, merge, chain) {
    if (!merge) { nodes.clear(); edges.clear(); view = { x: 0, y: 0, k: 1 }; }
    const anchor = nodes.get(g.center);
    const at = new Map((chain || []).map((id, i) => [id, { x: (i - (chain.length - 1) / 2) * GAP, y: 0 }]));
    for (const n of g.nodes) {
      const old = nodes.get(n.id), fix = at.get(n.id);
      if (old) { Object.assign(old, { label: n.label || old.label, types: n.types || old.types, cold: n.cold ?? old.cold }); continue; }
      const a = Math.random() * Math.PI * 2, r = 90 + Math.random() * 90, o = anchor || { x: 0, y: 0 };
      nodes.set(n.id, { id: n.id, label: n.label, types: n.types || [], cold: n.cold, x: fix ? fix.x : o.x + Math.cos(a) * r,
        y: fix ? fix.y : o.y + Math.sin(a) * r, vx: 0, vy: 0, pin: !!fix, path: !!fix });
    }
    for (const e of g.edges) edges.set(`${e.from}|${e.pid}|${e.to}`, e);
    if (!merge) centerId = chain && chain.length ? null : g.center;
    selected = g.center || (chain || [])[0];
    heat = 1;
    if (reduced()) { for (let i = 0; i < 300; i++) step(); fit(); } else { for (let i = 0; i < 120; i++) step(); fit(); kick(); }
  }

  function step() {
    const ns = [...nodes.values()];
    for (let i = 0; i < ns.length; i++) {
      const a = ns[i];
      for (let j = i + 1; j < ns.length; j++) {
        const b = ns[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = dx * dx + dy * dy + 0.01, f = 9000 / d2;
        const d = Math.sqrt(d2); dx /= d; dy /= d;
        a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
      }
    }
    for (const e of edges.values()) {
      const a = nodes.get(e.from), b = nodes.get(e.to); if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01, f = (d - (e.on_path === false ? 150 : 200)) * 0.02;
      a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
    }
    for (const n of ns) {
      if (n.pin || (drag && drag.node === n)) { n.vx = n.vy = 0; continue; }
      n.vx = (n.vx - n.x * 0.003) * 0.82; n.vy = (n.vy - n.y * 0.003) * 0.82;
      n.x += Math.max(-30, Math.min(30, n.vx)); n.y += Math.max(-30, Math.min(30, n.vy));
    }
    heat *= 0.985;
  }

  function kick() { if (!raf) raf = requestAnimationFrame(tick); }
  function tick() { raf = 0; step(); draw(); if (heat > 0.02 || drag) kick(); }

  function edge(e, a, b, few) {
    const on = selected && (e.from === selected || e.to === selected), path = e.on_path === true, ctxe = e.on_path === false;
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, ux = dx / d, uy = dy / d;
    const sx = a.x + ux * (S + 4), sy = a.y + uy * (S + 4), ex = b.x - ux * (S + 6), ey = b.y - uy * (S + 6);
    const col = path ? C.path : on ? C.sel : ctxe ? C.ctx : C.line;
    ctx.strokeStyle = col; ctx.lineWidth = (path ? 2.2 : 1.2) / Math.max(view.k, 0.6);
    ctx.setLineDash(ctxe ? [6, 5] : []); ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ux * 9 - uy * 4.5, ey - uy * 9 + ux * 4.5); ctx.lineTo(ex - ux * 9 + uy * 4.5, ey - uy * 9 - ux * 4.5); ctx.fill();
    if (!(few || on || path)) return;
    const t = `${e.label || e.pid}  ${e.pid}`, mx = (sx + ex) / 2, my = (sy + ey) / 2;
    ctx.font = '11px Segoe UI, system-ui, sans-serif'; const w = ctx.measureText(t).width + 14;
    rrect(ctx, mx - w / 2, my - 10, w, 20, 6); ctx.fillStyle = C.pill; ctx.fill();
    ctx.strokeStyle = path ? 'rgba(56,189,248,.55)' : 'rgba(148,163,184,.3)'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = path ? C.path : on ? C.sel : C.sub; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, mx, my + 0.5);
  }

  function node(n) {
    const sel = n.id === selected, main = n.id === centerId || n.path;
    rrect(ctx, n.x - S, n.y - S, 2 * S, 2 * S, 11); ctx.fillStyle = C.box; ctx.fill();
    ctx.lineWidth = sel ? 2.4 : main ? 1.8 : 1; ctx.strokeStyle = sel ? C.sel : main ? C.path : n === hover ? '#CBD5E1' : 'rgba(148,163,184,.35)'; ctx.stroke();
    ctx.save(); ctx.translate(n.x - 12, n.y - 15); ctx.strokeStyle = main || sel ? C.sel : C.sub; ctx.lineWidth = 1.7; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.stroke(icons[kind(n.types)]); ctx.restore();
    ctx.font = '9px Cascadia Mono, Consolas, monospace'; ctx.fillStyle = C.sub; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(n.id, n.x, n.y + 14);
    if (n.cold !== undefined && n.cold !== null) {
      ctx.fillStyle = n.cold ? C.cold : C.hot; ctx.beginPath(); ctx.arc(n.x + S - 3, n.y - S + 3, 4, 0, Math.PI * 2); ctx.fill();
    }
    const label = n.label || n.id, short = label.length > 28 ? label.slice(0, 27) + '…' : label;
    ctx.font = `${main ? 600 : 500} 12px Segoe UI, system-ui, sans-serif`; const w = ctx.measureText(short).width + 16;
    rrect(ctx, n.x - w / 2, n.y + S + 6, w, 22, 11); ctx.fillStyle = C.pill; ctx.fill();
    ctx.strokeStyle = main || sel ? 'rgba(56,189,248,.45)' : 'rgba(148,163,184,.25)'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = C.text; ctx.fillText(short, n.x, n.y + S + 17.5);
  }

  function draw() {
    const w = W(), h = H();
    ctx.clearRect(0, 0, w, h);
    ctx.save(); ctx.translate(w / 2 + view.x, h / 2 + view.y); ctx.scale(view.k, view.k);
    const few = edges.size <= 24;   // more than that: only the selected node's and the path's edges say their property
    for (const e of edges.values()) { const a = nodes.get(e.from), b = nodes.get(e.to); if (a && b) edge(e, a, b, few); }
    for (const n of nodes.values()) node(n);
    ctx.restore();
  }

  function fit() {
    if (!nodes.size) { draw(); return; }
    const xs = [...nodes.values()].map((n) => n.x), ys = [...nodes.values()].map((n) => n.y);
    const x0 = Math.min(...xs) - 80, x1 = Math.max(...xs) + 80, y0 = Math.min(...ys) - 60, y1 = Math.max(...ys) + 90;
    const k = Math.max(0.25, Math.min(1.4, Math.min(W() / (x1 - x0), H() / (y1 - y0))));
    view = { k, x: -(x0 + x1) / 2 * k, y: -(y0 + y1) / 2 * k };
    draw();
  }
  function zoom(f, px = W() / 2, py = H() / 2) {
    const k = Math.max(0.25, Math.min(4, view.k * f)), mx = px - W() / 2, my = py - H() / 2;
    view.x = mx - (mx - view.x) * (k / view.k); view.y = my - (my - view.y) * (k / view.k); view.k = k; draw();
  }

  function hit(px, py) {
    const p = toWorld(px, py);
    for (const n of [...nodes.values()].reverse()) if (Math.abs(n.x - p.x) <= S + 2 && Math.abs(n.y - p.y) <= S + 30) return n;
    return null;
  }
  const local = (ev) => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };

  canvas.addEventListener('pointerdown', (ev) => {
    const [px, py] = local(ev), n = hit(px, py);
    canvas.setPointerCapture(ev.pointerId);
    drag = n ? { node: n, moved: false } : { pan: true, px, py, vx: view.x, vy: view.y, moved: false };
  });
  canvas.addEventListener('pointermove', (ev) => {
    const [px, py] = local(ev);
    if (!drag) { const h0 = hover; hover = hit(px, py); canvas.style.cursor = hover ? 'pointer' : 'grab'; if (h0 !== hover) draw(); return; }
    drag.moved = true;
    if (drag.node) { const p = toWorld(px, py); drag.node.x = p.x; drag.node.y = p.y; drag.node.pin = true; heat = Math.max(heat, 0.3); kick(); }
    else { view.x = drag.vx + px - drag.px; view.y = drag.vy + py - drag.py; draw(); }
  });
  canvas.addEventListener('pointerup', () => {
    if (drag && drag.node && !drag.moved) { selected = drag.node.id; draw(); onSelect && onSelect(drag.node.id); }
    drag = null;
  });
  canvas.addEventListener('dblclick', (ev) => { const n = hit(...local(ev)); if (n && onExpand) onExpand(n.id); });
  canvas.addEventListener('wheel', (ev) => { ev.preventDefault(); zoom(Math.exp(-ev.deltaY * 0.0015), ...local(ev)); }, { passive: false });
  window.addEventListener('resize', () => { size(); draw(); });
  size();

  return { setData, fit, zoom, select(id) { selected = id; draw(); }, count: () => [nodes.size, edges.size], redraw() { size(); fit(); } };
}
