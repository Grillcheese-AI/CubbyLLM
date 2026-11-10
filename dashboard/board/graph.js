// The Knowledge section's graph (order 0055's store): entities as nodes, statements as labelled arrows, laid out by a
// small force simulation on a canvas. Drag a node to move it, drag the background to pan, wheel to zoom; a click
// selects a node, a double-click expands it (its own neighbours are fetched and merged). Plain canvas, no library.
import { reduced } from './api.js';

const COL = { node: '#B9CBD8', center: '#F4B860', sel: '#8ED8E8', edge: 'rgba(214,230,240,.38)', text: '#EEF3F6', sub: '#C3D0DA' };

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

  function setData(g, merge) {
    if (!merge) { nodes.clear(); edges.clear(); view = { x: 0, y: 0, k: 1 }; }
    const anchor = nodes.get(g.center);
    for (const n of g.nodes) {
      if (nodes.has(n.id)) { if (n.label) nodes.get(n.id).label = n.label; continue; }
      const a = Math.random() * Math.PI * 2, r = 60 + Math.random() * 80;
      nodes.set(n.id, { id: n.id, label: n.label, x: (anchor ? anchor.x : 0) + Math.cos(a) * r, y: (anchor ? anchor.y : 0) + Math.sin(a) * r, vx: 0, vy: 0, pin: false });
    }
    for (const e of g.edges) edges.set(`${e.from}|${e.pid}|${e.to}`, e);
    if (!merge) centerId = g.center;
    selected = g.center;
    heat = 1;
    if (reduced()) { for (let i = 0; i < 300; i++) step(); draw(); } else kick();
  }

  function step() {
    const ns = [...nodes.values()];
    for (let i = 0; i < ns.length; i++) {
      const a = ns[i];
      for (let j = i + 1; j < ns.length; j++) {
        const b = ns[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = dx * dx + dy * dy + 0.01, f = 5200 / d2;
        const d = Math.sqrt(d2); dx /= d; dy /= d;
        a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
      }
    }
    for (const e of edges.values()) {
      const a = nodes.get(e.from), b = nodes.get(e.to); if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01, f = (d - 170) * 0.02;
      a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
    }
    for (const n of ns) {
      if (n.pin || (drag && drag.node === n)) { n.vx = n.vy = 0; continue; }
      n.vx = (n.vx - n.x * 0.004) * 0.82; n.vy = (n.vy - n.y * 0.004) * 0.82;
      n.x += Math.max(-30, Math.min(30, n.vx)); n.y += Math.max(-30, Math.min(30, n.vy));
    }
    heat *= 0.985;
  }

  function kick() { if (!raf) raf = requestAnimationFrame(tick); }
  function tick() { raf = 0; step(); draw(); if (heat > 0.02 || drag) kick(); }

  function arrow(a, b, color) {
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, ux = dx / d, uy = dy / d;
    const ex = b.x - ux * 13, ey = b.y - uy * 13;
    ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(a.x + ux * 10, a.y + uy * 10); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ux * 8 - uy * 4, ey - uy * 8 + ux * 4); ctx.lineTo(ex - ux * 8 + uy * 4, ey - uy * 8 - ux * 4); ctx.fill();
  }

  function draw() {
    const w = W(), h = H();
    ctx.clearRect(0, 0, w, h);
    ctx.save(); ctx.translate(w / 2 + view.x, h / 2 + view.y); ctx.scale(view.k, view.k);
    ctx.lineWidth = 1.2 / view.k;
    const few = edges.size <= 30;   // more than that: only the selected node's edges say their property
    for (const e of edges.values()) {
      const a = nodes.get(e.from), b = nodes.get(e.to); if (!a || !b) continue;
      const on = selected && (e.from === selected || e.to === selected);
      arrow(a, b, on ? 'rgba(142,216,232,.8)' : COL.edge);
      if (few || on) {
        ctx.fillStyle = on ? COL.sel : COL.sub; ctx.font = `${11 / view.k}px Segoe UI, system-ui, sans-serif`; ctx.textAlign = 'center';
        ctx.fillText(e.label || e.pid, (a.x + b.x) / 2, (a.y + b.y) / 2 - 4 / view.k);
      }
    }
    for (const n of nodes.values()) {
      const c = n.id === selected ? COL.sel : n.id === centerId ? COL.center : COL.node;
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(n.x, n.y, (n.id === centerId ? 9 : 7) / Math.sqrt(view.k), 0, Math.PI * 2); ctx.fill();
      if (n === hover) { ctx.strokeStyle = '#FFF'; ctx.stroke(); }
      ctx.fillStyle = COL.text; ctx.font = `${12 / view.k}px Segoe UI, system-ui, sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(n.label ? `${n.label}` : n.id, n.x + 11 / view.k, n.y + 4 / view.k);
    }
    ctx.restore();
  }

  function hit(px, py) {
    const p = toWorld(px, py);
    let best = null, bd = (14 / view.k) ** 2;
    for (const n of nodes.values()) { const d = (n.x - p.x) ** 2 + (n.y - p.y) ** 2; if (d < bd) { bd = d; best = n; } }
    return best;
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
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const f = Math.exp(-ev.deltaY * 0.0015), k = Math.max(0.25, Math.min(4, view.k * f));
    const [px, py] = local(ev), mx = px - W() / 2, my = py - H() / 2;
    view.x = mx - (mx - view.x) * (k / view.k); view.y = my - (my - view.y) * (k / view.k); view.k = k; draw();
  }, { passive: false });
  window.addEventListener('resize', () => { size(); draw(); });
  size();

  return { setData, select(id) { selected = id; draw(); }, count: () => nodes.size, redraw() { size(); draw(); } };
}
