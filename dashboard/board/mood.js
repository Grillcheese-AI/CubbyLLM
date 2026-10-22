// Mood (order 0027): how Cubby feels right now, from /board/mood -- the server's /state, a display heartbeat (40 + 55 x
// arousal bpm: a display mapping, not a claim), the point in the Lövheim cube with its nearest corner, and the levels.
// The emotional arcs of a book or a chat wait for order 0020 and say so.
import { get, el, fmt } from './api.js';

let timer = null;

function bar(name, v) {
  const w = Math.max(0, Math.min(1, Number(v) || 0));
  return el('div', { class: 'lvl' }, el('span', { class: 'ln' }, name),
    el('span', { class: 'lb', role: 'meter', 'aria-label': name, 'aria-valuemin': '0', 'aria-valuemax': '1', 'aria-valuenow': w.toFixed(2) },
      el('i', { style: `width:${(100 * w).toFixed(1)}%` })),
    el('span', { class: 'lv mono' }, w.toFixed(2)));
}

function cube(c) {
  // the cube drawn in a fixed oblique projection: x = 5-HT, y = DA (up), z = NE (back)
  const P = (x, y, z) => [40 + 120 * x + 50 * z, 170 - 120 * y - 40 * z];
  const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 220 200');
  svg.setAttribute('class', 'cube');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `The point in the cube: 5-HT ${c.pos[0]}, DA ${c.pos[1]}, NE ${c.pos[2]}; ${c.at_rest ? 'at rest' : 'nearest corner ' + c.corner}`);
  for (const [a, b] of edges) {
    const [x1, y1] = P(...corners[a]), [x2, y2] = P(...corners[b]);
    const l = document.createElementNS(ns, 'line');
    Object.entries({ x1, y1, x2, y2 }).forEach(([k, v]) => l.setAttribute(k, v));
    svg.appendChild(l);
  }
  const [px, py] = P(...c.pos);
  const dot = document.createElementNS(ns, 'circle');
  Object.entries({ cx: px, cy: py, r: 6 }).forEach(([k, v]) => dot.setAttribute(k, v));
  svg.appendChild(dot);
  return svg;
}

export async function show(root) {
  clearInterval(timer);
  const r = await get('/board/mood');
  const m = r.body || {};
  if (!r.ok || !m.built) {
    root.replaceChildren(el('div', { class: 'empty glass' }, el('h1', { class: 'h1' }, 'Mood'), el('p', {}, m.message || m.error || 'unavailable')));
    return;
  }
  const beat = 60 / Math.max(1, m.bpm);
  const c = m.cube;
  root.replaceChildren(
    el('div', {}, el('div', { class: 'eyebrow' }, 'Section V · Mood'), el('h1', { class: 'h1' }, 'Mood ', el('span', {}, '& arcs'))),
    el('section', { class: 'sysgrid' },
      el('article', { class: 'panel glass', 'aria-label': 'Heartbeat' },
        el('div', { class: 'ptitle' }, el('span', {}, 'Heartbeat'), el('span', {}, 'from arousal now')),
        el('div', { class: 'heart' }, el('span', { class: 'pulse', style: `animation-duration:${beat.toFixed(2)}s`, 'aria-hidden': 'true' }),
          el('div', {}, el('div', { class: 'big' }, fmt(m.bpm), el('span', { class: 'unit' }, 'bpm')),
            el('div', { class: 'cap' }, `arousal ${Number(m.arousal).toFixed(2)} · valence ${m.valence === null || m.valence === undefined ? '—' : Number(m.valence).toFixed(2)}`))),
        el('div', { class: 'small' }, '40 + 55 × arousal: a display mapping, not a measured heart.')),
      el('article', { class: 'panel glass', 'aria-label': 'The cube' },
        el('div', { class: 'ptitle' }, el('span', {}, 'Cube'), el('span', {}, 'Lövheim: 5-HT, DA, NE')),
        c ? [cube(c), el('div', { class: 'cap' }, c.at_rest ? 'at rest: the centre of the cube, where no corner is nearer than another'
          : `nearest corner ${c.corner} · intensity ${c.intensity.toFixed(2)} · then ${c.runner_up}`)]
          : el('div', { class: 'small' }, 'This server reports no cube point.')),
      el('article', { class: 'panel glass', 'aria-label': 'Levels' },
        el('div', { class: 'ptitle' }, el('span', {}, 'Levels'), el('span', {}, m.emotion || '')),
        Object.entries(m.levels || {}).map(([k, v]) => bar(k, v)))),
    el('section', { class: 'empty glass', 'aria-label': 'Arcs' }, el('h2', {}, 'Arcs'), el('p', {}, m.arcs.message)));
  timer = setInterval(() => { if (!root.hidden) show(root); else clearInterval(timer); }, 10000);
}
