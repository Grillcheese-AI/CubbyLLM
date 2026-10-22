// The board's one door to the server it came from (order 0027). The audience is decided on the server: an audit
// token (opened as /board?token=...) is sent only while the Public switch is off; with no token the page is public.
let token = null;
try { token = new URLSearchParams(location.search).get('token') || sessionStorage.getItem('cubby-board-token'); } catch (e) { token = null; }
try { if (token) sessionStorage.setItem('cubby-board-token', token); } catch (e) { /* private window: keep it in memory */ }
let pub = true;
const listeners = [];

export const hasToken = () => !!token;
export const isPublic = () => pub || !token;
export function setPublic(v) { pub = !!v || !token; listeners.forEach((f) => f(isPublic())); }
export function onAudience(f) { listeners.push(f); }

function headers(extra) {
  const h = Object.assign({}, extra || {});
  if (!isPublic()) h['X-Audit-Token'] = token;
  return h;
}

export async function get(path) {
  const r = await fetch(path, { headers: headers(), cache: 'no-store' });
  let body = null;
  try { body = await r.json(); } catch (e) { body = { error: 'not JSON' }; }
  return { ok: r.ok, status: r.status, body };
}

export async function post(path, data) {
  const r = await fetch(path, { method: 'POST', headers: headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(data) });
  let body = null;
  try { body = await r.json(); } catch (e) { body = { error: 'not JSON' }; }
  return { ok: r.ok, status: r.status, body };
}

export function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}

export const fmt = (n) => (n === null || n === undefined) ? '—' : (typeof n === 'number' ? n.toLocaleString('en') : String(n));
export const reduced = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function bars(v) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'chart'); svg.setAttribute('viewBox', '0 0 160 64'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'trend, last ' + v.length + ' hours');
  const hi = Math.max(1, ...v), gap = 148 / Math.max(v.length - 1, 1);
  v.forEach((y, i) => {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', 'M' + (6 + i * gap).toFixed(1) + ' 58V' + (58 - Math.max(3, (y / hi) * 50)).toFixed(1));
    p.setAttribute('stroke', i === v.length - 1 ? '#F4B860' : '#B9CBD8'); p.setAttribute('stroke-width', '3');
    p.setAttribute('stroke-linecap', 'round'); p.setAttribute('fill', 'none');
    svg.append(p);
  });
  return svg;
}
