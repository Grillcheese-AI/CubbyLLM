// The Knowledge section's pieces (order 0055's store): the stats strip, entity cards with a type icon, a path's step
// rows, and the entity panel with its key statements. Each takes an `open(id)` to follow a click.
import { el, fmt } from './api.js';

const SVG = 'http://www.w3.org/2000/svg';
export const ICONS = {
  person: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21c0-4 3.6-7 8-7s8 3 8 7',
  place: 'M3 21h18M5 21V10l7-5 7 5v11M9 21v-6h6v6',
  work: 'M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h11',
  thing: 'M12 3l8 4.5v9L12 21l-8-4.5v-9z',
};
const PLACE = /country|state|city|town|village|municipal|commune|region|province|county|island|settlement|capital|district|territory/i;
const WORK = /book|novel|literary|written work|edition|version|album|film|series|song|article|publication|work/i;

export function kind(types) {
  const ids = (types || []).map((t) => t.id), names = (types || []).map((t) => t.label || '').join(' ');
  if (ids.includes('Q5')) return 'person';
  if (PLACE.test(names)) return 'place';
  if (WORK.test(names)) return 'work';
  return 'thing';
}

export function icon(types) {
  const s = document.createElementNS(SVG, 'svg'); s.setAttribute('viewBox', '0 0 24 24');
  const p = document.createElementNS(SVG, 'path'); p.setAttribute('d', ICONS[kind(types)]); s.append(p);
  return el('span', { class: 'kico' }, s);
}

export const typeText = (types) => (types || []).length ? `${types[0].label || types[0].id} (${types[0].id})` : '';

export function stats(b) {
  const c = b.counts || {}, rel = b.release || {};
  const copy = el('button', { class: 'kcopy mono', title: 'Copy the snapshot hash', onclick: () => navigator.clipboard && navigator.clipboard.writeText(b.snapshot || '') },
    `snapshot ${String(b.snapshot || '—').slice(0, 12)}`, ' ⧉');
  return el('div', { class: 'kstats glass' },
    el('span', {}, el('i', { class: 'dot' }), el('b', {}, fmt(c.entities_hot)), 'hot', el('i', { class: 'dot cold' }), fmt(c.entities_cold), ' cold entities'),
    el('span', {}, el('b', {}, fmt((c.statements_hot || 0) + (c.statements_cold || 0))), 'statements'),
    el('span', {}, el('b', {}, fmt(c.redirects)), 'redirects'),
    el('span', { class: 'mono' }, `dump ${b.dump_date || '—'}`, rel.name ? ` · ${rel.name}` : '', rel.licence ? el('span', { class: 'kbadge cc' }, rel.licence) : null),
    el('span', {}, copy));
}

export function entityCard(n, open, on) {
  return el('button', { class: 'kent' + (on ? ' on' : ''), onclick: () => open(n.id) },
    icon(n.types), el('span', { class: 'kt' }, el('span', { class: 'kl' }, n.label || n.id), el('span', { class: 'ks' }, typeText(n.types) || n.description || '')),
    n.cold ? el('span', { class: 'kbadge' }, 'COLD') : null, el('span', { class: 'kidb' }, n.id));
}

export function pathList(b, open) {
  if (!b.steps) return [el('div', { class: 'khd' }, el('span', {}, b.message || b.error || 'no answer'), el('span', { class: 'kms' }, ms(b)))];
  const out = [el('div', { class: 'khd' }, el('span', {}, el('span', { class: 'kidb' }, `${b.steps.length} step${b.steps.length === 1 ? '' : 's'}`), ` (${fmt(b.reads)} entities read)`), el('span', { class: 'kms' }, ms(b)))];
  b.steps.forEach((s, i) => {
    if (i === 0) out.push(entityCard({ id: s.from, label: s.from_label, types: s.from_types }, open, true));
    const q = (s.qualifiers || [])[0];
    out.push(el('div', { class: 'kedge' },
      el('div', { style: 'display:flex;gap:8px;align-items:center' }, `→ ${s.label || s.pid}`, el('span', { class: 'kidb' }, s.pid),
        s.rank ? el('span', { class: 'krank' }, `✓ ${s.rank}${s.references ? ` · ${s.references} ref${s.references > 1 ? 's' : ''}` : ''}`) : null),
      q ? el('div', { class: 'kq' }, el('span', { class: 'kwarn' }, q.label || q.pid), ' ', q.value, s.qualifiers.length > 1 ? ` (+${s.qualifiers.length - 1})` : '') : null));
    out.push(entityCard({ id: s.to, label: s.to_label, types: s.to_types }, open, i === b.steps.length - 1));
  });
  return out;
}

export const ms = (b) => (b && b.ms !== undefined) ? `ϟ ${b.ms} ms` : '';

function claim(g, open) {
  const s = g.statements[0], ent = s.raw && s.raw.entity;
  const more = g.statements.length > 1 ? ` (+${g.statements.length - 1})` : '';
  return el('div', { class: 'kclaim' + (s.rank === 'deprecated' ? ' dep' : '') },
    el('div', { class: 'kp' }, g.pid, ent ? el('button', { class: 'kout', onclick: () => open(ent) }, 'outgoing →') : el('span', {}, s.holds ? s.holds.replace('_', ' ') : '')),
    el('div', { class: 'kpl' }, g.label || g.pid),
    el('div', { class: 'kvl' }, s.value + more, s.qualifiers.length ? ` · ${s.qualifiers.map((x) => `${x.label || x.pid}: ${x.value}`).join(' · ')}` : ''));
}

export function entityPanel(e, open, download) {
  if (!e) return [el('div', { class: 'small' }, 'Click an entity, or a node in the graph, to read it here.')];
  if (!e.found) return [el('div', { class: 'small' }, `${e.id}: not in the store${e.error ? ' (' + e.error + ')' : ''}`)];
  const desc = e.names.find((n) => n.kind === 'description' && n.lang === 'en') || e.names.find((n) => n.kind === 'description');
  const byPid = Object.fromEntries(e.groups.map((g) => [g.pid, g]));
  const key = (e.key || []).map((p) => byPid[p]).filter(Boolean);
  return [
    el('div', { class: 'kcardhd' }, el('span', { class: 'kidb' }, e.id), el('span', { class: 'kbadge' + (e.cold ? '' : ' hot') }, e.cold ? 'COLD SET' : 'HOT STORE'),
      e.redirected_from ? el('span', { class: 'kwarn' }, `from ${e.redirected_from}`) : null, e.as_of ? el('span', { class: 'kbadge cc' }, `as of ${e.as_of}`) : null,
      el('button', { class: 'kcopy', title: 'Export the entity as JSON', onclick: () => download(e) }, '⤓')),
    el('h2', { class: 'kh2' }, e.label || e.id),
    e.types.length ? el('div', { class: 'kinst' }, 'Instance of ', el('b', {}, typeText(e.types))) : null,
    desc ? el('p', { class: 'kdesc' }, desc.text) : null,
    el('div', { class: 'ktiles' },
      el('div', { class: 'ktile' }, el('div', { class: 'kk' }, 'TOTAL STATEMENTS'), el('div', { class: 'kv' }, fmt(e.statements_total))),
      el('div', { class: 'ktile' }, el('div', { class: 'kk' }, 'INDEX RESIDENCY'), el('div', { class: 'kv mono' }, e.residency || '—')),
      el('div', { class: 'ktile' }, el('div', { class: 'kk' }, 'REVISION'), el('div', { class: 'kv mono' }, String(e.revision))),
      el('div', { class: 'ktile' }, el('div', { class: 'kk' }, 'MODIFIED'), el('div', { class: 'kv mono' }, (e.modified || '').slice(0, 10)))),
    el('div', { class: 'ksec' }, 'KEY STATEMENTS & CLAIMS', el('b', {}, `${key.length} PROPERTIES`)),
    ...key.map((g) => claim(g, open)),
  ];
}
