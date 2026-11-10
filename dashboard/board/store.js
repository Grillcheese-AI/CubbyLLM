// Knowledge (order 0055): the hot Wikidata store, searched, read, drawn and queried. Audit view only (the store names
// its dataset and sources). Modes: Search a name; open an Entity (optionally as of a date); Who has a statement;
// the Path between two entities; Resolve a redirected id. A clicked entity opens its card and centres the graph.
import { get, el, fmt, isPublic } from './api.js';
import { graphView } from './graph.js';

let mode = 'search', g = null, current = null, asOf = '';
const MODES = [['search', 'Search'], ['entity', 'Entity'], ['who', 'Who has'], ['path', 'Path'], ['resolve', 'Resolve']];
const isId = (s) => /^[QPL][1-9][0-9]*$/.test((s || '').trim());

function chip(text, cls) { return el('span', { class: 'tag' + (cls ? ' ' + cls : '') }, text); }
function link(id, label, root) {
  return el('button', { class: 'klink', onclick: () => open(id, root), title: id }, label ? `${label} ` : '', el('span', { class: 'kid mono' }, id));
}

// how: 'new' (the graph starts over from id), 'expand' (id's neighbours merged in), 'card' (the card alone)
async function open(id, root, how = 'new') {
  const box = root.querySelector('#k-card');
  box.replaceChildren(el('div', { class: 'small' }, `Reading ${id}…`));
  const [e, gr] = await Promise.all([get(`/board/store/entity/${id}${asOf ? '?as_of=' + encodeURIComponent(asOf) : ''}`),
    how === 'card' ? Promise.resolve(null) : get(`/board/store/graph/${id}?limit=40`)]);
  if (gr && gr.ok && gr.body.nodes) g.setData(gr.body, how === 'expand');
  current = e.body;
  box.replaceChildren(card(e.body, root));
}

function card(e, root) {
  if (!e || !e.found) return el('div', { class: 'small' }, `${e && e.id}: not in the store${e && e.error ? ' (' + e.error + ')' : ''}`);
  const desc = e.names.find((n) => n.kind === 'description' && n.lang === 'en') || e.names.find((n) => n.kind === 'description');
  const aliases = e.names.filter((n) => n.kind === 'alias').map((n) => `${n.text} @${n.lang}`);
  const head = el('div', { class: 'khead' },
    el('h2', { class: 'kh2' }, e.label || e.id, ' ', el('span', { class: 'kid mono' }, e.id)),
    el('div', { class: 'tags' }, chip(e.cold ? 'cold set' : 'hot set', e.cold ? 'int' : ''), e.redirected_from ? chip(`redirected from ${e.redirected_from}`, 'int') : null,
      e.as_of ? chip(`as of ${e.as_of}`) : null, chip(`revision ${e.revision}`), chip(e.modified)),
    desc ? el('p', { class: 'kdesc' }, desc.text) : null,
    aliases.length ? el('div', { class: 'small' }, 'Also: ' + aliases.slice(0, 12).join(' · ')) : null,
    e.sitelinks.length ? el('div', { class: 'small' }, e.sitelinks.map(([s, t]) => `${s.replace(/^[a-z]+:\/\//, '').replace(/\/$/, '')}: ${t}`).join(' · ')) : null,
    el('button', { class: 'pill glass kexp', onclick: () => download(e) }, 'Export JSON'));
  const rows = e.groups.map((gp) => el('div', { class: 'kgroup' },
    el('div', { class: 'kprop' }, gp.label || gp.pid, ' ', el('span', { class: 'kid mono' }, gp.pid)),
    el('div', { class: 'kvals' }, ...gp.statements.map((s) => el('div', { class: 'kval' + (s.rank === 'deprecated' ? ' dep' : '') },
      s.raw && s.raw.entity ? link(s.raw.entity, s.value.replace(/ \(Q\d+\)$|\s\(P\d+\)$|\s\(L\d+\)$/, ''), root) : el('span', {}, s.value),
      s.rank !== 'normal' ? chip(s.rank, s.rank === 'deprecated' ? 'int' : '') : null,
      s.holds ? chip(s.holds.replace('_', ' '), s.holds === 'holds' ? '' : 'int') : null,
      s.references ? el('span', { class: 'small' }, ` ${s.references} ref${s.references > 1 ? 's' : ''}`) : null,
      s.qualifiers.length ? el('div', { class: 'kqual small' }, s.qualifiers.map((q) => `${q.label || q.pid}: ${q.value}`).join(' · ')) : null)))));
  return el('div', {}, head, el('div', { class: 'kgroups' }, rows));
}

function download(e) {
  const a = el('a', { href: URL.createObjectURL(new Blob([JSON.stringify(e, null, 1)], { type: 'application/json' })), download: `${e.id}.json` });
  document.body.append(a); a.click(); a.remove();
}

async function run(root) {
  const v = (id) => (root.querySelector('#' + id) || {}).value || '';
  const list = root.querySelector('#k-list');
  asOf = v('k-asof').trim();
  list.replaceChildren(el('div', { class: 'small' }, 'Asking the store…'));
  if (mode === 'search') {
    const r = await get('/board/store/search?q=' + encodeURIComponent(v('k-q')));
    const b = r.body || {};
    if (b.too_many) { list.replaceChildren(el('div', { class: 'small' }, `Over ${b.too_many} near names: be more exact.`)); return; }
    list.replaceChildren(el('div', { class: 'small' }, `${(b.hits || []).length} entities, in id order (never by fame)`),
      ...(b.hits || []).map((h) => el('div', { class: 'khit' }, link(h.id, h.label, root),
        el('div', { class: 'small' }, [h.description, `${h.how} match (${h.lang})`, h.statements !== undefined ? `${h.statements} statements` : null].filter(Boolean).join(' · ')))));
    if (b.hits && b.hits.length === 1) open(b.hits[0].id, root);
  } else if (mode === 'entity') {
    if (isId(v('k-q'))) { list.replaceChildren(); open(v('k-q').trim(), root); } else list.replaceChildren(el('div', { class: 'small' }, 'An id: Q42, P31…'));
  } else if (mode === 'who') {
    const r = await get(`/board/store/who?p=${encodeURIComponent(v('k-p').trim())}&v=${encodeURIComponent(v('k-v').trim())}&limit=200`);
    const b = r.body || {};
    list.replaceChildren(el('div', { class: 'small' }, b.error ? b.error : `${fmt(b.total)} entities have ${b.p_label || b.p} = ${b.v_label || b.v}${b.total > (b.hits || []).length ? ` (first ${b.hits.length})` : ''}`),
      ...(b.hits || []).map((h) => el('div', { class: 'khit' }, link(h.id, h.label, root), h.rank !== 'normal' ? chip(h.rank, h.rank === 'deprecated' ? 'int' : '') : null)));
  } else if (mode === 'path') {
    const r = await get(`/board/store/path?from=${encodeURIComponent(v('k-q').trim())}&to=${encodeURIComponent(v('k-to').trim())}`);
    const b = r.body || {};
    list.replaceChildren(el('div', { class: 'small' }, b.steps ? `${b.steps.length} step${b.steps.length === 1 ? '' : 's'} (${fmt(b.reads)} entities read)` : (b.message || b.error || 'no answer')),
      ...(b.steps || []).map((s) => el('div', { class: 'khit' }, link(s.from, s.from_label, root), el('span', { class: 'small' }, ` — ${s.label || s.pid} → `), link(s.to, s.to_label, root))));
    if (b.steps && b.steps.length) g.setData({ center: b.from, nodes: [...new Set(b.steps.flatMap((s) => [s.from, s.to]))].map((id) => ({ id, label: (b.steps.find((s) => s.from === id) || {}).from_label || (b.steps.find((s) => s.to === id) || {}).to_label })),
      edges: b.steps.map((s) => ({ from: s.from, to: s.to, pid: s.pid, label: s.label })) }, false);
  } else {
    const r = await get('/board/store/resolve?id=' + encodeURIComponent(v('k-q').trim()));
    const b = r.body || {};
    list.replaceChildren(el('div', { class: 'khit' }, b.id ? el('span', {}, `${v('k-q').trim()} → `) : null, b.id ? link(b.id, null, root) : el('span', { class: 'small' }, b.error || 'no answer')));
  }
}

function bar(root) {
  const f = (id, ph, w) => el('input', { id, type: 'text', placeholder: ph, autocomplete: 'off', class: 'kin', style: w ? `max-width:${w}` : null,
    onkeydown: (ev) => { if (ev.key === 'Enter') run(root); } });
  const inputs = {
    search: [f('k-q', 'A name: Douglas Adams, Montréal, P31…')],
    entity: [f('k-q', 'An id: Q42', '180px'), f('k-asof', 'As of: 2019, 2019-05 (optional)', '240px')],
    who: [f('k-p', 'Property: P31', '140px'), f('k-v', 'Value: Q5', '140px')],
    path: [f('k-q', 'From: Q42', '160px'), f('k-to', 'To: Q60', '160px')],
    resolve: [f('k-q', 'An id that may be redirected', '260px')],
  }[mode];
  return el('div', { class: 'kbar glass' },
    el('div', { class: 'tags' }, ...MODES.map(([m, n]) => el('button', { class: 'pill glass' + (m === mode ? ' on' : ''), onclick: () => { mode = m; show(root); } }, n))),
    el('div', { class: 'krow' }, ...inputs, el('button', { class: 'send', style: 'width:auto;padding:0 18px', onclick: () => run(root) }, 'Ask')));
}

export async function show(root) {
  const r = await get('/board/store');
  const b = r.body || {};
  if (isPublic() || b.audit_only || !b.configured) {
    root.replaceChildren(el('div', { class: 'eyebrow' }, 'Knowledge'), el('div', { class: 'empty glass' }, el('h1', { class: 'h1' }, 'Knowledge'),
      el('p', {}, b.message || b.error || 'unavailable'), isPublic() ? el('p', {}, 'Turn the Public switch off (audit token) to open it.') : null));
    return;
  }
  const c = b.counts || {}, rel = b.release || {};
  root.replaceChildren(
    el('div', { class: 'eyebrow' }, 'Knowledge'),
    el('h1', { class: 'h1' }, 'Wikidata ', el('span', {}, 'hot store')),
    el('div', { class: 'tags kstat' }, chip(`${fmt(c.entities_hot)} hot · ${fmt(c.entities_cold)} cold entities`), chip(`${fmt((c.statements_hot || 0) + (c.statements_cold || 0))} statements`),
      chip(`${fmt(c.redirects)} redirects`), chip(`dump ${b.dump_date || '—'}`), rel.name ? chip(`${rel.name} · ${rel.licence || ''}`) : null,
      chip(`snapshot ${String(b.snapshot || '—').slice(0, 12)}`, 'mono')),
    bar(root),
    el('div', { class: 'kgrid' },
      el('div', { class: 'klist glass', id: 'k-list' }, el('div', { class: 'small' }, 'Search a name, open an id, or ask who has a statement.')),
      el('div', { class: 'kright' },
        el('div', { class: 'kgraph glass' }, el('canvas', { id: 'k-canvas', 'aria-label': 'Graph of the entity and its neighbours' })),
        el('div', { class: 'kcard glass', id: 'k-card' }, el('div', { class: 'small' }, 'Click a node to read it; double-click to expand it.')))));
  g = graphView(root.querySelector('#k-canvas'), { onSelect: (id) => open(id, root, 'card'), onExpand: (id) => open(id, root, 'expand') });
  if (current && current.found) open(current.id, root);
}
