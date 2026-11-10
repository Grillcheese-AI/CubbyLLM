// Knowledge (order 0055): the hot Wikidata store, searched, read, drawn and queried. Audit view only (the store names
// its dataset and sources). Modes: the Path between two entities; Search a name; open an Entity (optionally as of a
// date); Who has a statement; Resolve a redirected id. Results on the left, the graph in the middle, the entity on the
// right: a click on any entity opens it there, a double-click on a node adds its neighbours to the graph.
import { get, el, fmt, isPublic } from './api.js';
import { graphView } from './graph.js';
import { stats, entityCard, pathList, entityPanel, ms, typeText, icon } from './store_views.js';

let mode = 'path', g = null, current = null, context = false, status = null;
const vals = { from: 'Q42', to: 'Q35', q: '', asof: '', p: 'P31', v: 'Q5' };
const picked = {};   // field -> {id, label, types} once a name or id is resolved
const MODES = [['path', 'Path'], ['search', 'Search'], ['entity', 'Entity'], ['who', 'Who has'], ['resolve', 'Resolve']];
const PRESETS = {
  path: [['Q42 → Q35', { from: 'Q42', to: 'Q35' }], ['Q42 → Q60488168', { from: 'Q42', to: 'Q60488168' }], ['Q937 → Q5', { from: 'Q937', to: 'Q5' }]],
  search: [['Douglas Adams', { q: 'Douglas Adams' }], ['Montréal', { q: 'Montréal' }], ['Paris', { q: 'Paris' }]],
  entity: [['Q42', { q: 'Q42' }], ['Q90 as of 1900', { q: 'Q90', asof: '1900' }], ['Q64', { q: 'Q64' }]],
  who: [['P31 = Q5', { p: 'P31', v: 'Q5' }], ['P50 = Q42', { p: 'P50', v: 'Q42' }], ['P26 = Q42', { p: 'P26', v: 'Q42' }]],
  resolve: [['Q6958747', { q: 'Q6958747' }], ['Q9200127', { q: 'Q9200127' }]],
};
const isId = (s) => /^[QPL][1-9][0-9]*$/.test((s || '').trim());
const $ = (root, id) => root.querySelector('#' + id);

function download(e) {
  const a = el('a', { href: URL.createObjectURL(new Blob([JSON.stringify(e, null, 1)], { type: 'application/json' })), download: `${e.id}.json` });
  document.body.append(a); a.click(); a.remove();
}

// how: 'new' (the graph starts over from id), 'expand' (its neighbours merged in), 'card' (the right panel alone)
async function open(id, root, how = 'card') {
  const box = $(root, 'k-card');
  box.replaceChildren(el('div', { class: 'small' }, `Reading ${id}…`));
  const [e, gr] = await Promise.all([get(`/board/store/entity/${id}${vals.asof ? '?as_of=' + encodeURIComponent(vals.asof) : ''}`),
    how === 'card' ? Promise.resolve(null) : get(`/board/store/graph/${id}?limit=30`)]);
  if (gr && gr.ok && gr.body.nodes) { g.setData(gr.body, how === 'expand'); counts(root); }
  current = e.body;
  box.replaceChildren(...entityPanel(e.body, (x) => open(x, root, 'card'), download));
  if (how === 'card') g.select(id);
}

function counts(root) { const [n, m] = g.count(); $(root, 'k-count').textContent = `${n} nodes · ${m} edges`; }
function timing(root, b) { $(root, 'k-ms').textContent = ms(b); }

// A field's text to an id: an id as it is; a name through search, taken only when it names exactly one entity (the
// candidates are never ordered by fame, so with several the person picks).
async function resolve(field, root) {
  const text = (vals[field] || '').trim();
  if (isId(text)) return text;
  if (picked[field] && picked[field].text === text) return picked[field].id;
  const b = (await get('/board/store/search?q=' + encodeURIComponent(text))).body || {};
  const hits = b.hits || [];
  if (hits.length === 1) { picked[field] = { ...hits[0], text }; return hits[0].id; }
  const list = $(root, 'k-list');
  list.replaceChildren(el('div', { class: 'khd' }, el('span', {}, b.too_many ? `Over ${b.too_many} near names: be more exact.` : `${hits.length} entities named “${text}”: pick one`), el('span', { class: 'kms' }, ms(b))),
    ...hits.map((h) => entityCard(h, () => { vals[field] = h.id; picked[field] = { ...h, text: h.id }; show(root, true); }, false)));
  return null;
}

async function run(root) {
  const list = $(root, 'k-list');
  list.replaceChildren(el('div', { class: 'small' }, 'Asking the store…'));
  if (mode === 'path') {
    const [a, b] = [await resolve('from', root), await resolve('to', root)];
    if (!a || !b) return;
    const r = (await get(`/board/store/path?from=${a}&to=${b}${context ? '&context=1' : ''}`)).body || {};
    list.replaceChildren(...pathList(r, (x) => open(x, root)));
    timing(root, r);
    if (r.steps) {
      g.setData({ nodes: r.nodes, edges: r.edges }, false, [r.steps[0] ? r.steps[0].from : a, ...r.steps.map((s) => s.to)]);
      const last = r.steps[r.steps.length - 1] || {};
      picked.from = { id: a, types: (r.steps[0] || {}).from_types, text: vals.from }; picked.to = { id: b, types: last.to_types, text: vals.to };
      $(root, 'k-panel').replaceWith(panel(root));
      counts(root); open(r.steps.length ? r.steps[r.steps.length - 1].to : a, root);
    }
  } else if (mode === 'search') {
    const b = (await get('/board/store/search?q=' + encodeURIComponent(vals.q))).body || {};
    timing(root, b);
    list.replaceChildren(el('div', { class: 'khd' }, el('span', {}, b.too_many ? `Over ${b.too_many} near names: be more exact.` : `${(b.hits || []).length} entities, in id order (never by fame)`), el('span', { class: 'kms' }, ms(b))),
      ...(b.hits || []).map((h) => entityCard(h, (x) => open(x, root, 'new'), false)));
    if (b.hits && b.hits.length === 1) open(b.hits[0].id, root, 'new');
  } else if (mode === 'entity') {
    if (!isId(vals.q)) { list.replaceChildren(el('div', { class: 'small' }, 'An id: Q42, P31…')); return; }
    list.replaceChildren(el('div', { class: 'small' }, `${vals.q.trim()}${vals.asof ? ' as of ' + vals.asof : ''}: its neighbours are in the graph.`));
    open(vals.q.trim(), root, 'new');
  } else if (mode === 'who') {
    const b = (await get(`/board/store/who?p=${encodeURIComponent(vals.p.trim())}&v=${encodeURIComponent(vals.v.trim())}&limit=200`)).body || {};
    timing(root, b);
    list.replaceChildren(el('div', { class: 'khd' }, el('span', {}, b.error ? b.error : `${fmt(b.total)} have ${b.p_label || b.p} = ${b.v_label || b.v}${b.total > (b.hits || []).length ? ` (first ${b.hits.length})` : ''}`), el('span', { class: 'kms' }, ms(b))),
      ...(b.hits || []).map((h) => entityCard(h, (x) => open(x, root, 'new'), false)));
  } else {
    const b = (await get('/board/store/resolve?id=' + encodeURIComponent(vals.q.trim()))).body || {};
    timing(root, b);
    list.replaceChildren(el('div', { class: 'khd' }, el('span', {}, b.id ? `${vals.q.trim()} leads to` : (b.error || 'no answer')), el('span', { class: 'kms' }, ms(b))),
      b.id ? entityCard({ id: b.id }, (x) => open(x, root, 'new'), true) : null);
    if (b.id) open(b.id, root, 'new');
  }
}

function field(root, key, tag, ph) {
  const p = picked[key], shown = p && (p.id === (vals[key] || '').trim() || p.text === vals[key]);
  return el('label', { class: 'kfield' + (shown ? '' : ' empty') }, el('span', { class: 'ktag' }, tag),
    shown ? icon(p.types) : null,
    el('input', { type: 'text', value: vals[key] || '', placeholder: ph, autocomplete: 'off', 'aria-label': tag,
      oninput: (ev) => { vals[key] = ev.target.value; }, onkeydown: (ev) => { if (ev.key === 'Enter') run(root); } }),
    shown ? el('span', { class: 'ktype' }, typeText(p.types)) : null, shown ? el('span', { class: 'kidb' }, p.id) : null);
}

function panel(root) {
  const f = (k, t, ph) => field(root, k, t, ph);
  const inputs = {
    path: [f('from', 'FROM', 'An id or a name: Q42, Douglas Adams'),
      el('button', { class: 'kswap', title: 'Swap', onclick: () => { [vals.from, vals.to] = [vals.to, vals.from]; show(root, true); } }, '⇄'),
      f('to', 'TO', 'An id or a name: Q35, Denmark')],
    search: [f('q', 'NAME', 'Douglas Adams, Montréal, P31…')],
    entity: [f('q', 'ID', 'Q42'), f('asof', 'AS OF', '2019, 2019-05 (optional)')],
    who: [f('p', 'PROPERTY', 'P31'), f('v', 'VALUE', 'Q5')],
    resolve: [f('q', 'ID', 'An id that may be redirected')],
  }[mode];
  return el('div', { class: 'kpanel glass', id: 'k-panel' },
    el('div', { class: 'ktop' },
      el('div', { class: 'ktabs', role: 'tablist' }, ...MODES.map(([m, n]) => el('button', { class: m === mode ? 'on' : '', role: 'tab', 'aria-selected': String(m === mode), onclick: () => { mode = m; show(root, true); } }, n))),
      el('div', { class: 'kpresets' }, 'Try', ...PRESETS[mode].map(([n, set]) => el('button', { class: 'kpreset', onclick: () => { Object.assign(vals, set); show(root, true).then(() => run(root)); } }, n)))),
    el('div', { class: 'krow' }, ...inputs, el('button', { class: 'kask', onclick: () => run(root) }, 'Ask the store →')));
}

export async function show(root, keep) {
  const b = (keep && status) || (await get('/board/store')).body || {};
  status = b.configured && !b.audit_only ? b : null;
  if (isPublic() || b.audit_only || !b.configured) {
    root.replaceChildren(el('div', { class: 'eyebrow' }, 'Knowledge'), el('div', { class: 'empty glass' }, el('h1', { class: 'h1' }, 'Knowledge'),
      el('p', {}, b.message || b.error || 'unavailable'), isPublic() ? el('p', {}, 'Turn the Public switch off (audit sign-in) to open it.') : null));
    return;
  }
  const tool = (t, title, fn, on) => el('button', { title, class: on ? 'on' : '', onclick: fn }, t);
  root.replaceChildren(
    el('div', { class: 'khero' }, el('div', {}, el('div', { class: 'eyebrow' }, 'Knowledge'), el('h1', { class: 'h1' }, 'Wikidata ', el('span', {}, 'hot store'))), stats(b)),
    panel(root),
    el('div', { class: 'kgrid' },
      el('div', { class: 'kcol glass', id: 'k-list' }, el('div', { class: 'small' }, 'Ask the store: a path, a name, an id, or who has a statement.')),
      el('div', { class: 'kgraphwrap glass' },
        el('div', { class: 'kgbar' }, el('div', { class: 'kgt' }, 'Graph', el('span', { class: 'kcount', id: 'k-count' }, '0 nodes'), el('span', { class: 'kms', id: 'k-ms' })),
          el('div', { class: 'ktools' },
            mode === 'path' ? tool('+1 hop', 'Show a few more neighbours of each node on the path', () => { context = !context; show(root, true).then(() => run(root)); }, context) : null,
            tool('−', 'Zoom out', () => g.zoom(1 / 1.25)), tool('+', 'Zoom in', () => g.zoom(1.25)), tool('Fit', 'Fit the graph', () => g.fit()))),
        el('div', { class: 'kgraph' }, el('canvas', { id: 'k-canvas', 'aria-label': 'Graph of the entities and the statements between them' }),
          el('div', { class: 'khint' }, 'click: read · double-click: expand · drag: move · wheel: zoom'))),
      el('div', { class: 'kcol kright glass', id: 'k-card' }, el('div', { class: 'small' }, 'Click an entity, or a node in the graph, to read it here.'))));
  g = graphView($(root, 'k-canvas'), { onSelect: (id) => open(id, root, 'card'), onExpand: (id) => open(id, root, 'expand') });
  if (!keep) run(root);
  else if (current && current.found) open(current.id, root);
}
