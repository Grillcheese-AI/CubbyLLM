// Home: every section's numbers at once (order 0027; Nick, 2026-11-10: stats instead of the 3-D turntable). A strip of
// headline numbers (each live section's first card, and in the audit view the knowledge store's counts), then a panel
// per live section with all its cards, then the sections still waiting on their order. Cards come from /board/home: a
// section whose order is not done shows "not built yet (order NNNN)" and no numbers.
import { get, el, fmt, bars, isPublic } from './api.js';

export const SECTIONS = [
  { id: 'watch', line: 'Where the world is heating up, and which warnings are live right now.' },
  { id: 'chat', line: 'Talk to Cubby and watch each question travel: route, world, facts, check, reply.' },
  { id: 'worlds', line: 'Explore what Cubby knows and how things relate, as of any date you pick.' },
  { id: 'forecast', line: 'What might happen next, branch by branch, and how often Cubby was right.' },
  { id: 'mood', line: 'How Cubby feels right now, as a heartbeat, and the emotional arc of a book or a chat.' },
  { id: 'system', line: 'The order queue, the news poller, the copy-or-refuse rates and training runs at a glance.' },
  { id: 'agents', line: 'Launch an agent on a goal, with only the tools and budget you allow. Planned.' },
];
const COMPACT = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 });
const num = (n) => (typeof n === 'number' && Math.abs(n) >= 1e6) ? COMPACT.format(n) : fmt(n);

function kpi(label, value, unit, cap, href) {
  return el('a', { class: 'kpi glass', href: href || null },
    el('div', { class: 'k' }, label),
    el('div', { class: 'v' }, num(value), unit ? el('span', { class: 'u' }, unit) : null),
    cap ? el('div', { class: 'c' }, cap) : null);
}

function tile(c) {
  return el('div', { class: 'stat' },
    el('div', { class: 'st' }, c.t, el('span', {}, c.s)),
    el('div', { class: 'sb' },
      el('div', {}, el('div', { class: 'sv' }, num(c.big), el('span', { class: 'u' }, c.unit || '')), el('div', { class: 'cap' }, c.cap || '')),
      Array.isArray(c.trend) && c.trend.length ? bars(c.trend) : null));
}

function panel(s, line) {
  return el('section', { class: 'sec glass', 'aria-label': s.name },
    el('header', { class: 'sechd' },
      el('div', {}, el('h2', {}, s.name, ' ', el('span', { class: 'dim' }, s.dim || '')), el('p', {}, line)),
      el('span', { class: 'chip ok' }, '● live'),
      el('a', { class: 'secopen', href: '#' + s.id }, 'Open →')),
    el('div', { class: 'stats' }, s.cards.map(tile)));
}

function render(root, data, store) {
  const secs = (data && data.sections) || [];
  const live = secs.filter((s) => s.built), waiting = secs.filter((s) => !s.built);
  const line = (s) => (SECTIONS.find((x) => x.id === s.id) || {}).line || '';
  const heads = live.map((s) => s.cards[0] && kpi(`${s.name} · ${s.cards[0].t}`, s.cards[0].big, s.cards[0].unit, s.cards[0].cap, '#' + s.id));
  if (store) {
    const c = store.counts || {};
    heads.push(kpi('Knowledge · entities', (c.entities_hot || 0) + (c.entities_cold || 0), '',
      `${num((c.statements_hot || 0) + (c.statements_cold || 0))} statements · dump ${(store.dump_date || '').slice(0, 10)}`, '#store'));
  }
  root.replaceChildren(el('div', { class: 'homev' },
    el('div', { class: 'hhero' },
      el('div', {}, el('div', { class: 'eyebrow' }, 'Home'), el('h1', { class: 'h1' }, 'Cubby ', el('span', {}, 'at a glance')),
        el('p', { class: 'lede' }, 'Every section’s numbers, live. A section whose order is not done shows none.')),
      el('div', { class: 'note glass' }, data && data.audience === 'audit' ? 'Audit view' : 'Public view',
        el('small', {}, data && data.audience === 'audit' ? 'Sources are named here; replies never name them.' : 'The server sends only what any user may see.'))),
    data ? null : el('div', { class: 'small' }, 'Reading the sections…'),
    heads.length ? el('div', { class: 'kpis', 'aria-label': 'Headline numbers' }, heads) : null,
    live.length ? el('div', { class: 'secgrid' }, live.map((s) => panel(s, line(s)))) : null,
    waiting.length ? el('section', { class: 'waiting glass', 'aria-label': 'On the way' },
      el('h2', {}, 'On the way'),
      el('ul', {}, waiting.map((s) => el('li', {}, el('a', { href: '#' + s.id }, s.name), el('span', {}, line(s)),
        el('span', { class: 'tag' }, s.message || 'not built yet'))))) : null));
}

export async function show(node) {
  render(node, null, null);
  const [r, st] = await Promise.all([get('/board/home'), isPublic() ? Promise.resolve(null) : get('/board/store')]);
  const data = r.ok ? r.body : { sections: SECTIONS.map((s) => ({ id: s.id, name: s.id, built: false, message: (r.body || {}).error || 'unavailable' })) };
  const store = st && st.body && st.body.configured && !st.body.audit_only ? st.body : null;
  render(node, data, store);
}
