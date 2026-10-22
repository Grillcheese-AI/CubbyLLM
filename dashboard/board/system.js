// System (order 0027): the order queue, the news poller, the copy-or-refuse rates and training runs, from /board/system.
// The feeds' names come only in the audit view (0016): the public one gets counts.
import { get, el, fmt } from './api.js';

const pct = (x) => (x === null || x === undefined) ? '—' : (100 * x).toFixed(1) + '%';

function tile(t, s, big, unit, cap) {
  return el('article', { class: 'tile glass' }, el('div', {}, el('div', { class: 'ct' }, t), el('div', { class: 'cs' }, s)),
    el('div', {}, el('div', { class: 'big' }, fmt(big), el('span', { class: 'unit' }, unit)), el('div', { class: 'cap' }, cap)));
}

function table(head, rows) {
  return el('div', { class: 'tablebox' }, el('table', {}, el('thead', {}, el('tr', {}, head.map((h) => el('th', { scope: 'col' }, h)))),
    el('tbody', {}, rows.map((r) => el('tr', {}, r.map((c) => (c instanceof Node ? el('td', {}, c) : el('td', { class: typeof c === 'number' ? 'num mono' : '' }, fmt(c)))))))));
}

export async function show(root) {
  const r = await get('/board/system');
  if (!r.ok) { root.replaceChildren(el('div', { class: 'empty glass' }, r.body.error || 'unavailable')); return; }
  const d = r.body, q = d.orders.counts, p = d.poller, cr = d.copy_or_refuse[0];
  const groups = ['in progress', 'review', 'open', 'proposed', 'blocked', 'done'];
  const orders = el('article', { class: 'panel glass', 'aria-label': 'Order queue' },
    el('div', { class: 'ptitle' }, el('span', {}, 'Orders'), el('span', {}, `${d.orders.list.length} in all`)),
    groups.filter((g) => g !== 'done').map((g) => {
      const os = d.orders.list.filter((o) => o.status === g);
      return os.length ? [el('div', { class: 'grp' }, g), os.map((o) => el('div', { class: 'ord' }, el('span', { class: 'mono' }, o.no), el('span', {}, o.title)))] : null;
    }));
  const feeds = p.feeds
    ? table(['Feed', 'Kind', 'Every', 'Status', 'Last'], p.feeds.map((f) => [
      el('span', {}, f.id, f.internal ? el('span', { class: 'tag int' }, ' internal') : null), f.kind || '', `${f.every_h} h`,
      f.error ? `${f.status} · ${f.error}` : f.status, f.last ? f.last.slice(5, 16).replace('T', ' ') : '—']))
    : el('div', { class: 'small' }, `${fmt(p.active)} feeds polled, ${fmt(p.held)} held. Their names are in the audit view.`);
  root.replaceChildren(
    el('div', {}, el('div', { class: 'eyebrow' }, 'Section VI · System'), el('h1', { class: 'h1' }, 'System ', el('span', {}, 'health'))),
    el('section', { class: 'tiles', 'aria-label': 'At a glance' },
      tile('Orders', 'The queue', q.open || 0, 'open', `${q['in progress'] || 0} in progress · ${q.review || 0} in review · ${q.proposed || 0} proposed`),
      tile('News poller', 'hourly · names itself CubbyLLM/0.1', p.up, `of ${fmt(p.active)} feeds`,
        `last poll ${p.last_poll_ago || 'never'} · ${fmt(p.items_today)} items today`),
      tile('Copy or refuse', cr ? cr.what : 'not measured yet', cr ? (100 * cr.present_wrong).toFixed(1) : null, '% wrong',
        cr ? `${pct(cr.present_refused)} refused · ${pct(cr.absent_answered)} of absent answered` : 'no log yet'),
      tile('GPU', 'load now', d.gpu, '', d.gpu === null ? 'not reported by this backend' : '')),
    el('section', { class: 'sysgrid' },
      el('article', { class: 'panel glass', 'aria-label': 'Training runs' }, el('div', { class: 'ptitle' }, el('span', {}, 'Training runs'), el('span', {}, d.training.length)),
        d.training.length ? table(['Run', 'Step', 'Loss', 'Last synced line'], d.training.map((t) => [t.run, t.step, t.loss === null || t.loss === undefined ? '—' : Number(t.loss).toFixed(3),
          el('span', {}, t.as_of || t.updated, t.stale ? el('span', { class: 'tag int' }, ' stale') : null)]))
          : el('div', { class: 'small' }, 'No run logs here (CUBBY_RUNS names their folder).'),
        el('div', { class: 'ptitle' }, el('span', {}, 'Copy or refuse'), el('span', {}, 'test documents')),
        table(['Measured', 'Exact', 'Wrong', 'Refused', 'Absent answered'], d.copy_or_refuse.map((c) => [c.what, pct(c.exact), pct(c.present_wrong), pct(c.present_refused), pct(c.absent_answered)]))),
      el('article', { class: 'panel glass', 'aria-label': 'Sources' }, el('div', { class: 'ptitle' }, el('span', {}, 'Sources'), el('span', {}, d.audience)), feeds),
      orders),
    el('section', { class: 'agents glass', id: 'agents', 'aria-label': 'Agents (planned)' },
      el('div', { class: 'eyebrow' }, 'Section VII · Agents'), el('h2', {}, 'Agents, planned'),
      el('p', { class: 'small' }, 'An agent will take a goal and work on it with only the tools you tick, within a budget, every step in the audit trail. Coming in a later order: nothing runs from here.'),
      el('button', { disabled: true }, 'Launch (later order)')));
}
