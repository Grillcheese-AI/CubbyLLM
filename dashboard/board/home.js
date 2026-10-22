// Home: the seven sections on a CSS-3D turntable (order 0027). Only the front object plays its animation; the rest
// show their still, and with prefers-reduced-motion nothing moves. Cards come from /board/home: a section whose order
// is not done shows "not built yet (order NNNN)" and no numbers.
import { get, el, fmt, bars, reduced } from './api.js';

export const SECTIONS = [
  { id: 'watch', line: 'Where the world is heating up, and which warnings are live right now.', views: ['Map', 'Tensions', 'Warnings', 'News'] },
  { id: 'chat', line: 'Talk to Cubby and watch each question travel: route, world, facts, check, reply.', views: ['Talk', 'Prompt journey', 'Audit trail', 'History'] },
  { id: 'worlds', line: 'Explore what Cubby knows and how things relate, as of any date you pick.', views: ['Explore', 'Relations', 'Timeline', 'Sources'] },
  { id: 'forecast', line: 'What might happen next, branch by branch, and how often Cubby was right.', views: ['Questions', 'Tree', 'Track record', 'Analogues'] },
  { id: 'mood', line: 'How Cubby feels right now, as a heartbeat, and the emotional arc of a book or a chat.', views: ['Heartbeat', 'Cube', 'Arcs', 'Datasets'] },
  { id: 'system', line: 'The order queue, the news poller, the copy-or-refuse rates and training runs at a glance.', views: ['Health', 'Training', 'Sources', 'Orders'] },
  { id: 'agents', line: 'Launch an agent on a goal, with only the tools and budget you allow. Planned.', views: ['Launch', 'Running', 'Tools', 'Log'] }
];
const NUMS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];
const N = SECTIONS.length, R = 400;
const mod = (a, n) => ((a % n) + n) % n;
let turn = 1, data = null, root = null;

function card(c) {
  return el('article', { class: 'card glass' },
    el('div', {}, el('div', { class: 'ct' }, c.t), el('div', { class: 'cs' }, c.s)),
    el('div', { class: 'cbody' },
      el('div', {}, el('div', { class: 'big' }, fmt(c.big), el('span', { class: 'unit' }, c.unit || '')), el('div', { class: 'cap' }, c.cap || '')),
      Array.isArray(c.trend) && c.trend.length ? bars(c.trend) : null));
}

function render() {
  const f = mod(turn, N), F = SECTIONS[f], meta = (data && data.sections[f]) || { name: F.id, dim: '', built: false };
  const still = reduced();
  const items = SECTIONS.map((s, i) => {
    const a = (i - turn) * (360 / N), k = Math.pow((1 + Math.cos(a * Math.PI / 180)) / 2, 1.6);
    const m = (data && data.sections[i]) || { name: s.id };
    const front = i === f;
    const b = el('button', { class: 'item' + (front ? ' front' : '') + (m.built ? '' : ' wait'),
      style: `transform: rotateY(${a.toFixed(2)}deg) translateZ(${R}px) rotateY(${(-a).toFixed(2)}deg); opacity: ${(0.28 + 0.72 * k).toFixed(3)}; filter: blur(${((1 - k) * 1.6).toFixed(2)}px)`,
      'aria-label': front ? `${m.name} (in front)` : `Turn to ${m.name}`, tabindex: front ? '0' : '-1',
      onclick: () => { goTo(i); } },
      el('img', { src: `/board/assets/${s.id}${front && !still ? '' : '_still'}.webp`, alt: '', width: 240, height: 240 }),
      el('span', { class: 'iname' }, m.name),
      el('span', { class: 'istat' }, m.built ? 'live' : (m.message || '')));
    return b;
  });
  const cards = meta.built ? meta.cards.map(card)
    : [el('article', { class: 'card glass' }, el('div', { class: 'ct' }, meta.name), el('div', { class: 'cap' }, meta.message || 'loading…'))];
  root.replaceChildren(
    el('div', { class: 'home' },
      el('nav', { class: 'subs', 'aria-label': 'Views in this section' },
        F.views.map((v, i) => el('a', { class: 'sub' + (i === 0 ? ' on' : ''), href: '#' + F.id },
          el('span', { class: 'n' }, i === 0 ? '0' : NUMS[i - 1]), el('span', { class: 'l' }, v)))),
      el('div', { class: 'mid' },
        el('div', { class: 'eyebrow' }, `Section ${NUMS[f]} · ${f + 1} of ${N}`),
        el('h1', { class: 'title' }, el('span', { class: 't1' }, meta.name), el('span', { class: 't2' }, meta.dim || '')),
        el('p', { class: 'lede' }, F.line),
        el('a', { class: 'open', href: '#' + F.id }, meta.built ? `Open ${meta.name}` : `See ${meta.name}`),
        el('div', { class: 'stagewrap', onkeydown: keys },
          el('div', { class: 'scene' }, el('div', { class: 'ring' }, el('div', { class: 'disc' }), el('div', { class: 'ticks' }), items))),
        el('div', { class: 'controls' },
          el('button', { class: 'round', 'aria-label': 'Turn left', onclick: () => { turn -= 1; render(); } }, '‹'),
          el('span', { class: 'counter', 'aria-live': 'polite' }, `${NUMS[f]} / VII`),
          el('button', { class: 'round', 'aria-label': 'Turn right', onclick: () => { turn += 1; render(); } }, '›'))),
      el('aside', { class: 'notes', 'aria-label': 'Notes' },
        el('div', { class: 'note glass' }, data && data.audience === 'audit' ? 'Audit view' : 'Public view',
          el('small', {}, data && data.audience === 'audit' ? 'Sources are named here; replies never name them.' : 'The server sends only what any user may see.')))),
    el('section', { class: 'cards', 'aria-label': 'At a glance' }, cards));
}

function keys(e) {
  if (e.key === 'ArrowLeft') { turn -= 1; render(); e.preventDefault(); }
  if (e.key === 'ArrowRight') { turn += 1; render(); e.preventDefault(); }
}

function goTo(i) {
  const cur = mod(turn, N);
  let d = mod(i - cur, N);
  if (d > N / 2) d -= N;
  turn += d;
  render();
}

export async function show(node) {
  root = node;
  render();
  const r = await get('/board/home');
  data = r.ok ? r.body : { sections: SECTIONS.map((s) => ({ name: s.id, built: false, message: r.body.error || 'unavailable' })) };
  render();
}
