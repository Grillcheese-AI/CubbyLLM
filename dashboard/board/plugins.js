// Plugins (order 0028): what is installed, its status and permissions, "which plugin would wake?", the selected
// plugin's PLUGIN.md, and -- in the audit view only -- Approve / Pin / Disable. Panels of loaded plugins as cards.
import { get, post, el, fmt, isPublic } from './api.js';

let sel = null, last = null, review = null;
const STATUS_CLASS = (s) => s.startsWith('blocked') || s.startsWith('refused') ? 'tag int' : (s === 'needs approval' ? 'tag int' : 'tag');

function perms(p) {
  const n = (p.permissions && p.permissions.network) || [];
  return [n.length ? `network: ${n.join(', ')}` : 'no network', p.permissions && p.permissions.write ? 'writes its folder' : 'writes nothing',
    p.permissions && p.permissions.gpu ? 'uses the GPU' : 'no GPU'].join(' · ');
}

async function showReview(name, root) {
  const r = await get('/board/plugins/review?name=' + encodeURIComponent(name));
  review = r.body && r.body.review ? r.body.review : null;
  sel = name;
  last = review ? null : `${name}: no review (audit view only)`;
  await show(root);
}

async function act(name, what, root) {
  const r = await post(`/board/plugins/${encodeURIComponent(name)}/${what}`, what === 'approve' ? { digest: review && review.digest } : {});
  if (what === 'approve') review = null;
  last = r.ok ? `${name}: ${r.body.status}` : `${name}: ${r.body.error || r.status}${r.body.why ? ' -- ' + r.body.why[0] : ''}`;
  await show(root);
}

function card(p, root) {
  const audit = !isPublic();
  return el('article', { class: 'tile glass' },
    el('div', { class: 'ptitle' }, el('button', { class: 'pill glass', onclick: () => { sel = p.name; show(root); }, 'aria-label': `Show ${p.name}` }, p.name),
      el('span', { class: STATUS_CLASS(p.status) }, p.status)),
    el('div', { class: 'small' }, `${p.version} · ${p.kinds.join(', ')}${p.audience === 'internal' ? ' · internal' : ''}`),
    el('div', {}, p.description),
    el('div', { class: 'small' }, perms(p)),
    audit ? el('div', { class: 'tags' },
      ['needs approval'].includes(p.status) || p.status.startsWith('blocked') ? el('button', { class: 'send', style: 'width:auto;padding:0 14px', onclick: () => showReview(p.name, root) }, 'Review to approve') : null,
      el('button', { class: 'pill glass', onclick: () => act(p.name, p.status === 'pinned' ? 'unpin' : 'pin', root) }, p.status === 'pinned' ? 'Unpin' : 'Pin'),
      el('button', { class: 'pill glass', onclick: () => act(p.name, p.status === 'disabled' ? 'enable' : 'disable', root) }, p.status === 'disabled' ? 'Enable' : 'Disable'))
      : null);
}

async function tryRouter(q, box) {
  const r = await get('/board/plugins/route?q=' + encodeURIComponent(q));
  const b = r.body || {};
  box.replaceChildren(
    ...Object.entries(b.scores || {}).map(([n, s]) => el('div', { class: 'step' },
      el('span', { class: 'sn' }, s >= b.tau ? '●' : '○'), el('div', {}, el('div', { class: 'st' }, n), el('div', { class: 'sd' }, s >= b.tau ? 'would wake' : 'stays asleep')),
      el('span', { class: 'sms mono' }, s.toFixed(3)))),
    el('div', { class: 'small' }, (b.would_wake || []).length ? `Wakes: ${b.would_wake.join(', ')} (line ${b.tau})` : `Nothing wakes (line ${b.tau}): Cubby answers from what is mounted, or says it doesn't know.`),
    Object.keys(b.declined || {}).length ? el('div', { class: 'small' }, 'Over the line but declined: ' + Object.entries(b.declined).map(([n, w]) => `${n} (${w})`).join('; ')) : null);
}

export async function show(root) {
  const r = await get('/board/plugins');
  const b = r.body || {};
  if (!r.ok || !b.built) { root.replaceChildren(el('div', { class: 'empty glass' }, el('h1', { class: 'h1' }, 'Plugins'), el('p', {}, b.message || b.error || 'unavailable'))); return; }
  const doc = sel ? (await get('/board/plugins/doc?name=' + encodeURIComponent(sel))).body.doc : null;
  const q = el('input', { id: 'route-q', type: 'text', placeholder: 'A question to try…', autocomplete: 'off' });
  const scores = el('div', {});
  const loaded = b.plugins.filter((p) => ['loaded', 'pinned'].includes(p.status)).length;
  root.replaceChildren(...[
    el('div', {}, el('div', { class: 'eyebrow' }, 'System · Plugins'), el('h1', { class: 'h1' }, 'Plugins ', el('span', {}, 'on demand')),
      el('p', { class: 'small' }, 'A plugin is a folder with a PLUGIN.md. Cubby keeps one line about each and wakes a plugin only when a question needs it.')),
    last ? el('div', { class: 'note glass', role: 'status' }, last) : null,
    el('section', { class: 'tiles', 'aria-label': 'At a glance' },
      el('article', { class: 'tile glass' }, el('div', { class: 'ct' }, 'Loaded now'), el('div', { class: 'big' }, fmt(loaded), el('span', { class: 'unit' }, `of ${b.plugins.length}`))),
      el('article', { class: 'tile glass' }, el('div', { class: 'ct' }, 'Wake line'), el('div', { class: 'big' }, fmt(b.tau)), el('div', { class: 'cap' }, 'a question at or above it loads the plugin'))),
    el('div', { class: 'sysgrid' },
      el('section', { class: 'tiles', style: 'grid-template-columns:minmax(0,1fr)', 'aria-label': 'Installed plugins' }, b.plugins.map((p) => card(p, root))),
      el('section', { class: 'panel glass', 'aria-label': 'Try the router' }, el('div', { class: 'ptitle' }, 'Which plugin would wake?'),
        el('form', { class: 'ask', onsubmit: (e) => { e.preventDefault(); tryRouter(q.value, scores); } }, el('label', { for: 'route-q', class: 'sr' }, 'A question'), q,
          el('button', { class: 'send', type: 'submit', 'aria-label': 'Try' }, '→')), scores),
      review && review.name === sel
        ? el('section', { class: 'panel glass', 'aria-label': 'Review before approving' },
          el('div', { class: 'ptitle' }, `${sel} · review before approving`),
          el('div', { class: 'small' }, `permissions: ${JSON.stringify(review.permissions)}`),
          review.check.length ? el('div', { class: 'small' }, 'fails review: ' + review.check.join('; ')) : null,
          el('pre', { class: 'audit mono' }, review.files.map((f) => `--- ${f.status} ${f.file}\n` +
            (f.status === 'new' || f.status === 'changed' ? (f.diff || f.text || `(binary, ${f.bytes} bytes)`) : '')).join('\n')),
          el('button', { class: 'send', style: 'width:auto;padding:0 14px', disabled: review.check.length > 0,
            onclick: () => act(sel, 'approve', root) }, 'Approve these files'))
        : el('section', { class: 'panel glass', 'aria-label': 'Selected plugin' }, el('div', { class: 'ptitle' }, sel ? `${sel} · PLUGIN.md` : 'PLUGIN.md'),
          el('pre', { class: 'audit mono' }, doc || 'Pick a plugin to read its PLUGIN.md.'))),
    Object.entries(b.panels || {}).map(([n, pn]) => el('article', { class: 'card glass', 'aria-label': pn.title },
      el('div', { class: 'ct' }, pn.title), el('div', { class: 'cs' }, pn.subtitle || n),
      pn.data ? el('div', { class: 'big' }, fmt(pn.data[pn.big]), el('span', { class: 'unit' }, pn.unit || '')) : el('div', { class: 'cap' }, pn.error || 'no data yet'),
      pn.data && pn.list ? el('div', { class: 'small' }, (pn.data[pn.list] || []).map((it) => el('div', {}, (pn.item || []).map((k) => it[k]).join(' · ')))) : null))].flat().filter(Boolean));
}
