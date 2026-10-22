// Chat (order 0027): ask through the server's own POST /ask, then see how the selected answer was made -- the question's
// spans (entity, relation, time) and its steps from /board/journey/<id> -- and, in the audit view only, its whole
// record from /audit/<id>. In public mode the audit tab is disabled and the server sends no source anyway.
import { get, post, el, isPublic, onAudience } from './api.js';

const turns = [];
let sel = null, tab = 'journey', root = null;

function said(rec) {
  if (rec.error) return rec.error;
  return rec.reply || rec.answer || (rec.verified ? '' : "I don't know that one yet.");
}

function tags(rec) {
  const t = [];
  if (rec.verified || rec.reply) t.push(el('span', { class: 'tg ok' }, rec.tier ? rec.tier.replace(/_/g, ' ') : 'answered'));
  else t.push(el('span', { class: 'tg no' }, 'refused'));
  if (rec.wall_s !== undefined) t.push(el('span', { class: 'tg' }, `${Math.round(1000 * rec.wall_s)} ms`));
  return t;
}

async function journeyPanel() {
  if (!sel || !sel.id) return el('section', { class: 'panel glass' }, el('div', { class: 'small' }, 'Ask a question, then pick an answer to see how it was made.'));
  const r = await get('/board/journey/' + encodeURIComponent(sel.id));
  if (!r.ok) return el('section', { class: 'panel glass' }, el('div', { class: 'small' }, r.body.error || 'no journey'));
  const j = r.body;
  return el('section', { class: 'panel glass', 'aria-label': 'How the selected answer was made' },
    el('div', { class: 'ptitle' }, 'The question, read'),
    el('div', { class: 'tokens' }, j.tokens.map((k) => el('span', { class: 'tok mono ' + (k.role || '') }, k.text))),
    el('div', { class: 'keyrow' }, el('span', {}, '● entity'), el('span', {}, '◆ relation'), el('span', {}, '▲ time')),
    el('div', { class: 'ptitle' }, 'Its journey'),
    el('div', {}, j.steps.map((s, i) => el('div', { class: 'step' + (s.stop ? ' stop' : '') },
      el('span', { class: 'sn' }, i + 1),
      el('div', {}, el('div', { class: 'st' }, s.title),
        el('div', { class: 'sd' }, Object.entries(s.counts).map(([k, n]) => `${n} ${k}`).join(' · ')),
        s.facts.length ? el('div', { class: 'sd' }, s.facts.join('; ')) : null,
        s.sources && s.sources.length ? el('div', { class: 'sd' }, 'from: ' + s.sources.join(', ')) : null),
      el('span', { class: 'sms mono' }, s.ms === null || s.ms === undefined ? '' : `${s.ms} ms`)))),
    el('div', { class: 'small' }, `${j.verified ? 'Answered' : 'Not answered'} in ${j.wall_s === null ? '—' : Math.round(1000 * j.wall_s) + ' ms'}` +
      (j.reason ? ` · ${j.reason.replace(/_/g, ' ')}` : '')));
}

async function auditPanel() {
  if (isPublic()) return el('section', { class: 'panel glass' }, el('div', { class: 'small' }, 'The audit trail is for the owner and auditors; public mode hides it.'));
  if (!sel || !sel.id) return el('section', { class: 'panel glass' }, el('div', { class: 'small' }, 'Pick an answer first.'));
  const r = await get('/audit/' + encodeURIComponent(sel.id));
  return el('section', { class: 'panel glass', 'aria-label': 'Audit trail' },
    el('div', { class: 'ptitle' }, `Record ${sel.id} · internal view`),
    el('pre', { class: 'audit mono' }, JSON.stringify(r.body, null, 1)),
    el('div', { class: 'small' }, 'The audit view names every source. Replies never do.'));
}

async function render() {
  const input = el('input', { id: 'ask-input', type: 'text', placeholder: 'Ask Cubby about its worlds…', autocomplete: 'off' });
  const send = async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    const r = await post('/ask', { text });
    const rec = r.ok ? r.body : { error: r.status === 503 ? 'The ask loop is not mounted here (start the server with --ask).' : (r.body.error || 'error') };
    turns.push({ q: text, rec });
    sel = rec.error ? sel : rec;
    render();
  };
  const msgs = el('div', { class: 'msgs', 'aria-live': 'polite' }, turns.map((t) => [
    el('div', { class: 'me' }, t.q),
    el('button', { class: 'cubby' + (sel === t.rec ? ' sel' : ''), 'aria-pressed': String(sel === t.rec),
      onclick: () => { if (!t.rec.error) { sel = t.rec; render(); } } },
      el('span', {}, said(t.rec)), el('span', { class: 'tags' }, t.rec.error ? [] : tags(t.rec)))]));
  const tabs = el('div', { class: 'tabs glass', role: 'tablist' },
    el('button', { role: 'tab', 'aria-selected': String(tab === 'journey'), onclick: () => { tab = 'journey'; render(); } }, 'Prompt journey'),
    el('button', { role: 'tab', 'aria-selected': String(tab === 'audit'), disabled: isPublic(), onclick: () => { tab = 'audit'; render(); } }, 'Audit trail'));
  if (isPublic() && tab === 'audit') tab = 'journey';
  const panel = tab === 'journey' ? await journeyPanel() : await auditPanel();
  root.replaceChildren(
    el('div', {}, el('div', { class: 'eyebrow' }, 'Section II · Chat'), el('h1', { class: 'h1' }, 'Talk to Cubby ', el('span', {}, 'and see why'))),
    el('div', { class: 'chat' },
      el('section', { class: 'convo glass', 'aria-label': 'Conversation' }, msgs,
        el('form', { class: 'ask', onsubmit: send }, el('label', { for: 'ask-input', class: 'sr' }, 'Ask Cubby'), input,
          el('button', { class: 'send', type: 'submit', 'aria-label': 'Send' }, '→'))),
      el('aside', { class: 'side' }, tabs, panel)));
}

onAudience(() => { if (root && !root.hidden) render(); });

export async function show(node) { root = node; await render(); }
