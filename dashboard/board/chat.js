// Chat (order 0027): unified POST /turn (use_ask: true) — casual chat goes through CubbyChat, while factual world
// questions go through AskLoop and expose their spans/steps on /board/journey/<id> and /audit/<id>.
import { get, post, el, isPublic, onAudience } from './api.js';

const turns = [];
let sel = null, tab = 'journey', root = null, loaded = false;

async function loadHistory() {
  if (loaded) return;
  loaded = true;
  const r = await get('/board/chat/history');
  if (r.ok && Array.isArray(r.body.turns) && r.body.turns.length) {
    turns.splice(0, turns.length, ...r.body.turns);
    sel = turns[turns.length - 1].rec;
  }
}

function said(rec) {
  if (rec.error) return rec.error;
  if (rec.reply || rec.answer) return rec.reply || rec.answer;
  if (rec.clarify && rec.clarify.question) {
    const opts = (rec.clarify.choices || []).map((c) => c.detail ? `${c.label} (${c.detail})` : c.label).filter(Boolean);
    return opts.length ? `${rec.clarify.question} ${opts.join(' · ')}` : rec.clarify.question;
  }
  return rec.verified ? '' : "I don't know that one yet.";
}

function tags(rec) {
  const t = [];
  if (rec.kind === 'chat' || rec.kind === 'help' || rec.kind === 'learn') {
    t.push(el('span', { class: 'tg ok' }, rec.kind));
    if (rec.emotion) t.push(el('span', { class: 'tg' }, rec.emotion));
  } else if (rec.verified || rec.reply) {
    t.push(el('span', { class: 'tg ok' }, rec.tier ? rec.tier.replace(/_/g, ' ') : 'answered'));
  } else {
    t.push(el('span', { class: 'tg no' }, 'refused'));
  }
  if (rec.wall_s !== undefined && rec.wall_s > 0) t.push(el('span', { class: 'tg' }, `${Math.round(1000 * rec.wall_s)} ms`));
  return t;
}

async function journeyPanel() {
  if (!sel) return el('section', { class: 'panel glass' }, el('div', { class: 'small' }, 'Say hi or ask a question, then pick a reply to see how it was made.'));
  if (!sel.id) {
    const bits = [];
    if (sel.kind) bits.push(`Mode: ${sel.kind}`);
    if (sel.emotion) bits.push(`Mood: ${sel.emotion}`);
    if (sel.register) bits.push(`Register: ${sel.register}`);
    return el('section', { class: 'panel glass', 'aria-label': 'Chat turn summary' },
      el('div', { class: 'ptitle' }, 'Casual chat turn'),
      el('div', { class: 'small' }, bits.length ? bits.join(' · ') : 'Handled by CubbyChat (no world lookup needed).'));
  }
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
  const input = el('input', { id: 'ask-input', type: 'text', placeholder: 'Talk to Cubby or ask about its worlds…', autocomplete: 'off' });
  const send = async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    const r = await post('/turn', { text, use_ask: true });
    const rec = r.ok ? r.body : { error: r.body.error || 'error' };
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
  const clearChat = async () => {
    const r = await post('/board/chat/clear', {});
    if (r.ok) { turns.length = 0; sel = null; render(); }
  };
  const header = el('div', { style: 'display:flex;align-items:baseline;justify-content:space-between;gap:12px' },
    el('div', {}, el('div', { class: 'eyebrow' }, 'Section II · Chat'), el('h1', { class: 'h1' }, 'Talk to Cubby ', el('span', {}, 'and see why'))),
    turns.length ? el('button', { class: 'pill', type: 'button', onclick: clearChat }, 'New chat') : null);
  root.replaceChildren(
    header,
    el('div', { class: 'chat' },
      el('section', { class: 'convo glass', 'aria-label': 'Conversation' }, msgs,
        el('form', { class: 'ask', onsubmit: send }, el('label', { for: 'ask-input', class: 'sr' }, 'Ask Cubby'), input,
          el('button', { class: 'send', type: 'submit', 'aria-label': 'Send' }, '→'))),
      el('aside', { class: 'side' }, tabs, panel)));
  msgs.scrollTop = msgs.scrollHeight;
}

onAudience(() => { if (root && !root.hidden) { loaded = false; loadHistory().then(render); } });

export async function show(node) { root = node; await loadHistory(); await render(); }
