// The board's router (order 0027): one page, a view per section, chosen by the URL's hash.
import { get, el, hasToken, isPublic, setPublic, onAudience } from './api.js';
import * as home from './home.js';
import * as chat from './chat.js';
import * as system from './system.js';
import * as plugins from './plugins.js';
import * as mood from './mood.js';

const NAMES = { home: 'Home', watch: 'World Watch', chat: 'Chat', worlds: 'Worlds', forecast: 'Forecast', mood: 'Mood', system: 'System', plugins: 'Plugins', agents: 'Agents' };
const views = { home: ['v-home', home.show], chat: ['v-chat', chat.show], system: ['v-system', system.show], plugins: ['v-plugins', plugins.show], mood: ['v-mood', mood.show] };

async function emptyView(node, id) {
  const r = await get('/board/' + id);
  const b = r.body || {};
  node.replaceChildren(el('div', { class: 'eyebrow' }, NAMES[id] || id),
    el('div', { class: 'empty glass' }, el('h1', { class: 'h1' }, NAMES[id] || id),
      el('p', {}, b.message || 'not built yet'),
      el('p', {}, 'This section shows its numbers once its order is done. Until then it shows none.')));
}

async function route() {
  const id = (location.hash || '#home').slice(1) || 'home';
  document.querySelectorAll('#nav a').forEach((a) => {
    const on = a.dataset.v === id;
    a.classList.toggle('on', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  document.querySelectorAll('.view').forEach((v) => { v.hidden = true; });
  const [vid, show] = views[id] || ['v-empty', (n) => emptyView(n, id)];
  const node = document.getElementById(vid);
  node.hidden = false;
  document.title = `Cubby · ${NAMES[id] || id}`;
  await show(node);
}

function audienceUi(pub) {
  const b = document.getElementById('pub'), sw = document.getElementById('pub-sw');
  b.setAttribute('aria-pressed', String(pub));
  sw.classList.toggle('on', pub);
  b.disabled = !hasToken();
  b.title = hasToken() ? (pub ? 'Public: the server sends what any user may see' : 'Audit: sources are shown') : 'Open the board with ?token= for the audit view';
  document.getElementById('aud-chip').textContent = pub ? 'public' : 'audit';
}

document.getElementById('pub').addEventListener('click', () => { setPublic(!isPublic()); });
onAudience((pub) => { audienceUi(pub); route(); });
window.addEventListener('hashchange', route);
audienceUi(isPublic());
route();
