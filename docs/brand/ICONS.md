# Cubby icons: the list to draw

Every icon the Cubby interfaces need, drawn in the style of the Cubby logo, so one set serves the control board
(CubbyLLM), the sign-in (cubby-auth) and CubbyConsul. Nick asked for it on 2026-11-10, with the one look (the
board's navy, `dashboard/brand/cubby-tokens.css`) and the logos (`dashboard/brand/`, from the UX reference repo's
`logos/`).

About 95 icons, plus the mark's own exports. Today the board's icons are hand-drawn placeholders
(`dashboard/board/index.html`, `dashboard/board/store_views.js`) and CubbyConsul uses about 45 stock Lucide icons;
the "Replaces" column names what each one takes the place of.

## The style

Taken from the logo: open arcs, round ends, and the small "circuit node" dots where a line ends.

| | |
|---|---|
| Grid | 24 × 24 px, 2 px padding (a 20 px live area) |
| Stroke | 1.75 px, round caps and joins, no fills except the node dots |
| Node dots | filled circles, 3 px across, at the end of a stroke, as on the logo's arcs; at most two per icon |
| Shapes | open arcs and strokes before closed outlines; one idea per icon; no text inside an icon |
| Variants | **mono**: one colour, `currentColor`, for the interface (white on the navy); **colour**: the logo's three colours (cyan `#00C1FA`, green `#64F592`, amber `#F0B42E`) at 48 px, for section tiles and Home |
| Files | real vectors (SVG, `viewBox="0 0 24 24"`, strokes not outlined, no embedded images); not traced from a bitmap. The current logo SVGs are traced and crack at small sizes: that is the thing to avoid |
| Names | `cubby-<group>-<name>.svg`, lower case, hyphens: `cubby-nav-knowledge.svg`; the colour variant adds `-color` |
| Check | readable at 16 px; the same weight next to the logo at 28 px; no sideways bleed past the 24 px box |

## 0. The mark itself (first)

| Item | Size | Notes |
|---|---|---|
| Clean vector of the logo | any | full colour and white; drawn, not traced (today's SVGs have hairline cracks) |
| Favicon | 16, 32, 48 px | tuned for small sizes (fewer, thicker arcs if needed); one `.ico` with the three |
| Apple touch icon | 180 px | on the navy `#0A111D`, not transparent |
| App icons | 192, 512 px | plus a maskable 512 (the mark inside the central 80 %) |
| Social card | 1200 × 630 px | the mark and "Cubby" on the navy stage |

## 1. Navigation (13)

| Icon | File | Used in | Replaces |
|---|---|---|---|
| Home | `cubby-nav-home` | board nav, CubbyConsul rail | house placeholder, Lucide `House` |
| World Watch | `cubby-nav-watch` | board nav | globe placeholder |
| Chat | `cubby-nav-chat` | board nav | speech-bubble placeholder |
| Worlds | `cubby-nav-worlds` | board nav | stacked layers placeholder, Lucide `Layers` |
| Knowledge | `cubby-nav-knowledge` | board nav | three-node graph placeholder |
| Forecast | `cubby-nav-forecast` | board nav | branching placeholder |
| Mood | `cubby-nav-mood` | board nav | heart placeholder |
| System | `cubby-nav-system` | board nav | chip placeholder, Lucide `Cpu` |
| Plugins | `cubby-nav-plugins` | board nav | plug placeholder, Lucide `Plug` |
| Agents | `cubby-nav-agents` | board nav | sparkle placeholder |
| Sessions | `cubby-nav-sessions` | CubbyConsul rail | Lucide `MessageSquareText` |
| Crew | `cubby-nav-crew` | CubbyConsul rail | Lucide `Boxes`, `Users` |
| Settings | `cubby-nav-settings` | CubbyConsul rail | Lucide `Settings` |

## 2. Knowledge store (14)

| Icon | File | Used in |
|---|---|---|
| Path | `cubby-store-path` | the Path tab: two nodes joined by an arc |
| Search | `cubby-store-search` | the Search tab; CubbyConsul search (Lucide `Search`, `SearchCheck`) |
| Entity | `cubby-store-entity` | the Entity tab |
| Who has | `cubby-store-who` | the Who-has tab: many nodes pointing at one |
| Resolve | `cubby-store-resolve` | the Resolve tab: a redirect (Lucide `CornerDownRight`) |
| Swap | `cubby-store-swap` | FROM ⇄ TO (Lucide `ArrowLeftRight`) |
| Zoom in | `cubby-store-zoom-in` | graph tools |
| Zoom out | `cubby-store-zoom-out` | graph tools |
| Fit | `cubby-store-fit` | graph tools |
| Expand one hop | `cubby-store-expand` | graph tools: "+1 hop" |
| As of | `cubby-store-as-of` | the as-of date (Lucide `Clock`) |
| Copy | `cubby-store-copy` | snapshot hash, keys (Lucide `Copy`) |
| Export | `cubby-store-export` | an entity as JSON |
| Outgoing | `cubby-store-outgoing` | a claim's "outgoing →" |

## 3. Kinds of thing (7)

Shown on entity cards and graph nodes (today `store_views.js` draws four placeholders).

| Icon | File | Means |
|---|---|---|
| Person | `cubby-kind-person` | instance of human (Q5) (Lucide `User`) |
| Place | `cubby-kind-place` | a country, city, region (Lucide `Building2`) |
| Work | `cubby-kind-work` | a book, film, album, article (Lucide `BookOpen`) |
| Thing | `cubby-kind-thing` | anything else |
| Property | `cubby-kind-property` | a statement's property, P-ids (Lucide `Hash`) |
| Hot set | `cubby-kind-hot` | the entity is in the hot store |
| Cold set | `cubby-kind-cold` | the entity is in the cold set |

## 4. Status (9)

| Icon | File | Means |
|---|---|---|
| Live | `cubby-status-live` | a section or a run is up |
| Not built yet | `cubby-status-waiting` | a section waiting on its order |
| Verified | `cubby-status-verified` | the VM certified it (Lucide `CircleCheck`, `ShieldCheck`) |
| Refused | `cubby-status-refused` | a refusal: a result, not an error; must not look like a failure |
| Wrong | `cubby-status-wrong` | a wrong answer: the defect; the strongest of the set |
| Warning | `cubby-status-warning` | a qualifier, a caveat (Lucide `TriangleAlert`) |
| Info | `cubby-status-info` | a note (Lucide `Info`, `CircleAlert`) |
| Loading | `cubby-status-loading` | waiting on the store or a run (Lucide `RefreshCw`) |
| Unreachable | `cubby-status-unreachable` | a service cannot be reached (the 503 from the sign-in gate) |

## 5. Home stats (8)

Beside the headline numbers on the board's Home, and as 48 px colour tiles.

| Icon | File | The number |
|---|---|---|
| Questions asked | `cubby-stat-asked` | questions today |
| Copy or refuse | `cubby-stat-copy-refuse` | the wrong / refused rates |
| Prompt journey | `cubby-stat-journey` | the last question's steps |
| Heartbeat | `cubby-stat-heartbeat` | bpm from arousal |
| Mood cube | `cubby-stat-cube` | the nearest corner |
| Orders | `cubby-stat-orders` | the order queue (Lucide `Receipt`) |
| News feed | `cubby-stat-news` | the news poller |
| Training run | `cubby-stat-training` | runs with logs |

## 6. Chat and sessions (17)

| Icon | File | Replaces |
|---|---|---|
| Send | `cubby-chat-send` | Lucide `SendHorizontal` |
| New | `cubby-chat-new` | Lucide `Plus` |
| Quick ask | `cubby-chat-quick-ask` | Lucide `Lightbulb` |
| Audit trail | `cubby-chat-audit` | Lucide `Eye` (audit) |
| History | `cubby-chat-history` | Lucide `Clock` (history) |
| Right | `cubby-chat-right` | Lucide `Check` |
| Wrong | `cubby-chat-wrong` | Lucide `X` (feedback) |
| Pin | `cubby-chat-pin` | Lucide `Pin` |
| Play | `cubby-chat-play` | Lucide `Play` |
| Pause | `cubby-chat-pause` | Lucide `Pause` |
| Stop | `cubby-chat-stop` | Lucide `Square` |
| Speaker on | `cubby-chat-speaker-on` | Lucide `Volume2` |
| Speaker off | `cubby-chat-speaker-off` | Lucide `VolumeX` |
| Delete | `cubby-chat-delete` | Lucide `Trash2` |
| Close | `cubby-chat-close` | Lucide `X` |
| Menu | `cubby-chat-menu` | Lucide `Menu` |
| Expand / collapse | `cubby-chat-chevron` | Lucide `ChevronDown`, `ChevronRight`, `ArrowDown` |

## 7. Account and security (14)

Mostly for cubby-auth's account page and the board's header.

| Icon | File | Replaces |
|---|---|---|
| Account | `cubby-auth-account` | Lucide `User` (account) |
| Passkey | `cubby-auth-passkey` | Lucide `KeyRound` (passkey) |
| Lock | `cubby-auth-lock` | the board's "Lock" button, Lucide `Lock` |
| Unlock | `cubby-auth-unlock` | Lucide `Unlock`, `LockOpen` |
| Sign out | `cubby-auth-sign-out` | |
| Sign out everywhere | `cubby-auth-sign-out-all` | |
| Company | `cubby-auth-company` | Lucide `Building2` (tenants) |
| Invite | `cubby-auth-invite` | |
| API key | `cubby-auth-api-key` | Lucide `KeyRound` (API access) |
| Roles | `cubby-auth-roles` | Lucide `ShieldCheck` (roles) |
| Idle timer | `cubby-auth-idle` | the idle-lock setting |
| Recovery link | `cubby-auth-recovery` | |
| Show | `cubby-auth-show` | Lucide `Eye` |
| Hide | `cubby-auth-hide` | Lucide `EyeOff` |

## 8. CubbyConsul's seats and models (8)

The five seat icons take the seat-role colours already in CubbyConsul's `theme.css` (`--r-proposer` …) in the colour
variant.

| Icon | File | Replaces |
|---|---|---|
| Proposer | `cubby-seat-proposer` | Lucide `Lightbulb` (seat) |
| Prover | `cubby-seat-prover` | Lucide `Sigma` |
| Skeptic | `cubby-seat-skeptic` | Lucide `SearchCheck` (seat) |
| Librarian | `cubby-seat-librarian` | Lucide `BookOpen` (seat) |
| Scribe | `cubby-seat-scribe` | Lucide `Feather` |
| Local model | `cubby-seat-local-model` | Lucide `Cpu` (local models) |
| Plugin | `cubby-seat-plugin` | Lucide `Plug` (plugins) |
| Sealed session | `cubby-seat-sealed` | Lucide `Lock` (sealed sessions) |

## Delivery

- One folder of SVGs named as above (mono, and `-color` where a colour variant is listed), plus the mark's exports
  in section 0.
- Drop them in the UX reference repo's `icons/` beside `logos/`; they get copied into each app (CubbyLLM
  `dashboard/brand/icons/`, cubby-auth `web/`, CubbyConsul `web/public/icons/`), as the logos were.
- First wave: section 0, then 1 (navigation) and 3 (kinds of thing), which today are placeholders on the board.
