/* Clausa app window: projects in a sidebar; the project shown, its recordings and their measures.
   A project is a named set of recordings in one language, built up over several sessions
   (runner/projects.py); analyses made before projects are listed read-only as earlier analyses.
   The page talks to the backend only through `api`: window.pywebview.api inside the app (ready
   after the pywebviewready event), otherwise, or with ?mock=..., the built-in mock at the end of
   this file, which serves example data. The backend owns the state; the page polls get_state()
   and patches the DOM in place, so the selection, open menus, focus and the log's scroll position
   survive every poll.
   Mock views: ?mock=project (default) | running | welcome. Options: &project=<id>, &select=N,
   &filter=done|running|waiting|failed|new, &side=0|1 (sidebar; default: shown from 1000 px),
   &log=1, &dialog=new (&name=..., &dlang=ja), &confirm=delete|again, &menu=project|settings,
   &still=1 (no live ticking, for screenshots). */
(function () {
  'use strict';

  var CODES = ['AR', 'SR', 'MLR', 'MCPR', 'ECPR', 'PR', 'MCPD', 'ECPD', 'MPD'];
  var DECIMALS = [2, 2, 1, 3, 3, 3, 2, 2, 2];
  // the nine measures fall into three groups of three: speed, how often pauses come, how long they last
  var GROUPS = [{ en: 'Speed', ja: 'Speed', from: 0 }, { en: 'Pauses per syllable', ja: 'Pauses per mora', from: 3 }, { en: 'Mean pause duration', ja: 'Mean pause duration', from: 6 }];
  var NAMES = ['Articulation rate', 'Speech rate', 'Mean length of run', 'Mid-clause', 'End-clause', 'All pauses', 'Mid-clause', 'End-clause', 'All pauses'];
  // a failed recording's row names the step it stopped in (runner/app.py, STEPS)
  var FAILED_AT = { 'Reading audio': 'Reading failed', 'Speech recognition': 'Recognition failed', 'Word alignment': 'Alignment failed',
                    'Clauses': 'Clauses failed', 'Filled pauses': 'Filled pauses failed', 'Pauses': 'Pauses failed', 'Measures': 'Measures failed' };
  // the running stage in one word, where the progress column is narrow (the status bar has it in full)
  var SHORT = { 'Speech recognition': 'Recognition', 'Word alignment': 'Alignment', 'Filled pauses': 'Fillers', 'Reading audio': 'Reading' };
  var LANGS = { en: 'English', ja: 'Japanese' }, TAGS = { en: 'EN', ja: 'JA' };
  // "waiting" is queued in the current session; "new" was added and has not been analysed
  var STATUS = { done: 'Done', failed: 'Failed', running: 'Running', waiting: 'Waiting', new: 'New' };
  var FILTERS = ['done', 'running', 'waiting', 'failed', 'new'];
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var LEVELS = { info: 'plain', ok: 'ok', error: 'bad' };
  var POLL_ACTIVE_MS = 500, POLL_IDLE_MS = 2000, BRIDGE_WAIT_MS = 1500, NOTICE_MS = 6000;
  // The sidebar is shown from SIDE_MIN_W; while the details panel is open it hides itself unless the
  // window holds both beside the table: 232 + the compact table (944) + the panel (304).
  var SIDE_MIN_W = 1000, SIDE_AND_DETAILS_W = 1480;
  // outline icons, one set: 24-unit grid, 1.75 stroke, drawn at 14-16 px in currentColor
  function icon(d, size) { return '<svg aria-hidden="true" width="' + (size || 16) + '" height="' + (size || 16) + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">' + d + '</svg>'; }
  var FOLDER = icon('<path d="M3.5 7.5a2 2 0 0 1 2-2h3.6l2 2h7.4a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>');
  var CROSS = icon('<path d="M7 7l10 10M17 7L7 17"/>');
  var I_DONE = icon('<path d="M5.5 12.5l4 4 9-9"/>', 14);
  var I_FAIL = icon('<path d="M12 7.5v5.5"/><path d="M12 16.5v.01"/><circle cx="12" cy="12" r="8.5"/>', 14);
  var I_WAIT = icon('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>', 14);
  var I_NEW = icon('<circle cx="12" cy="12" r="7.5" stroke-dasharray="2.5 2.6"/>', 14);
  var I_FILE = icon('<path d="M5.5 12.5l4 4 9-9"/>', 14);
  var I_NOFILE = icon('<circle cx="12" cy="12" r="7.5" stroke-dasharray="3 3"/>', 14);
  var I_DIR = icon('<path d="M3.5 7.5a2 2 0 0 1 2-2h3.6l2 2h7.4a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>', 14);
  var PROG = '<div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100"><span class="stage"></span><div class="track"><div class="fill"></div></div><span class="pct"></span></div>';

  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);

  var api = null, mock = null;
  // sel: selected recording id; log: drawer open; follow: the log sticks to its bottom; filter: the
  // summary part shown alone; side: the sidebar as the user set it (null: by the window's width);
  // dialog: New project; confirm: {title, text, ok, danger, action}; renaming: the title is a field
  var S = { state: null, sel: null, log: false, follow: true, notice: '', noticeId: 0, drag: false, pendingSelect: 0,
            filter: null, side: null, dialog: false, newLang: 'en', confirm: null, renaming: false, shown: null };

  /* ---------- helpers ---------- */

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many || one + 's'); }
  function baseName(p) { var parts = String(p || '').split(/[\\/]+/).filter(Boolean); return parts.length ? parts[parts.length - 1] : ''; }
  function fmtDur(s) {
    if (s == null || !isFinite(s)) return '';
    s = Math.round(s);
    var h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60;
    return (h ? h + ':' + pad2(m) : m) + ':' + pad2(x);
  }
  function fmtElapsed(s) {
    if (s == null || !isFinite(s)) return '';
    s = Math.round(s);
    if (s < 60) return s + ' s';
    if (s < 3600) return Math.floor(s / 60) + ' min' + (s % 60 ? ' ' + (s % 60) + ' s' : '');
    return Math.floor(s / 3600) + ' h ' + (Math.floor(s / 60) % 60) + ' min';
  }
  function fmtEta(s) {
    if (s == null || !isFinite(s) || s < 0) return '';
    if (s < 10) return 'a few seconds';
    if (s < 55) return 'about ' + Math.round(s / 10) * 10 + ' s';
    if (s < 3570) return 'about ' + Math.max(1, Math.round(s / 60)) + ' min';
    var m = Math.round(s / 60);
    return 'about ' + Math.floor(m / 60) + ' h' + (m % 60 ? ' ' + (m % 60) + ' min' : '');
  }
  // eta_s: seconds left, null when unknown, 0 once the estimate has run out
  function etaText(s) {
    if (s == null || !isFinite(s)) return '';
    return s <= 0 ? 'almost done' : fmtEta(s) + ' left';
  }
  function fmtTime(t) {
    if (typeof t !== 'number') return String(t == null ? '' : t);
    var d = new Date(t > 1e12 ? t : t * 1000);
    return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }
  // An ISO date or time ('2026-10-15T13:40:02+09:00') as the rows show it: 'today', '15 Oct', or
  // '14 Nov 2025' in another year (year: always with the year, as in "last 15 Oct 2026").
  function dayText(iso, year) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    if (!m) return '';
    var now = new Date(), today = now.getFullYear() + '-' + pad2(now.getMonth() + 1) + '-' + pad2(now.getDate());
    if (m[0] === today) return 'today';
    return Number(m[3]) + ' ' + MONTHS[Number(m[2]) - 1] + (year || Number(m[1]) !== now.getFullYear() ? ' ' + m[1] : '');
  }
  // the line under a recording's name: the date it was added (New) or last analysed
  function whenText(r) {
    if (r.status === 'new') return r.added ? 'added ' + dayText(r.added) : '';
    return r.status === 'done' || r.status === 'failed' ? dayText(r.analysed) : '';
  }
  function fmtMeasure(v, k) {
    var n = typeof v === 'number' ? v : parseFloat(v);
    return isFinite(n) ? n.toFixed(DECIMALS[k]) : null;
  }
  function units(lang) {
    var u = lang === 'ja' ? 'morae' : 'syll', p = lang === 'ja' ? '/mora' : '/syll';
    return [u + '/s', u + '/s', u, p, p, p, 's', 's', 's'];
  }
  function projOf(st) { return (st && st.project) || null; }
  function readonly(st) { var p = projOf(st); return !!p && p.kind === 'earlier'; }
  function langOf(st) { var p = projOf(st); return p && p.language === 'ja' ? 'ja' : 'en'; }
  function setText(el, t) { if (el.textContent !== t) el.textContent = t; }
  function setDisabled(el, v) { el.setAttribute('aria-disabled', String(!!v)); }
  function enabled(el) { return el.getAttribute('aria-disabled') !== 'true'; }

  // Replace an element's HTML only when it changed; keep focus on an element with the same data-key.
  function patch(el, html) {
    if (el._html === html) return;
    var a = document.activeElement, key = a && a !== el && el.contains(a) ? a.getAttribute('data-key') : null;
    el.innerHTML = html;
    el._html = html;
    if (!key) return;
    var all = el.querySelectorAll('[data-key]');
    for (var i = 0; i < all.length; i++) if (all[i].getAttribute('data-key') === key) { all[i].focus(); return; }
  }

  /* ---------- session and recording state ---------- */

  function runOf(st) { return (st && st.run) || {}; }
  function isActive(run) { return run.status === 'running' || run.status === 'stopping'; }
  // one session at a time, in one project; the window may show another
  function runHere(st) { var run = runOf(st), p = projOf(st); return isActive(run) && !!p && run.project_id === p.id; }
  function recsOf(st) { return (st && st.recordings) || []; }
  function findRec(id) {
    var R = recsOf(S.state);
    for (var i = 0; i < R.length; i++) if (String(R[i].id) === String(id)) return R[i];
    return null;
  }
  function isCurrent(r, run) {
    var cf = run.current_file;
    return isActive(run) && cf != null && (String(cf) === String(r.id) || cf === r.name || baseName(cf) === r.name);
  }
  // The runner works stage by stage over all recordings of the session: only current_file shows
  // Running, the others in the session wait for their turn (each with its own progress).
  function shownStatus(r, run) {
    var s = r.status;
    if (s === 'running') return isCurrent(r, run) ? 'running' : 'waiting';
    return s === 'done' || s === 'failed' || s === 'waiting' ? s : 'new';
  }
  function inRun(r, run) { return isActive(run) && (r.status === 'running' || r.status === 'waiting'); }
  function progPct(r, s) {
    if (s === 'done') return 100;
    var p = Number(r.progress) || 0;
    return Math.round(Math.max(0, Math.min(99, p)) * 10) / 10;
  }
  function measureList(r) {
    if (!r.measures) return null;
    return CODES.map(function (k, j) { return fmtMeasure(r.measures[k], j); });
  }
  function failedText(r) { return FAILED_AT[r.failed_stage] || 'Failed'; }
  function groupStart(k) { return k === 3 || k === 6; }
  function counts(st) {
    var run = runOf(st), c = { all: 0, done: 0, running: 0, waiting: 0, failed: 0, new: 0 };
    recsOf(st).forEach(function (r) { c.all++; c[shownStatus(r, run)]++; });
    return c;
  }

  /* ---------- adapter ---------- */

  function connect() {
    var view = params.get('mock');
    if (view !== null) { useMock(view); return; }
    if (bridgeReady()) { usePywebview(); return; }
    window.addEventListener('pywebviewready', usePywebview);
    // No bridge in a page opened from a file: a plain browser. The mock then says so in the
    // status bar; a bridge that turns up later still takes over. The app serves its page over
    // http and only ever waits for its bridge, however long that takes.
    if (location.protocol === 'file:') setTimeout(function () { if (!api) useMock('project', true); }, BRIDGE_WAIT_MS);
  }
  function bridgeReady() { return !!(window.pywebview && window.pywebview.api && typeof window.pywebview.api.get_state === 'function'); }
  function usePywebview() {
    if (!bridgeReady() || (api && !mock)) return;
    if (mock) { mock.dispose(); mock = null; }
    api = window.pywebview.api;
    refresh();
  }
  function useMock(view, preview) {
    mock = createMock(view, preview);
    api = mock.api;
    S.log = params.get('log') === '1';
    S.pendingSelect = Number(params.get('select')) || 0;
    S.filter = FILTERS.indexOf(params.get('filter')) >= 0 ? params.get('filter') : null;
    if (params.get('side') === '0' || params.get('side') === '1') S.side = params.get('side') === '1';
    refresh().then(function () {
      if (params.get('dialog') === 'new') openDialog(params.get('name') || '', params.get('dlang') === 'ja' ? 'ja' : 'en');
      if (params.get('confirm') === 'delete') confirmDelete();
      if (params.get('confirm') === 'again') confirmAgainAll();
      if (params.get('confirm') === 'remove' && recsOf(S.state).length) confirmRemove(recsOf(S.state)[0]);
      if (params.get('menu') === 'project') $('projMenuBtn').click();
      if (params.get('menu') === 'settings') $('settingsBtn').click();
    });
  }

  // Call a backend action, then poll at once so the result shows without waiting.
  function call(name) {
    var args = Array.prototype.slice.call(arguments, 1), p;
    if (!api || typeof api[name] !== 'function') return Promise.resolve(null);
    try { p = Promise.resolve(api[name].apply(api, args)); } catch (e) { p = Promise.reject(e); }
    return p.then(function (v) { refresh(); return v; }, function (e) {
      refresh();
      note('Something went wrong: ' + ((e && e.message) || String(e)));
      return null;
    });
  }

  var pollTimer = null, polling = null, pollAgain = false;
  // Returns a promise that settles once this poll (or the one running) has rendered.
  function refresh() {
    if (!api) return Promise.resolve();
    if (polling) { pollAgain = true; return polling; }
    clearTimeout(pollTimer);
    var from = api, p;
    try { p = Promise.resolve(api.get_state()); } catch (e) { p = Promise.reject(e); }
    polling = p.then(function (st) {
      if (from === api && st) { S.state = st; render(); }
    }, function () { /* keep the last state; the next poll tries again */ }).then(function () {
      polling = null;
      if (pollAgain) { pollAgain = false; return refresh(); }
      pollTimer = setTimeout(refresh, isActive(runOf(S.state)) ? POLL_ACTIVE_MS : POLL_IDLE_MS);
    });
    return polling;
  }

  var noticeTimer = null;
  function note(text) {
    S.notice = text;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(function () { S.notice = ''; if (S.state) renderBars(S.state); }, NOTICE_MS);
    if (S.state) renderBars(S.state);
  }

  /* ---------- rendering ---------- */

  function render() {
    var st = S.state;
    if (!st) return;
    var p = projOf(st), shown = p ? p.id : null;
    if (shown !== S.shown) {  // another project: its own selection, filter and title
      if (S.shown !== null) { S.sel = null; S.filter = null; }
      S.shown = shown;
      endRename(false);
    }
    if (S.pendingSelect) {
      var pr = recsOf(st)[S.pendingSelect - 1];
      if (pr) S.sel = String(pr.id);
      S.pendingSelect = 0;
    }
    if (S.sel !== null && !findRec(S.sel)) S.sel = null;
    if (st.notice && st.notice.id && st.notice.id !== S.noticeId) {
      S.noticeId = st.notice.id;
      if (st.notice.text) note(st.notice.text);
    }
    var none = !p, wel = $('welcome'), bar = document.querySelector('.toolbar');
    $('win').classList.toggle('no-projects', none);
    $('win').classList.toggle('readonly', readonly(st));
    wel.hidden = !none;
    // one title per screen, first in the document: the welcome's own h1 goes ahead of the (hidden)
    // project title when there is no project; the stylesheet keeps it under the toolbar
    if (none && wel.nextElementSibling !== bar) bar.parentNode.insertBefore(wel, bar);
    if (!none && wel.nextElementSibling === bar) bar.parentNode.insertBefore(wel, $('bodyBox').nextSibling);
    applySide();
    renderSidebar(st);
    renderBars(st);
    renderDialog(st);
    if (none) return;
    renderSummary(st);
    renderHead(st);
    renderRows(st);
    renderInspector(st);
    renderLog(st);
  }

  // The sidebar: shown or hidden by the toolbar's toggle; by default hidden under 1000 px, and while
  // the details panel is open in a window that cannot hold both beside the table.
  function sideOpen() {
    var w = window.innerWidth, user = S.side === null ? w >= SIDE_MIN_W : S.side;
    return user && !(S.sel !== null && w < SIDE_AND_DETAILS_W);
  }
  function applySide() {
    var open = sideOpen(), t = $('sideToggle');
    $('win').classList.toggle('side-closed', !open);
    t.setAttribute('aria-expanded', String(open));
    t.setAttribute('aria-label', open ? 'Hide the sidebar' : 'Show the sidebar');
    t.title = open ? 'Hide the sidebar' : 'Show the sidebar';
  }

  function ring(pct) {
    var c = 2 * Math.PI * 5.5, on = Math.max(0, Math.min(100, pct)) / 100 * c;
    return '<svg class="ring" aria-hidden="true" width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5.5"/><circle class="on" cx="7" cy="7" r="5.5" stroke-dasharray="' + on.toFixed(2) + ' ' + c.toFixed(2) + '" transform="rotate(-90 7 7)"/></svg>';
  }
  function sideItem(p, cur, earlier) {
    var side = '';
    if (p.status === 'running') side = '<span class="prun" title="Analysing: ' + Math.floor(p.progress) + '% of this session">' + ring(p.progress) + '<span>' + Math.floor(p.progress) + '%</span></span>';
    else if (p.failed) side = '<span class="pfail" title="' + plural(p.failed, 'failed recording') + '">' + p.failed + '</span>';
    var lang = p.language === 'ja' ? 'ja' : 'en';
    return '<button type="button" class="proj' + (earlier ? ' earlier' : '') + '" data-project="' + esc(p.id) + '" data-key="p-' + esc(p.id) + '"' +
      (cur && cur.id === p.id ? ' aria-current="page"' : '') + ' title="' + esc(earlier ? 'Analysed on ' + p.name + ' (read-only)' : p.name) + '">' +
      '<span class="pn">' + esc(p.name) + '</span>' +
      '<span class="pm">' + (p.language ? '<span class="tag" title="' + LANGS[lang] + '">' + TAGS[lang] + '</span>' : '') + plural(p.count, 'recording') + '</span>' +
      (side ? '<span class="ps">' + side + '</span>' : '') + '</button>';
  }
  function renderSidebar(st) {
    var P = st.projects || [], E = st.earlier || [], cur = projOf(st), set = st.settings || {};
    patch($('projList'), P.map(function (p) { return sideItem(p, cur, false); }).join('') || '<p class="side-empty">None yet</p>');
    $('earlierLabel').hidden = $('earlierList').hidden = !E.length;
    patch($('earlierList'), E.map(function (e) { return sideItem(e, cur, true); }).join(''));
    var dir = set.projects_dir_short || set.projects_dir || '';
    setText($('folderPath'), dir);
    $('settingsBtn').title = 'Projects are saved in ' + dir;
    setDisabled($('changeFolderBtn'), isActive(runOf(st)));
    $('changeFolderBtn').title = isActive(runOf(st)) ? 'Not while a session runs' : '';
  }

  function statusText(st) {
    var run = runOf(st), set = st.settings || {}, p = projOf(st), n = run.total || 0, k = run.done_count || 0, f = run.failed_count || 0;
    if (!(st.projects || []).length && !p) return 'Ready · projects are saved in ' + (set.projects_dir_short || set.projects_dir || '');
    switch (run.status) {
      case 'running': {
        // the current recording, its step, and the session's share (also in the sidebar)
        var parts = ['Analysing ' + (run.current_name || baseName(run.current_file) || '') + (runHere(st) ? '' : ' in ' + (run.project_name || 'another project'))];
        if (run.stage_label) parts.push(run.stage_label + (run.stage_index ? ' ' + run.stage_index + '/' + (run.stage_count || 5) : ''));
        parts.push(Math.floor(run.progress || 0) + '% of the session');
        var eta = etaText(run.eta_s);
        if (eta) parts.push(eta);
        return parts.join(' · ');
      }
      case 'stopping': return 'Stopping…';
    }
    if (readonly(st)) return 'An analysis made before projects (read-only) · ' + (p.dir || '');
    if ((run.status === 'done' || run.status === 'stopped') && p && run.project_id === p.id) {
      return [run.status === 'done' ? 'Done' : 'Stopped', k + ' of ' + n + ' analysed in this session', f ? f + ' failed' : '', fmtElapsed(run.elapsed_s)].filter(Boolean).join(' · ');
    }
    if (run.status === 'failed' && p && run.project_id === p.id) return 'The session failed · see the log';
    var c = counts(st), todo = c.new + c.failed;
    if (!c.all) return 'Ready · add recordings to this project';
    if (todo) {
      var what = [c.new ? c.new + ' new' : '', c.failed ? c.failed + ' failed' : ''].filter(Boolean).join(' and ');
      var verb = c.new ? 'analyses' : 'tries';
      return 'Ready · ' + what + ': Start ' + verb + ' ' + (todo === 1 ? 'it' : 'them') + (c.new ? '' : ' again') + (c.done ? '; the ' + c.done + ' done keep their results' : '');
    }
    return 'All ' + c.done + ' analysed · uf_measures.csv has ' + plural(c.done, 'row') + (p && p.last_session ? ' · last session ' + dayText(p.last_session, true) : '');
  }

  function renderBars(st) {
    var run = runOf(st), R = recsOf(st), p = projOf(st), app = st.app || {};
    var busy = isActive(run), here = runHere(st), lang = langOf(st), c = counts(st), ro = readonly(st);
    var focusStart = document.activeElement === $('startBtn'), focusStop = document.activeElement === $('stopBtn');

    $('empty').hidden = !!R.length || ro;
    $('rowsTable').classList.toggle('is-empty', !R.length);
    renderDrop();

    // Start analyses every recording that is not done; Stop shows in the project that runs
    $('startBtn').hidden = here;
    $('stopBtn').hidden = !here;
    setDisabled($('startBtn'), busy || !(c.new + c.failed));
    $('startBtn').title = busy ? 'One session at a time: ' + (run.project_name || 'another project') + ' is being analysed'
      : c.new + c.failed ? 'Analyse the ' + plural(c.new + c.failed, 'recording') + ' not done' : 'Every recording is analysed (Analyse again is in the project menu and the details)';
    setDisabled($('stopBtn'), run.status === 'stopping');
    setText($('stopLabel'), run.status === 'stopping' ? 'Stopping…' : 'Stop');
    if (here && focusStart) $('stopBtn').focus();
    if (!here && focusStop) $('startBtn').focus();
    setDisabled($('csvBtn'), !(p && p.csv));
    $('csvBtn').title = p && p.csv ? 'uf_measures.csv: one row per analysed recording' : 'No recording is analysed yet';
    setDisabled($('renameBtn'), here);
    setDisabled($('deleteBtn'), here);
    setDisabled($('againAllBtn'), busy || !R.length);

    if (!S.renaming) setText($('projName'), p ? p.name : '');
    $('projName').title = ro ? 'Analysed on ' + (p ? p.name : '') : here ? 'Rename it when the session has ended' : 'Click to rename';
    setText($('projLang'), TAGS[lang]);
    $('projLang').title = LANGS[lang] + (lang === 'ja' ? ': measures count morae' : ': measures count syllables');
    setText($('projCount'), plural(R.length, 'recording'));

    var dot = busy ? 'run' : run.status === 'failed' ? 'bad' : run.status === 'done' && p && run.project_id === p.id ? '' : 'idle';
    $('sdot').className = 'dot' + (dot ? ' ' + dot : '');
    setText($('smsg'), S.notice || statusText(st));
    setText($('ver'), [app.device, app.version ? 'v' + app.version : ''].filter(Boolean).join(' · '));
  }

  // The summary line names the parts of the project, and each part filters the table.
  function renderSummary(st) {
    var c = counts(st), p = projOf(st), run = runOf(st);
    if (S.filter && !c[S.filter]) S.filter = null;
    var h = '<button type="button" class="fbtn all" data-filter="" data-key="f-all" aria-pressed="' + String(!S.filter) + '">' + plural(c.all, 'recording') + '</button>';
    FILTERS.forEach(function (k) {
      if (!c[k]) return;
      h += '<span class="fsep" aria-hidden="true">·</span><button type="button" class="fbtn" data-filter="' + k + '" data-key="f-' + k + '" aria-pressed="' + String(S.filter === k) + '"><i class="sd ' + k + '" aria-hidden="true"></i>' + c[k] + ' ' + k + '</button>';
    });
    patch($('filters'), h);
    var stamp = '';
    if (p && p.kind === 'earlier') stamp = 'Analysed ' + dayText(p.last_session, true) + ' · read-only';
    else if (p) {
      if (runHere(st)) stamp = 'Session ' + ((p.sessions || 0) + 1) + ' · started ' + (run.started || '');
      else if (p.sessions) stamp = plural(p.sessions, 'session') + ' · last ' + dayText(p.last_session, true);
      else stamp = 'No session yet';
    }
    setText($('stamp'), stamp);
  }

  function renderHead(st) {
    var u = units(langOf(st)), n = runOf(st).stage_count || 5;
    var h = '<th scope="col" class="no"><div class="h">#</div><div class="u">&nbsp;</div></th><th scope="col" class="rec-h"><div class="h">Recording</div><div class="u">name · length · date</div></th>' +
            '<th scope="col"><div class="h">Progress</div><div class="u">' + n + ' stages</div></th>' +
            '<th scope="col"><div class="h">Status · TextGrid</div><div class="u">W words · C clauses</div></th>';
    for (var i = 0; i < 9; i++) h += '<th scope="col" class="num' + (groupStart(i) ? ' g' : '') + '" title="' + esc(NAMES[i]) + '"><div class="h">' + CODES[i] + '</div><div class="u">' + u[i] + '</div></th>';
    patch($('headRow'), h + '<th scope="col"><span class="sr-only">Actions</span></th>');
  }

  function badge(s) {
    if (s === 'running') return '<span class="badge running"><span class="pulse" aria-hidden="true"></span>Running</span>';
    var ic = { done: I_DONE, failed: I_FAIL, waiting: I_WAIT, new: I_NEW }[s];
    return '<span class="badge ' + s + '">' + ic + STATUS[s] + '</span>';
  }
  function tgBadge(letter, kind, state) {
    state = state === 'yes' || state === 'wip' ? state : 'no';
    var what = { yes: 'saved', wip: 'being written', no: 'not yet' }[state];
    return '<span class="tg ' + state + '" title="' + kind + ' TextGrid: ' + what + '" aria-label="' + kind + ' TextGrid ' + what + '">' + letter + '</span>';
  }
  // A row's one action: its folder when done; remove when it is not part of a running session.
  function rowAction(r, run, st) {
    if (r.status === 'done') return 'open';
    return readonly(st) || inRun(r, run) ? null : 'remove';
  }
  function actionButton(r, run, st) {
    var a = rowAction(r, run, st), id = esc(r.id);
    if (a === 'open') return '<button type="button" class="act" data-action="open-folder" data-id="' + id + '" data-key="act" aria-label="Open the folder of ' + esc(r.name) + '" title="Open its session folder">' + FOLDER + '</button>';
    if (a === 'remove') return '<button type="button" class="act" data-action="remove" data-id="' + id + '" data-key="act" aria-label="Remove ' + esc(r.name) + ' from the project" title="Remove from the project">' + CROSS + '</button>';
    return '';
  }
  // The progress cell: a thin bar and the percentage; above them, the stage of the running row or
  // where a failed one stopped ("Alignment failed", never cut: it wraps where the column is narrow).
  // A row queued in the session shows its real share; a New one shows a grey dash.
  function setProg(box, r, s, run) {
    var p = progPct(r, s), bar = box.querySelector('.prog'), fill = box.querySelector('.fill'), pct = box.querySelector('.pct'), stage = box.querySelector('.stage');
    var fc = s === 'failed' ? 'fill bad' : s === 'running' ? 'fill live' : s === 'done' ? 'fill' : 'fill part';
    if (fill.className !== fc) fill.className = fc;
    if (s === 'new') p = 0;
    if (fill.style.width !== p + '%') fill.style.width = p + '%';
    var label = s === 'running' ? run.stage_label || 'Analysing' : '';
    setText(stage, s === 'running' ? label : s === 'failed' ? failedText(r) : '');
    stage.setAttribute('data-short', SHORT[label] || label);
    setText(pct, s === 'failed' ? '' : s === 'new' ? '—' : Math.floor(p) + '%');
    bar.className = 'prog ' + s;
    bar.setAttribute('aria-valuenow', String(Math.floor(p)));
    bar.setAttribute('aria-label', r.name + ' progress');
  }

  var rowEls = {};
  function cell(cls) { var td = document.createElement('td'); if (cls) td.className = cls; return td; }
  function buildRow(id) {
    var tr = document.createElement('tr'), c = { no: cell('no'), rec: cell('rec-c'), prog: cell('prog-c'), st: cell(''), nums: [], act: cell('actcell') };
    tr.tabIndex = 0;
    tr.setAttribute('data-id', id);
    c.prog.innerHTML = PROG;
    tr.appendChild(c.no); tr.appendChild(c.rec); tr.appendChild(c.prog); tr.appendChild(c.st);
    for (var k = 0; k < 9; k++) { c.nums.push(cell('num none' + (groupStart(k) ? ' g' : ''))); tr.appendChild(c.nums[k]); }
    tr.appendChild(c.act);
    tr._c = c;
    return tr;
  }
  function renderRows(st) {
    var tbody = $('rows'), run = runOf(st), seen = {}, lostFocusAt = -1, shown = 0, R = recsOf(st), pid = (projOf(st) || {}).id;
    R.forEach(function (r, i) {
      var key = pid + '\u0000' + r.id, tr = rowEls[key] || (rowEls[key] = buildRow(r.id));
      seen[key] = true;
      if (tbody.children[i] !== tr) tbody.insertBefore(tr, tbody.children[i] || null);
      updateRow(tr, r, i, run, st);
      var hide = !!S.filter && shownStatus(r, run) !== S.filter;
      if (tr.hidden !== hide) tr.hidden = hide;
      if (!hide) shown++;
    });
    Object.keys(rowEls).forEach(function (key) {
      if (seen[key]) return;
      if (rowEls[key].contains(document.activeElement)) lostFocusAt = Array.prototype.indexOf.call(tbody.children, rowEls[key]);
      if (rowEls[key].parentNode) tbody.removeChild(rowEls[key]);
      delete rowEls[key];
    });
    if (lostFocusAt >= 0) {
      var next = tbody.children[lostFocusAt] || tbody.children[lostFocusAt - 1];
      if (next) next.focus();
    }
    // a filtered table shrinks to its rows and says what it leaves out
    $('addslot').hidden = !R.length || !!S.filter || readonly(st);
    $('filterinfo').hidden = !S.filter;
    if (S.filter) setText($('filterText'), 'Showing the ' + shown + ' ' + S.filter + ' of ' + plural(R.length, 'recording') + ' · ');
  }
  function updateRow(tr, r, i, run, st) {
    var c = tr._c, s = shownStatus(r, run), tg = r.textgrids || {}, vals = measureList(r), cls = s === 'running' ? 'running' : '', when = whenText(r);
    if (tr.className !== cls) tr.className = cls;
    tr.setAttribute('aria-selected', String(S.sel === String(r.id)));
    setText(c.no, pad2(i + 1));
    patch(c.rec, '<div class="rec" title="' + esc(r.name) + '"><b>' + esc(r.name) + '</b><span>' + fmtDur(r.duration_s) + (when ? (r.duration_s != null ? ' · ' : '') + esc(when) : '') + '</span></div>');
    setProg(c.prog, r, s, run);
    patch(c.st, '<div class="st">' + badge(s) + tgBadge('W', 'Words', tg.words) + tgBadge('C', 'Clauses', tg.clauses) + '</div>');
    for (var k = 0; k < 9; k++) {
      var v = vals ? vals[k] : null, nc = (v === null ? 'num none' : 'num') + (groupStart(k) ? ' g' : '');
      setText(c.nums[k], v === null ? '—' : v);
      if (c.nums[k].className !== nc) c.nums[k].className = nc;
    }
    patch(c.act, actionButton(r, run, st));
  }

  function renderInspector(st) {
    var run = runOf(st), r = S.sel === null ? null : findRec(S.sel), el = $('inspector'), ro = readonly(st);
    $('win').classList.toggle('ins-open', !!r);
    el.hidden = !r;
    if (!r) return;
    var s = shownStatus(r, run), id = esc(r.id), u = units(langOf(st)), when = whenText(r);

    patch($('insHead'), '<div class="ins-title"><h2 title="' + esc(r.name) + '">' + esc(r.name) + '</h2><div class="ins-meta">' + badge(s) + '<span class="d">' + fmtDur(r.duration_s) + (when ? (r.duration_s != null ? ' · ' : '') + esc(when) : '') + '</span></div></div>' +
      '<button type="button" class="iconbtn" data-action="close-inspector" data-key="close" aria-label="Close">' + CROSS + '</button>');

    var err = $('insErr');
    err.hidden = s !== 'failed';
    if (s === 'failed') patch(err, '<b>' + esc(failedText(r)) + '</b><span>' + esc(r.error || 'This recording could not be analysed.') + '</span>' +
      (ro ? '' : '<span class="again">Start tries it again in the next session.</span>') +
      '<button type="button" class="small" data-action="show-in-log" data-id="' + id + '" data-key="log">Show in log</button>');

    // the nine measures by group, each with its name spelled out: the table only has room for the codes
    var vals = measureList(r), tiles = $('insTiles'), lang = langOf(st);
    tiles.hidden = !vals;
    if (vals) patch(tiles, GROUPS.map(function (g) {
      var rows = '';
      for (var j = g.from; j < g.from + 3; j++) {
        rows += '<div class="mrow"><span class="mk">' + CODES[j] + '</span><span class="mn">' + NAMES[j] + '</span><span class="mv">' + (vals[j] === null ? '<i class="dash">—</i>' : vals[j]) + '</span><span class="mu">' + u[j] + '</span></div>';
      }
      return '<div class="mgroup"><div class="sect">' + g[lang] + '</div>' + rows + '</div>';
    }).join(''));

    var prog = $('insProg'), live = s === 'running';
    prog.hidden = !inRun(r, run);
    if (!prog.hidden) {
      var p = progPct(r, s), fill = prog.querySelector('.fill'), eta = etaText(run.eta_s);
      setText(prog.querySelector('.lbl'), live ? (run.stage_label || 'Analysing') + (run.stage_index ? ' · step ' + run.stage_index + ' of ' + (run.stage_count || 5) : '') : 'Waiting for its turn');
      setText(prog.querySelector('.p'), Math.floor(p) + '%');
      if (fill.className !== (live ? 'fill live' : 'fill part')) fill.className = live ? 'fill live' : 'fill part';
      if (fill.style.width !== p + '%') fill.style.width = p + '%';
      setText(prog.querySelector('.eta'), eta);
    }

    var files = r.files || [], fe = $('insFiles');
    fe.hidden = !files.length;
    if (files.length) patch(fe, '<div class="sect">Files</div><div class="files">' + files.map(function (f, j) {
      var mark = f.kind === 'dir' ? '<span class="dirm">' + I_DIR + '</span>' : f.exists ? '<span class="y">' + I_FILE + '</span>' : '<span class="n">' + I_NOFILE + '</span>';
      return f.exists
        ? '<div>' + mark + '<button type="button" class="link" data-action="reveal" data-path="' + esc(f.path) + '" data-key="file' + j + '" title="' + esc(f.path) + '">' + esc(f.label) + '</button></div>'
        : '<div>' + mark + '<span class="missing">' + esc(f.label) + '<span class="sr-only"> (not written yet)</span></span></div>';
    }).join('') + '</div>');

    // its actions: open, analyse again, remove; none while it is in the running session
    var foot = $('insFoot'), busy = isActive(run), inSession = inRun(r, run), html = '';
    if (!inSession) {
      var open = r.status === 'done' || ro && r.files && r.files.length;
      var again = !ro && r.status !== 'new';
      if (open || again) html += '<div class="ins-actions">' +
        (open ? '<button type="button" class="ins-foot" data-action="open-folder" data-id="' + id + '" data-key="foot-open">Open folder</button>' : '') +
        (again ? '<button type="button" class="ins-foot" data-action="again" data-id="' + id + '" data-key="foot-again"' + (busy ? ' aria-disabled="true" title="One session at a time: ' + esc(run.project_name || 'another project') + ' is being analysed"' : '') + '>Analyse again</button>' : '') + '</div>';
      if (!ro) html += '<button type="button" class="ins-remove" data-action="remove" data-id="' + id + '" data-key="foot-remove">Remove from the project</button>';
    }
    foot.hidden = !html;
    if (html) patch(foot, html);
  }

  function logPathText(p, set) {
    if (!p) return 'Written automatically for every session, as log.txt in its folder';
    var root = set.projects_dir;
    if (root && p.indexOf(root) === 0) p = baseName(root) + p.slice(root.length);
    return 'Written automatically for every session: ' + p.split(/[\\/]+/).filter(Boolean).join(' › ');
  }
  function lineKey(l) { return (l && l.t) + '\u0000' + (l && l.text); }
  function lineEl(l) {
    var d = document.createElement('div'), t = document.createElement('time'), x = document.createElement('span');
    t.textContent = fmtTime(l.t);
    x.className = LEVELS[l.level] || 'plain';
    x.textContent = l.text == null ? '' : String(l.text);
    d.appendChild(t);
    d.appendChild(x);
    return d;
  }
  var logKeys = [];
  function renderLog(st) {
    var log = st.log || {}, lines = log.lines || [], box = $('loglines'), n = logKeys.length;
    setText($('logPath'), logPathText(log.path, st.settings || {}));
    // While the log is visible, its own position decides whether it follows new lines.
    if (!$('drawer').hidden) S.follow = box.scrollHeight - box.scrollTop - box.clientHeight < 6;
    $('drawer').hidden = !S.log;
    $('logToggle').setAttribute('aria-expanded', String(S.log));
    // Lines are only appended; anything else (a trimmed log, another project in the mock) redraws.
    var grows = lines.length >= n && (n === 0 || (lineKey(lines[0]) === logKeys[0] && lineKey(lines[n - 1]) === logKeys[n - 1]));
    var top = box.scrollTop;
    if (!grows) { box.textContent = ''; logKeys = []; n = 0; }
    if (lines.length > n) {
      var frag = document.createDocumentFragment();
      for (var i = n; i < lines.length; i++) { frag.appendChild(lineEl(lines[i])); logKeys.push(lineKey(lines[i])); }
      box.appendChild(frag);
    }
    if (!S.log) return;
    if (S.follow) box.scrollTop = box.scrollHeight;
    else if (!grows) box.scrollTop = top;
  }

  var dragTimer = null;
  function renderDrop() {
    var empty = !recsOf(S.state).length;
    $('dropzone').classList.toggle('dropping', S.drag && !readonly(S.state));
    $('dropmask').hidden = !(S.drag && !empty && !readonly(S.state));
    var p = projOf(S.state);
    setText($('dropText'), 'Drop to add recordings' + (p ? ' to ' + p.name : ''));
  }

  /* ---------- the New project dialog ---------- */

  function openDialog(name, lang) {
    S.dialog = true;
    S.newLang = lang || 'en';
    $('dlgName').value = name || '';
    $('dlgError').hidden = true;
    closeMenus(false);
    if (S.state) renderDialog(S.state);
    setTimeout(function () { var i = $('dlgName'); i.focus(); i.setSelectionRange(i.value.length, i.value.length); }, 0);
  }
  function closeDialog() {
    S.dialog = false;
    if (S.state) renderDialog(S.state);
    $('newBtn').focus();
  }
  function renderDialog(st) {
    $('scrim').hidden = !S.dialog;
    $('confirmScrim').hidden = !S.confirm;
    if (S.confirm) {
      setText($('confirmTitle'), S.confirm.title);
      setText($('confirmText'), S.confirm.text);
      setText($('confirmOk'), S.confirm.ok);
      $('confirmOk').className = 'primary' + (S.confirm.danger ? ' danger' : '');
    }
    if (!S.dialog) return;
    var set = st.settings || {}, name = $('dlgName').value.trim();
    document.querySelectorAll('.segmented [data-lang]').forEach(function (b) { b.setAttribute('aria-checked', String(b.getAttribute('data-lang') === S.newLang)); });
    setText($('dlgLangHint'), (S.newLang === 'ja' ? 'Measures count morae (日本語).' : 'Measures count syllables.') + ' The language is fixed once the project is created.');
    setText($('dlgPath'), (set.projects_dir_short || set.projects_dir || '') + ((st.app || {}).sep || '/') + (name || 'Untitled project'));
    setDisabled($('dlgCreate'), !name);
  }
  function createProject() {
    var name = $('dlgName').value.trim();
    if (!name) { $('dlgName').focus(); return; }
    call('create_project', name, S.newLang).then(function (res) {
      if (res && res.ok) { S.dialog = false; S.sel = null; S.filter = null; if (S.state) render(); return; }
      var e = $('dlgError');
      e.textContent = (res && res.error) || 'The project could not be created.';
      e.hidden = false;
      $('dlgName').focus();
    });
  }

  /* ---------- confirmations, rename ---------- */

  function ask(c) {
    S.confirm = c;
    closeMenus(false);
    if (S.state) renderDialog(S.state);
    setTimeout(function () { $('confirmCancel').focus(); }, 0);
  }
  function closeConfirm() {
    S.confirm = null;
    if (S.state) renderDialog(S.state);
  }
  // The Trash (macOS) or the Recycle Bin (Windows), as the backend names it.
  function trashName(st) { return (st && st.app && st.app.trash) || 'Trash'; }
  function confirmDelete() {
    var p = projOf(S.state);
    if (!p || p.kind !== 'project' || !enabled($('deleteBtn'))) return;
    var bin = trashName(S.state);
    ask({ title: 'Delete “' + p.name + '”?', danger: true, ok: 'Move to the ' + bin,
          text: 'Its folder, with the copies of its ' + plural(recsOf(S.state).length, 'recording') + ', every session and its uf_measures.csv, moves to the ' + bin + '. You can put it back from there.',
          action: function () {
            return call('delete_project', p.id).then(function (res) {
              if (res && !res.ok) note(res.error || 'The project could not be moved to the ' + bin + '.');
            });
          } });
  }
  function confirmAgainAll() {
    var p = projOf(S.state), n = recsOf(S.state).length;
    if (!p || p.kind !== 'project' || !enabled($('againAllBtn'))) return;
    ask({ title: 'Analyse all ' + plural(n, 'recording') + ' again?', danger: false, ok: 'Analyse again',
          text: 'Each result is replaced when the new one is ready. The earlier sessions’ files stay where they are.',
          action: function () { return call('analyse_again', null); } });
  }
  function confirmRemove(r) {
    var p = projOf(S.state);
    if (!p || p.kind !== 'project') return;
    ask({ title: 'Remove “' + r.name + '” from the project?', danger: true, ok: 'Remove',
          text: 'Its row leaves the table' + (r.status === 'done' ? ' and the project’s uf_measures.csv' : '') +
                ', and the project’s copy of the file moves to the ' + trashName(S.state) + '. Your original file and the earlier sessions’ folders stay as they are.',
          action: function () {
            return call('remove', r.id).then(function (ok) {
              if (ok === false) note(r.name + ' could not be removed: it is being analysed.');
            });
          } });
  }
  function startRename() {
    var p = projOf(S.state);
    if (!p || p.kind !== 'project' || !enabled($('renameBtn'))) return;
    closeMenus(false);
    S.renaming = true;
    var i = $('renameInput');
    i.value = p.name;
    $('projName').hidden = true;
    i.hidden = false;
    i.focus();
    i.select();
  }
  function endRename(save) {
    if (!S.renaming) return;
    var i = $('renameInput'), p = projOf(S.state), name = i.value.trim();
    S.renaming = false;
    i.hidden = true;
    $('projName').hidden = false;
    if (!save || !p || !name || name === p.name) { if (S.state) renderBars(S.state); return; }
    setText($('projName'), name);
    call('rename_project', p.id, name).then(function (res) {
      if (res && !res.ok) note(res.error || 'The project could not be renamed.');
    });
  }

  /* ---------- actions ---------- */

  function select(id, focusRow) {
    S.sel = id === null ? null : String(id);
    render();
    if (focusRow && id !== null) focusRowOf(id);
  }
  function closeInspector() {
    var id = S.sel;
    S.sel = null;
    render();
    if (id !== null) focusRowOf(id);
  }
  function focusRowOf(id) {
    var tr = document.querySelector('#rows tr[data-id="' + CSS.escape(String(id)) + '"]');
    if (tr) tr.focus();
  }
  // Opening the log shows its latest lines, unless follow is false (Show in log).
  function setLog(open, follow) {
    if (open && !S.log) S.follow = follow !== false;
    S.log = open;
    if (S.state) renderLog(S.state);
  }
  function showInLog(r) {
    var lines = (S.state.log && S.state.log.lines) || [], box = $('loglines'), idx = -1, i;
    for (i = lines.length - 1; i >= 0 && idx < 0; i--) if (lines[i].level === 'error' && String(lines[i].text).indexOf(r.name) >= 0) idx = i;
    for (i = lines.length - 1; i >= 0 && idx < 0; i--) if (String(lines[i].text).indexOf(r.name) >= 0) idx = i;
    setLog(true, idx < 0);
    var el = box.children[idx];
    if (!el) return;
    box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 2);
    el.classList.add('hl');
    setTimeout(function () { el.classList.remove('hl'); }, 1600);
  }
  function doAction(a, id, el) {
    var r = id === null ? null : findRec(id);
    if (el && el.getAttribute('aria-disabled') === 'true') return;
    if (a === 'open-folder' && r) call('open_folder', r.id);
    else if (a === 'remove' && r) confirmRemove(r);
    else if (a === 'again' && r) call('analyse_again', [r.id]);
    else if (a === 'reveal') call('reveal', el.getAttribute('data-path'));
    else if (a === 'show-in-log' && r) showInLog(r);
    else if (a === 'close-inspector') closeInspector();
  }
  function selectProject(id) {
    var p = projOf(S.state);
    if (p && p.id === id) return;
    S.sel = null; S.filter = null;
    call('select_project', id);
  }

  // Copy the log: ClipboardItem takes the backend's promise inside the click (WebKit needs
  // the write to start there), then writeText, then a hidden textarea.
  function copyLog() {
    var text = call('copy_log').then(function (t) { return t == null ? '' : String(t); });
    var ok = function () { note('Log copied'); };
    var fail = function () { note('The log could not be copied. Use Save log as… instead.'); };
    var fallback = function () {
      return text.then(function (t) {
        if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(t).then(function () { return true; }, function () { return execCopy(t); });
        return execCopy(t);
      });
    };
    var first;
    try {
      if (!(navigator.clipboard && navigator.clipboard.write && window.ClipboardItem)) throw new Error('no ClipboardItem');
      first = navigator.clipboard.write([new ClipboardItem({ 'text/plain': text.then(function (t) { return new Blob([t], { type: 'text/plain' }); }) })]).then(function () { return true; }, fallback);
    } catch (e) {
      first = fallback();
    }
    // A clipboard that does not answer (a pending permission prompt) still gets a message;
    // a late success still reports, a late failure stays quiet.
    var failed = false;
    var report = function (done) {
      clearTimeout(watchdog);
      if (done) ok();
      else if (!failed) fail();
      failed = failed || !done;
    };
    var watchdog = setTimeout(function () { report(false); }, 3000);
    first.then(report, function () { report(false); });
  }
  function execCopy(t) {
    var ta = document.createElement('textarea'), ok = false;
    ta.value = t;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  // In the app, pywebview gives the dropped files' full paths only to the backend's own drop
  // handler, which adds them and then fires clausa:changed (bind).
  function addDropped(dt) {
    var files = dt && dt.files ? Array.prototype.slice.call(dt.files) : [];
    if (!files.length) return;
    var paths = files.map(function (f) { return f.pywebviewFullPath; }).filter(Boolean);
    if (paths.length) call('add_paths', paths);
    else if (mock) call('add_paths', files.map(function (f) { return f.name; }));
  }

  /* ---------- menus ---------- */

  var MENUS = [['settingsBtn', 'settingsMenu'], ['projMenuBtn', 'projMenu']];
  function closeMenus(returnFocus) {
    MENUS.forEach(function (p) {
      var btn = $(p[0]), m = $(p[1]);
      if (m.hidden) return;
      var had = m.contains(document.activeElement);
      m.hidden = true;
      btn.setAttribute('aria-expanded', 'false');
      if (returnFocus || had) btn.focus();
    });
  }
  function anyMenuOpen() { return MENUS.some(function (p) { return !$(p[1]).hidden; }); }
  function setupMenu(btnId, menuId) {
    var btn = $(btnId), m = $(menuId);
    btn.addEventListener('click', function () {
      if (!enabled(btn)) return;
      var open = m.hidden;
      closeMenus(false);
      if (!open) return;
      m.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      var first = m.querySelector('button:not([aria-disabled="true"])');
      if (first) first.focus();
    });
    m.addEventListener('keydown', function (e) {
      var keys = { ArrowDown: 1, ArrowUp: -1, Home: 0, End: 0 };
      if (!(e.key in keys)) return;
      e.preventDefault();
      var items = Array.prototype.slice.call(m.querySelectorAll('button')), i = items.indexOf(document.activeElement);
      i = e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1 : (i + keys[e.key] + items.length) % items.length;
      items[i].focus();
    });
  }

  /* ---------- events ---------- */

  function bind() {
    setupMenu('settingsBtn', 'settingsMenu');
    setupMenu('projMenuBtn', 'projMenu');
    $('changeFolderBtn').addEventListener('click', function () { if (enabled(this)) { closeMenus(true); call('choose_projects_dir'); } });
    $('sideToggle').addEventListener('click', function () { S.side = !sideOpen(); applySide(); });
    window.addEventListener('resize', function () { applySide(); });
    $('newBtn').addEventListener('click', function () { openDialog('', 'en'); });
    $('welcomeNewBtn').addEventListener('click', function () { openDialog('', 'en'); });
    $('projects').addEventListener('click', function (e) { var b = e.target.closest('[data-project]'); if (b) selectProject(b.getAttribute('data-project')); });
    $('filters').addEventListener('click', function (e) {
      var b = e.target.closest('[data-filter]');
      if (!b) return;
      var k = b.getAttribute('data-filter') || null;
      S.filter = k && S.filter !== k ? k : null;
      if (S.state) render();
    });
    $('showAllBtn').addEventListener('click', function () { S.filter = null; if (S.state) render(); });
    $('openFolderBtn').addEventListener('click', function () { call('open_project_folder'); });
    $('csvBtn').addEventListener('click', function () { if (enabled(this)) call('open_csv'); });
    $('addBtn').addEventListener('click', function () { call('add_files'); });
    $('chooseBtn').addEventListener('click', function () { call('add_files'); });
    $('addMoreBtn').addEventListener('click', function () { call('add_files'); });
    $('startBtn').addEventListener('click', function () { if (enabled(this)) call('start'); });
    $('stopBtn').addEventListener('click', function () { if (enabled(this)) call('stop'); });
    $('logToggle').addEventListener('click', function () { setLog(!S.log); });
    $('drawerClose').addEventListener('click', function () { setLog(false); $('logToggle').focus(); });
    $('copyLogBtn').addEventListener('click', copyLog);
    $('saveLogBtn').addEventListener('click', function () { call('save_log_as'); });

    // the project: rename (the title, or the menu), analyse all again, delete; each asks first
    $('projName').addEventListener('click', function () { if (!readonly(S.state)) startRename(); });
    $('renameBtn').addEventListener('click', startRename);
    $('againAllBtn').addEventListener('click', confirmAgainAll);
    $('deleteBtn').addEventListener('click', confirmDelete);
    $('renameInput').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); endRename(true); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); endRename(false); }
    });
    $('renameInput').addEventListener('blur', function () { endRename(true); });
    $('confirm').addEventListener('submit', function (e) {
      e.preventDefault();
      var c = S.confirm;
      closeConfirm();
      if (c && c.action) c.action();
    });
    $('confirmCancel').addEventListener('click', closeConfirm);
    $('confirmScrim').addEventListener('mousedown', function (e) { if (e.target === this) closeConfirm(); });

    // the New project dialog: Enter creates, Escape (below) or Cancel closes
    $('dialog').addEventListener('submit', function (e) { e.preventDefault(); if (enabled($('dlgCreate'))) createProject(); });
    $('dlgCancel').addEventListener('click', closeDialog);
    $('dlgName').addEventListener('input', function () { $('dlgError').hidden = true; if (S.state) renderDialog(S.state); });
    document.querySelectorAll('.segmented [data-lang]').forEach(function (b) {
      b.addEventListener('click', function () { S.newLang = b.getAttribute('data-lang'); if (S.state) renderDialog(S.state); });
    });
    $('scrim').addEventListener('mousedown', function (e) { if (e.target === this) closeDialog(); });

    document.addEventListener('click', function (e) {
      var t = e.target;
      if (!t.closest('.menu-wrap')) closeMenus(false);
      var b = t.closest('[data-action]');
      if (b) { doAction(b.getAttribute('data-action'), b.getAttribute('data-id'), b); return; }
      var tr = t.closest('#rows tr');
      if (tr) select(tr.getAttribute('data-id'), false);
    });
    $('rows').addEventListener('keydown', function (e) {
      var tr = e.target;
      if (tr.tagName !== 'TR') return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(tr.getAttribute('data-id'), true); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        var next = e.key === 'ArrowDown' ? tr.nextElementSibling : tr.previousElementSibling;
        while (next && next.hidden) next = e.key === 'ArrowDown' ? next.nextElementSibling : next.previousElementSibling;
        if (next) next.focus();
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      if (S.confirm) { e.preventDefault(); closeConfirm(); }
      else if (S.dialog) { e.preventDefault(); closeDialog(); }
      else if (anyMenuOpen()) { e.preventDefault(); closeMenus(true); }
      else if (S.sel !== null) closeInspector();
    });

    // Drag and drop over the table or the empty state. The highlight ends when the pointer
    // leaves the zone, or when dragover stops arriving (the browser sends it every few
    // hundred milliseconds while a drag hovers).
    var dz = $('dropzone');
    var hasFiles = function (e) { return !!(e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], 'Files') >= 0); };
    var dragOn = function () {
      if (!S.drag) { S.drag = true; renderDrop(); }
      clearTimeout(dragTimer);
      dragTimer = setTimeout(dragOff, 700);
    };
    var dragOff = function () { clearTimeout(dragTimer); if (S.drag) { S.drag = false; renderDrop(); } };
    dz.addEventListener('dragenter', function (e) { if (hasFiles(e)) { e.preventDefault(); dragOn(); } });
    dz.addEventListener('dragover', function (e) { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; dragOn(); } });
    dz.addEventListener('dragleave', function (e) {
      var b = dz.getBoundingClientRect();
      if (e.clientX <= b.left || e.clientX >= b.right || e.clientY <= b.top || e.clientY >= b.bottom) dragOff();
    });
    dz.addEventListener('drop', function (e) { e.preventDefault(); dragOff(); addDropped(e.dataTransfer); });
    window.addEventListener('clausa:changed', function () { refresh(); });
    // Elsewhere a drop must not make the web view open the file as a page.
    window.addEventListener('dragover', function (e) { if (!dz.contains(e.target)) { e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = 'none'; } });
    window.addEventListener('drop', function (e) { if (!dz.contains(e.target)) e.preventDefault(); });
  }

  /* ---------- mock backend: the app's API with example projects ---------- */

  function createMock(view, preview) {
    var STAGES = ['Speech recognition', 'Word alignment', 'Clauses', 'Pauses', 'Measures'];
    var STAGE_S = [5, 4, 2.5, 1.5, 1];  // simulated seconds per recording and stage (about 14 s each in a batch)
    var DIR = '~/Documents/Clausa projects', OLD = '~/Documents/Clausa results';
    var now = new Date(), TODAY = now.getFullYear() + '-' + pad2(now.getMonth() + 1) + '-' + pad2(now.getDate());
    var ERRORS = {
      1: { text: 'No speech was found in this recording, so it was skipped. Check that the file holds the learner’s speech.', log: 'no speech found in the recording; skipped' },
      2: { text: 'Word alignment failed for this recording: MFA aligned none of its words. Its measures are not shown; the log has the details.', log: 'word alignment failed: MFA aligned none of its 214 words' }
    };

    // Measures that agree with each other: PR = MCPR + ECPR as shown; 1/SR = 1/AR + pause time per
    // syllable; MPD = that pause time / PR; MLR = syllables / (pauses + 1). The inputs come from a
    // generator seeded by the file name, with more pauses at lower speed.
    function hash(s) { var h = 2166136261; for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h; }
    function rng(seed) { return function () { seed = (seed + 0x6d2b79f5) >>> 0; var t = seed; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
    function r1(x) { return Math.round(x * 10 + 1e-9) / 10; }
    function r2(x) { return Math.round(x * 100 + 1e-9) / 100; }
    function sample(name, lang) {
      var r = rng(hash(name)), f = r(), en = lang !== 'ja', mix = function (w) { return w * f + (1 - w) * r(); };
      var ar = r2(en ? 3.4 + 1.45 * mix(0.75) : 4.6 + 1.5 * mix(0.75));
      var mk = Math.round(en ? 110 - 55 * mix(0.7) : 85 - 45 * mix(0.7)), ek = Math.round(en ? 70 - 25 * mix(0.6) : 58 - 22 * mix(0.6));
      var mcpd = r2(0.9 - 0.45 * mix(0.6)), ecpd = r2(mcpd + 0.25 + 0.2 * r()), dur = 70 + Math.floor(r() * 169);
      var mcpr = mk / 1000, ecpr = ek / 1000, pr = (mk + ek) / 1000, p = mcpr * mcpd + ecpr * ecpd, sr = 1 / (1 / ar + p), syl = sr * dur;
      var m = [ar, r2(sr), r1(syl / (pr * syl + 1)), mcpr, ecpr, pr, mcpd, ecpd, r2(p / pr)], out = {};
      CODES.forEach(function (k, j) { out[k] = m[j]; });
      return { dur: dur, m: out };
    }
    function iso(stamp) { return stamp.slice(0, 10) + 'T' + stamp.slice(11, 13) + ':' + stamp.slice(14, 16) + ':00'; }  // '2026-10-15 13.40' -> ISO
    function daysAgo(n, hm) { var d = new Date(now.getTime() - n * 86400000); return d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + (hm || '10.00'); }

    // The example projects, newest first; session stamps relative to today so the dates read the
    // same whenever the mock is opened. An entry: [file, session index it was analysed in, or
    // 'new', or 'fail:<stage>' (failed in the last session)].
    var SPEC = [
      { id: 'Picture story · 2025 cohort', language: 'en', sessions: [daysAgo(20, '10.12'), daysAgo(8, '13.40')], recs: [
        ['P01_picture_story.wav', 0], ['P02_picture_story.wav', 0], ['P04_picture_story.wav', 0], ['P05_picture_story.m4a', 0],
        ['P07_picture_story.wav', 1], ['P09_picture_story.wav', 1], ['P10_picture_story.wav', 1], ['P12_picture_story.wav', 1], ['P14_picture_story.wav', 1],
        ['P16_picture_story.wav', 'new', 2], ['P17_picture_story.m4a', 'new'], ['P19_picture_story.wav', 'new']] },
      { id: 'Spring 2026 retest', language: 'en', sessions: [daysAgo(225, '09.30')], recs: [
        ['R02_retest.wav', 0], ['R03_retest.wav', 0], ['R05_retest.wav', 0], ['R06_retest.wav', 0], ['R08_retest.wav', 0], ['R09_retest.wav', 0],
        ['R11_retest.wav', 'new'], ['R14_retest.wav', 'new'], ['R15_retest.mp3', 'new']] },
      { id: 'Story retell · JA learners', language: 'ja', sessions: [daysAgo(4, '16.02')], recs: [
        ['J02_story_retell.wav', 0], ['J04_story_retell.wav', 0], ['J06_story_retell.wav', 0], ['J09_story_retell.wav', 0],
        ['J10_story_retell.m4a', 0], ['J13_story_retell.wav', 0], ['J15_story_retell.wav', 0], ['J18_story_retell.wav', 0]] },
      { id: 'I-JAS interviews', language: 'ja', sessions: [daysAgo(32, '10.05'), daysAgo(25, '11.20')], recs: [] },
      { id: 'Pilot study (EN)', language: 'en', sessions: [daysAgo(51, '15.45')], recs: [] },
      { id: 'Picture story · 2024 cohort', language: 'en', sessions: [daysAgo(343, '10.05'), daysAgo(336, '14.30')], recs: [] }
    ];
    var i;
    for (i = 1; i <= 18; i++) SPEC[3].recs.push(['IJ' + pad2(i) + '_interview.wav', i === 7 ? 'fail:2' : i === 13 ? 'fail:1' : i <= 10 ? 0 : 1]);
    ['01', '03', '04', '06', '08', '11'].forEach(function (n) { SPEC[4].recs.push(['PS' + n + '_picture_story.wav', 0]); });
    for (i = 1; i <= 16; i++) SPEC[5].recs.push(['P' + pad2(i + (i > 9 ? 2 : 0)) + '_picture_story' + (i % 7 === 3 ? '.m4a' : '.wav'), i <= 9 ? 0 : 1]);
    var EARLIER = [{ stamp: daysAgo(40, '14.05'), language: 'en', files: ['speaker01', 'speaker03', 'speaker04'] },
                   { stamp: daysAgo(41, '09.20'), language: 'ja', files: ['CCH16-ST1'] }];

    var X = {}, clock = 14 * 3600 + 5 * 60 + 2, sim = null, stopTimer = null, noticeN = 0;
    var D = {
      app: { version: '1.0.1', device: preview ? 'Preview without the app · example data' : 'Mac GPU · MLX' },
      settings: { projects_dir: DIR, projects_dir_short: DIR },
      projects: [], earlier: [], current: null, run: idleRun(), notice: { id: 0, text: '' }
    };
    function idleRun() { return { status: 'idle', project_id: null, project_name: '', session_dir: null, csv_path: null, started: '', elapsed_s: 0, eta_s: null, progress: 0, stage_index: 0, stage_count: STAGES.length, stage_label: '', current_file: null, current_name: '', done_count: 0, failed_count: 0, total: 0 }; }
    function hms(s) { s = Math.floor(s); return pad2(Math.floor(s / 3600) % 24) + ':' + pad2(Math.floor(s / 60) % 60) + ':' + pad2(s % 60); }
    function proj(id) { for (var j = 0; j < D.projects.length; j++) if (D.projects[j].id === id) return D.projects[j]; return null; }
    function cur() { return proj(D.current); }
    function rec(id) { var p = sim ? sim.p : cur(); if (!p) return null; for (var k = 0; k < p.recordings.length; k++) if (p.recordings[k].id === id) return p.recordings[k]; return null; }
    function busy() { return D.run.status === 'running' || D.run.status === 'stopping'; }
    function plog(p, text, level, t) { p.log.lines.push({ t: t || hms(clock), text: text, level: level || 'info' }); }
    function notice(text) { D.notice = { id: ++noticeN, text: text }; }

    function addRec(p, name, opts) {
      var smp = sample(p.id + '/' + name, p.language), id = name.replace(/\.[^.]+$/, '');
      var r = { id: id, name: name, duration_s: smp.dur, status: 'new', progress: 0, textgrids: { words: 'no', clauses: 'no' }, measures: null, files: [], error: null, failed_stage: null, added: (opts && opts.added) || new Date().toISOString(), analysed: null };
      X[p.id + '/' + id] = { measures: smp.m, fail: opts && opts.fail ? Number(opts.fail) : 0, session: null };
      p.recordings.push(r);
      return r;
    }
    function x(p, r) { return X[p.id + '/' + r.id]; }
    function finishRec(p, r, session) {
      r.status = 'done'; r.progress = 100; r.textgrids = { words: 'yes', clauses: 'yes' }; r.measures = x(p, r).measures;
      r.error = null; r.failed_stage = null; x(p, r).session = session; r.analysed = iso(session);
    }
    function failRec(p, r, stage, session) {
      var e = ERRORS[stage] || ERRORS[2];
      r.status = 'failed'; r.error = e.text; r.failed_stage = STAGES[stage - 1]; r.progress = (stage - 1) * 20 + 12; r.measures = null;
      r.textgrids = { words: stage > 2 ? 'yes' : 'no', clauses: 'no' }; x(p, r).session = session; r.analysed = iso(session);
    }
    function build(spec) {
      var p = { id: spec.id, name: spec.id, language: spec.language, sessions: spec.sessions.slice(), recordings: [], log: { path: null, lines: [] } };
      D.projects.push(p);
      spec.recs.forEach(function (e) {
        var st = e[1], first = spec.sessions[0], added = typeof st === 'number' ? iso(spec.sessions[st]) : st === 'new' ? new Date().toISOString() : iso(first);
        var r = addRec(p, e[0], { fail: e[2], added: added });
        if (typeof st === 'number') finishRec(p, r, spec.sessions[st]);
        else if (String(st).indexOf('fail:') === 0) failRec(p, r, Number(st.slice(5)), spec.sessions[spec.sessions.length - 1]);
      });
      // the log of the last session, as the drawer would show it
      var last = spec.sessions[spec.sessions.length - 1], t0 = Number(last.slice(11, 13)) * 3600 + Number(last.slice(14, 16)) * 60;
      var inLast = p.recordings.filter(function (r) { return x(p, r).session === last; });
      if (inLast.length) {
        p.log.path = DIR + '/' + p.name + '/sessions/' + last + '/log.txt';
        plog(p, 'Session started: ' + plural(inLast.length, 'recording') + ' of ' + p.name + ', ' + LANGS[p.language], 'info', hms(t0));
        inLast.forEach(function (r, j) {
          var t = hms(t0 + 70 + j * 14);
          if (r.status === 'failed') plog(p, r.name + ': ' + (ERRORS[STAGES.indexOf(r.failed_stage) + 1] || ERRORS[2]).log, 'error', t);
          else plog(p, 'Processed: ' + r.id + '.TextGrid', 'info', t);
        });
        var nf = inLast.filter(function (r) { return r.status === 'failed'; }).length;
        plog(p, 'Session finished: ' + (inLast.length - nf) + ' of ' + inLast.length + ' analysed' + (nf ? ', ' + nf + ' failed' : ''), nf ? 'error' : 'ok', hms(t0 + 80 + inLast.length * 14));
      }
      return p;
    }

    function counts() {
      var ids = sim ? sim.all : [];
      D.run.done_count = ids.filter(function (id) { return rec(id) && rec(id).status === 'done'; }).length;
      D.run.failed_count = ids.filter(function (id) { return rec(id) && rec(id).status === 'failed'; }).length;
    }
    function pct() {
      if (!sim) return 0;
      var t = 0;
      sim.all.forEach(function (id) { var r = rec(id); t += r.status === 'failed' || r.status === 'done' ? 100 : r.progress; });
      return Math.round(t / sim.all.length * 10) / 10;
    }
    function sync() {
      var r = rec(sim.order[sim.j]), left = (STAGE_S[sim.stage - 1] - sim.t) + STAGE_S[sim.stage - 1] * (sim.order.length - sim.j - 1);
      for (var s = sim.stage + 1; s <= STAGES.length; s++) left += STAGE_S[s - 1] * sim.order.length;
      D.run.stage_index = sim.stage;
      D.run.stage_label = STAGES[sim.stage - 1];
      D.run.current_file = r ? r.id : null;
      D.run.current_name = r ? r.name : '';
      D.run.eta_s = Math.max(0, Math.round(left));
      D.run.progress = pct();
      counts();
    }
    function startSession(p, recs) {
      if (!p || busy() || !recs.length) return false;
      var stamp = TODAY + ' ' + pad2(Math.floor(clock / 3600)) + '.' + pad2(Math.floor(clock / 60) % 60);
      D.run = idleRun();
      D.run.status = 'running';
      D.run.project_id = p.id;
      D.run.project_name = p.name;
      D.run.session_dir = DIR + '/' + p.name + '/sessions/' + stamp;
      D.run.started = pad2(Math.floor(clock / 3600)) + ':' + pad2(Math.floor(clock / 60) % 60);
      D.run.total = recs.length;
      p.log = { path: D.run.session_dir + '/log.txt', lines: [] };
      recs.forEach(function (r) {
        r.before = { status: r.status, measures: r.measures, analysed: r.analysed, error: r.error, failed_stage: r.failed_stage, textgrids: r.textgrids };
        r.status = 'running'; r.progress = 0; r.measures = null; r.error = null; r.failed_stage = null;
        r.textgrids = { words: 'no', clauses: 'no' };
      });
      var ids = recs.map(function (r) { return r.id; }).sort();
      sim = { p: p, stamp: stamp, all: ids.slice(), order: ids.slice(), stage: 1, j: 0, t: 0 };
      plog(p, 'Session started: ' + plural(ids.length, 'recording') + ' of ' + p.name + ', ' + LANGS[p.language]);
      sync();
      return true;
    }
    function finishRun() {
      sim.order.forEach(function (id) { var r = rec(id); finishRec(sim.p, r, sim.stamp); r.before = null; });
      sim.all.forEach(function (id) { var r = rec(id); if (r) r.before = null; });
      counts();
      D.run.status = 'done'; D.run.eta_s = 0; D.run.current_file = null; D.run.current_name = ''; D.run.stage_label = ''; D.run.stage_index = 0; D.run.progress = 100;
      sim.p.sessions.push(sim.stamp);
      plog(sim.p, 'Session finished: ' + D.run.done_count + ' of ' + D.run.total + ' analysed' + (D.run.failed_count ? ', ' + D.run.failed_count + ' failed' : '') + ' in ' + fmtElapsed(D.run.elapsed_s), 'ok');
      sim = null;
    }
    function step(dt) {
      if (!sim || D.run.status !== 'running') return;
      clock += dt;
      D.run.elapsed_s += dt;
      sim.t += dt;
      var s = sim.stage, id = sim.order[sim.j], r = rec(id), f = Math.min(1, sim.t / STAGE_S[s - 1]);
      r.progress = Math.round(((s - 1) * 20 + 20 * f) * 10) / 10;
      if (s === 2 && r.textgrids.words !== 'yes') r.textgrids.words = 'wip';
      if (s === 3 && r.textgrids.clauses !== 'yes') r.textgrids.clauses = 'wip';
      var next = false;
      if (x(sim.p, r).fail === s && f >= 0.6) {
        failRec(sim.p, r, s, sim.stamp);
        plog(sim.p, r.name + ': ' + ERRORS[s].log, 'error');
        sim.order.splice(sim.j, 1);
        next = sim.j >= sim.order.length;
        sim.t = 0;
      } else if (f >= 1) {
        r.progress = s * 20;
        if (s === 2) r.textgrids.words = 'yes';
        if (s === 3) r.textgrids.clauses = 'yes';
        plog(sim.p, STAGES[s - 1] + ': ' + r.id + ' (' + Math.round(sim.t) + ' s)', 'ok');
        sim.j++;
        sim.t = 0;
        next = sim.j >= sim.order.length;
      }
      if (next) {
        sim.j = 0;
        sim.stage++;
        if (sim.stage > STAGES.length || !sim.order.length) { finishRun(); return; }
      }
      sync();
    }
    function fastForward(until) { while (sim && D.run.status === 'running' && !until()) step(0.25); }
    function stopRun() {
      if (D.run.status !== 'running') return false;
      D.run.status = 'stopping';
      stopTimer = setTimeout(function () {
        stopTimer = null;
        var back = 0;
        sim.all.forEach(function (id) {
          var r = rec(id);
          if (r && r.status === 'running') {
            var b = r.before || {};
            r.status = b.status || 'new'; r.measures = b.measures || null; r.analysed = b.analysed || null; r.error = b.error || null; r.failed_stage = b.failed_stage || null;
            r.textgrids = b.textgrids || { words: 'no', clauses: 'no' }; r.progress = r.status === 'done' ? 100 : 0; back++;
          }
          if (r) r.before = null;
        });
        counts();
        D.run.status = 'stopped'; D.run.current_file = null; D.run.current_name = ''; D.run.eta_s = null; D.run.stage_label = ''; D.run.stage_index = 0;
        plog(sim.p, 'Stopped by the user; ' + plural(back, 'recording') + ' back to how they were');
        sim = null;
      }, 700);
      return true;
    }
    function snapshot() {
      var p = cur(), runningId = busy() ? D.run.project_id : null, out = {
        app: D.app, settings: D.settings, run: D.run, notice: D.notice,
        projects: D.projects.map(function (q) {
          return { id: q.id, name: q.name, language: q.language, count: q.recordings.length, failed: q.recordings.filter(function (r) { return r.status === 'failed'; }).length,
                   status: q.id === runningId ? 'running' : 'idle', progress: q.id === runningId ? D.run.progress : 0 };
        }),
        earlier: D.earlier.map(function (e) { return { id: e.id, name: e.name, language: e.language, count: e.recordings.length }; }),
        project: null, recordings: [], log: { path: null, lines: [] }
      };
      var e = D.earlier.filter(function (q) { return q.id === D.current; })[0];
      if (e) {
        out.project = { id: e.id, name: e.name, language: e.language, kind: 'earlier', dir: OLD + '/' + e.name, created: '', sessions: 1, last_session: iso(e.name), csv: true };
        out.recordings = e.recordings;
        out.log = { path: OLD + '/' + e.name + '/log.txt', lines: [] };
      } else if (p) {
        var ended = p.sessions;
        out.project = { id: p.id, name: p.name, language: p.language, kind: 'project', dir: DIR + '/' + p.name, created: '', sessions: ended.length,
                        last_session: ended.length ? iso(ended[ended.length - 1]) : null, csv: p.recordings.some(function (r) { return r.status === 'done'; }) };
        out.recordings = p.recordings.map(function (r) {
          var stem = r.id, sess = x(p, r).session || (sim && sim.p === p && sim.all.indexOf(r.id) >= 0 ? sim.stamp : null), dir = DIR + '/' + p.name + '/sessions/' + (sess || '') + '/';
          r.files = r.status === 'new' ? [] : [
            { label: 'Words TextGrid', path: dir + 'asr/textgrids_clean/' + stem + '.TextGrid', exists: r.textgrids.words === 'yes', kind: 'file' },
            { label: 'Clauses TextGrid', path: dir + 'clauses/' + stem + '.TextGrid', exists: r.textgrids.clauses === 'yes', kind: 'file' }
          ];
          if (sess && r.status !== 'running') r.files.push({ label: 'Session ' + sess, path: dir, exists: true, kind: 'dir' });
          return r;
        });
        out.log = p.log;
      }
      return JSON.parse(JSON.stringify(out, function (k, v) { return k === 'before' ? undefined : v; }));
    }

    // the example states
    if (view !== 'welcome') {
      SPEC.forEach(build);
      D.current = proj(params.get('project')) ? params.get('project') : SPEC[0].id;
    }
    EARLIER.forEach(function (e) {
      var q = { id: 'earlier:' + OLD + '/' + e.stamp, name: e.stamp, language: e.language, recordings: [] };
      e.files.forEach(function (f, j) {
        var smp = sample(q.id + f, e.language), ok = j !== 1 || e.files.length < 3;
        q.recordings.push({ id: f, name: f + '.wav', duration_s: smp.dur, status: ok ? 'done' : 'failed', progress: ok ? 100 : 0, textgrids: { words: ok ? 'yes' : 'no', clauses: ok ? 'yes' : 'no' },
                            measures: ok ? smp.m : null, files: [], error: ok ? null : 'This analysis has no measures for this recording. Its log.txt has the details.',
                            failed_stage: null, added: '', analysed: iso(e.stamp) });
      });
      D.earlier.push(q);
    });
    if (params.get('project') && params.get('project').indexOf('earlier') === 0 && D.earlier[0]) D.current = D.earlier[0].id;
    if (view === 'running') {
      // the 2025 cohort, third session: three recordings in it, working stage by stage (P16 fails at
      // word alignment); P20 added after the session started, so it is New
      D.current = SPEC[0].id;
      var p0 = cur();
      startSession(p0, p0.recordings.filter(function (r) { return r.status === 'new'; }));
      fastForward(function () { var r = rec(sim.order[sim.j]); return sim.stage === 3 && r.name === 'P17_picture_story.m4a' && r.progress >= 52; });
      addRec(p0, 'P20_picture_story.mp3');
    }
    var ticker = params.get('still') === '1' ? null : setInterval(function () { step(0.25); }, 250);

    var mockApi = {
      get_state: function () { return Promise.resolve(snapshot()); },
      select_project: function (id) {
        if (!proj(id) && !D.earlier.some(function (e) { return e.id === id; })) return Promise.resolve(false);
        D.current = id; return Promise.resolve(true);
      },
      create_project: function (name, language) {
        name = String(name || '').trim();
        if (!name || /[\/:]/.test(name)) return Promise.resolve({ ok: false, error: 'A project name cannot contain / or : (it becomes the name of its folder).' });
        if (D.projects.some(function (q) { return q.name.toLowerCase() === name.toLowerCase(); })) return Promise.resolve({ ok: false, error: 'There is already a project or folder called ' + name + ' in ' + DIR + '.' });
        D.projects.unshift({ id: name, name: name, language: language === 'ja' ? 'ja' : 'en', sessions: [], recordings: [], log: { path: null, lines: [] } });
        D.current = name;
        return Promise.resolve({ ok: true, id: name });
      },
      rename_project: function (id, name) {
        var p = proj(id); name = String(name || '').trim();
        if (!p) return Promise.resolve({ ok: false, error: 'That project is not in the list any more.' });
        if (busy() && D.run.project_id === id) return Promise.resolve({ ok: false, error: 'The project is being analysed: rename it when the session has ended.' });
        if (!name || /[\/:]/.test(name)) return Promise.resolve({ ok: false, error: 'A project name cannot contain / or : (it becomes the name of its folder).' });
        if (D.projects.some(function (q) { return q !== p && q.name.toLowerCase() === name.toLowerCase(); })) return Promise.resolve({ ok: false, error: 'There is already a project or folder called ' + name + '.' });
        Object.keys(X).forEach(function (k) { if (k.indexOf(p.id + '/') === 0) { X[name + k.slice(p.id.length)] = X[k]; } });
        p.id = p.name = name; D.current = name;
        return Promise.resolve({ ok: true, id: name });
      },
      delete_project: function (id) {
        var p = proj(id);
        if (!p) return Promise.resolve({ ok: false, error: 'That project is not in the list any more.' });
        if (busy() && D.run.project_id === id) return Promise.resolve({ ok: false, error: 'The project is being analysed: stop the session first.' });
        D.projects.splice(D.projects.indexOf(p), 1);
        if (D.current === id) D.current = D.projects.length ? D.projects[0].id : null;
        return Promise.resolve({ ok: true });
      },
      add_files: function () {
        var p = cur(); if (!p) return Promise.resolve({ added: 0, refused: [] });
        var have = p.recordings.map(function (r) { return r.id; }), n = 0;
        for (var k = 21; n < 3 && k < 99; k++) {
          var name = (p.language === 'ja' ? 'J' : 'P') + pad2(k) + (p.language === 'ja' ? '_story_retell.wav' : '_picture_story.wav');
          if (have.indexOf(name.replace(/\.[^.]+$/, '')) < 0) { addRec(p, name); n++; }
        }
        return Promise.resolve({ added: n, refused: [] });
      },
      add_paths: function (paths) {
        var p = cur(), added = 0, refused = [];
        if (!p) { notice('Create or choose a project first, then add recordings to it.'); return Promise.resolve({ added: 0, refused: ['no project'] }); }
        (paths || []).forEach(function (q) {
          var name = baseName(q), id = name.replace(/\.[^.]+$/, '');
          if (p.recordings.some(function (r) { return r.id.toLowerCase() === id.toLowerCase(); })) refused.push(name + ': this project already has a recording named ' + id + '; rename the file to add it');
          else { addRec(p, name); added++; }
        });
        if (refused.length) notice(refused[0]);
        return Promise.resolve({ added: added, refused: refused });
      },
      remove: function (id) {
        var p = cur(), r = p && p.recordings.filter(function (q) { return q.id === id; })[0];
        if (!r || (busy() && sim && sim.p === p && sim.all.indexOf(r.id) >= 0)) return Promise.resolve(false);
        p.recordings.splice(p.recordings.indexOf(r), 1);
        return Promise.resolve(true);
      },
      start: function () { var p = cur(); return Promise.resolve(!!p && startSession(p, p.recordings.filter(function (r) { return r.status === 'new' || r.status === 'failed'; }))); },
      analyse_again: function (ids) { var p = cur(); return Promise.resolve(!!p && startSession(p, p.recordings.filter(function (r) { return !ids || ids.indexOf(r.id) >= 0; }))); },
      stop: function () { return Promise.resolve(stopRun()); },
      choose_projects_dir: function () { return Promise.resolve(DIR); },
      open_project_folder: function () { return Promise.resolve(true); },
      open_csv: function () { return Promise.resolve(true); },
      open_folder: function () { return Promise.resolve(true); },
      reveal: function () { return Promise.resolve(true); },
      copy_log: function () { var p = cur(); return Promise.resolve(p ? p.log.lines.map(function (l) { return l.t + '  ' + l.text; }).join('\n') + '\n' : ''); },
      save_log_as: function () { return Promise.resolve(true); }
    };
    return {
      api: mockApi,
      dispose: function () { clearInterval(ticker); clearTimeout(stopTimer); }
    };
  }

  bind();
  connect();
})();
