/* Clausa First start window: shows the state that runner/first_start_window.py folds from First
   start's events (runner/first_start_events.py, fold). The page keeps no state of its own: it
   polls Api.get_state() and calls retry(), open_app() and show_log().
   ?mock=<name> shows an example state from window.FIRST_START_MOCKS instead (screenshots);
   &still=1 stops the animations. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);
  var api = null;

  // ---------- numbers ----------
  function size(b) {
    if (b == null) return '';
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
    if (b >= 1e6) return Math.round(b / 1e6) + ' MB';
    return Math.max(1, Math.round(b / 1e3)) + ' KB';
  }
  function of(done, total) {  // "3.1 of 7.2 GB": both in the unit of the total
    var u = total >= 1e9 ? [1e9, ' GB', 1] : [1e6, ' MB', 0];
    return (done / u[0]).toFixed(u[2]) + ' of ' + (total / u[0]).toFixed(u[2]) + u[1];
  }
  function speed(bps) {
    if (bps >= 1e9) return (bps / 1e9).toFixed(1) + ' GB/s';
    return bps >= 1e6 ? (bps / 1e6).toFixed(bps >= 1e7 ? 0 : 1) + ' MB/s' : Math.round(bps / 1e3) + ' KB/s';
  }
  function left(s) {
    if (s < 45) return 'less than a minute left';
    if (s < 90) return 'about a minute left';
    if (s < 3600) return 'about ' + Math.round(s / 60) + ' min left';
    var h = Math.floor(s / 3600), m = Math.round((s - h * 3600) / 60);
    return 'about ' + h + ' h' + (m ? ' ' + m + ' min' : '') + ' left';
  }
  function pct(done, total) { return total ? Math.min(100, Math.floor(100 * done / total)) : 0; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ---------- icons ----------
  var ICON = {
    done: '<svg class="st-ico done" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="10"/><path d="M6 10.2l2.6 2.6L14 7.4" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    running: '<svg class="st-ico running" viewBox="0 0 20 20" aria-hidden="true"><circle class="ring" cx="10" cy="10" r="8" fill="none" stroke-width="2.5"/><path class="arc" d="M10 2a8 8 0 0 1 8 8" fill="none" stroke-width="2.5" stroke-linecap="round"/></svg>',
    waiting: '<svg class="st-ico waiting" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="none" stroke-width="1.5"/></svg>',
    skipped: '<svg class="st-ico skipped" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="10"/><path d="M6.5 10h7" fill="none" stroke-width="1.8" stroke-linecap="round"/></svg>',
    failed: '<svg class="st-ico failed" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="10"/><path d="M10 5.6v5.2" fill="none" stroke-width="2" stroke-linecap="round"/><circle class="dot" cx="10" cy="14.2" r="1.2"/></svg>'
  };
  ICON.checking = ICON.running;
  ICON.stopped = ICON.waiting.replace('waiting', 'stopped');
  var SAY = { done: 'done', running: 'in progress', checking: 'checking', waiting: 'to do', skipped: 'not needed', failed: 'stopped', stopped: 'stopped' };

  function again(platform) { return platform === 'windows' ? 'start Clausa again' : 'double-click First start again'; }
  function appName(platform) { return platform === 'windows' ? 'Clausa.exe' : 'Clausa.app'; }

  // ---------- one step ----------
  function itemRow(it) {
    var n = it.status === 'done' ? (it.total != null ? size(it.total) : 'Done')
      : it.status === 'checking' ? 'Checking…'
      : (it.status === 'running' || it.status === 'stopped') ? (it.total != null ? of(it.done, it.total) : 'Working…')
      : (it.total != null ? size(it.total) : '');
    var word = !/^[0-9]/.test(n);  // a word, not a number: not in the mono font
    var bar = (it.status === 'running' || it.status === 'stopped') && it.total
      ? '<div class="track"><div class="fill ' + (it.status === 'stopped' ? 'bad' : 'live') + '" style="width:' + pct(it.done, it.total) + '%"></div></div>' : '';
    return '<li class="item ' + it.status + '">' + (ICON[it.status] || ICON.waiting).replace('st-ico', 'st-ico i-ico') +
      '<span class="i-label">' + esc(it.label) + '<span class="sr-only">, ' + SAY[it.status] + '</span></span>' +
      '<span class="i-bar">' + bar + '</span><span class="i-nums' + (word ? ' word' : '') + '">' + esc(n) + '</span></li>';
  }

  function stepCard(s, st) {
    var open = s.status === 'running' || s.status === 'failed';
    var side = s.status === 'done' ? (s.note || (s.total ? size(s.total) : 'Done'))
      : s.status === 'skipped' ? 'Not needed'
      : s.status === 'failed' ? 'Stopped'
      : s.status === 'running' ? '' : (s.total ? size(s.total) : '');
    var out = '<li class="step ' + s.status + (open ? ' current' : '') + '"' + (s.status === 'running' ? ' aria-current="step"' : '') + '>' +
      '<div class="step-row">' + (ICON[s.status] || ICON.waiting) +
      '<span class="step-label">' + esc(s.label) + '<span class="sr-only">, ' + SAY[s.status] + '</span></span>' +
      '<span class="step-side' + (/^[0-9]/.test(side) ? '' : ' word') + '">' + esc(side) + '</span></div>';
    var note = s.note && s.status === 'skipped' ? '<p class="step-note">' + esc(s.note) + '</p>' : '';
    if (!open) return out + (note ? '<div class="step-detail">' + note + '</div>' : '') + '</li>';

    // A step without a total (an item that is not counted) that is busy with a counted item (Windows:
    // the GPU libraries in Preparing) shows that item's progress; otherwise a bar that moves.
    var counted = s.total != null, nums = [];
    var busy = s.items.filter(function (it) { return it.status === 'running' || it.status === 'checking'; });
    var big = !counted && busy.filter(function (it) { return it.total != null; })[0];
    var done = counted ? s.done : big ? big.done : 0, total = counted ? s.total : big ? big.total : null;
    if (big) nums.push([pct(done, total) + '%', 'pct inline'], [big.label, 'txt lab'], [of(done, total)]);
    else if (total != null) nums.push([pct(done, total) + '%', 'pct'], [of(done, total)]);
    if (s.status === 'running') {
      if (s.speed) nums.push([speed(s.speed)]);
      if (s.eta != null) nums.push([left(s.eta)]);
      else if (big && s.speed && big.status === 'running') nums.push([left((big.total - big.done) / s.speed)]);
      if (total == null && !big) {
        // an environment loading its libraries, or (Windows, a PC without NVIDIA) pip installing PyTorch
        var env = busy.some(function (it) { return it.total == null && it.id !== 'gpu-libs'; });
        var other = busy.filter(function (it) { return it.total == null; })[0];
        nums.push([s.id === 'prepare' && env ? 'Loading the libraries once, about a minute' : other ? other.label + '…' : 'Working…', 'txt']);
      }
      if (total != null && st.quiet >= 10) nums.push(['Still working: this part shows no numbers for a while', 'txt']);
    }
    out += '<div class="step-detail"><div class="big-bar"><div class="track" role="progressbar" aria-label="' + esc(s.label) + '"' +
      (total != null ? ' aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + pct(done, total) + '"' : '') + '>' +
      '<div class="fill ' + (s.status === 'failed' ? 'bad' : 'live') + (total != null ? '' : ' indet') + '" style="width:' + (total != null ? pct(done, total) : 30) + '%"></div></div>' +
      '<div class="bar-nums' + (big ? ' has-lab' : '') + '">' + nums.map(function (n) { return '<span' + (n[1] ? ' class="' + n[1] + '"' : '') + '>' + esc(n[0]) + '</span>'; }).join('') + '</div></div>';
    if (s.status === 'running' && s.resumed > 0 && s.id === 'download') {
      out += '<p class="step-note">Continued where it stopped: ' + size(s.resumed) + ' were already here.</p>';
    }
    if (s.items.length) out += '<ul class="items">' + s.items.map(itemRow).join('') + '</ul>';
    return out + '</div></li>';
  }

  // ---------- the page ----------
  var ERR_TITLE = {
    network: 'No connection to the download server', disk: 'Not enough free disk space',
    checksum: 'A downloaded file was damaged', folder: 'The Clausa folder cannot be used where it is',
    system: 'First start stopped unexpectedly'
  };
  var shown = { steps: null, phase: null };

  function render(st) {
    var platform = st.platform || 'macos', phase = st.phase;
    $('fs').setAttribute('aria-busy', phase === 'running' || phase === 'starting' ? 'true' : 'false');
    $('fs').className = 'fs phase-' + phase;
    var dl = st.steps.filter(function (s) { return s.id === 'download'; })[0];

    if (phase === 'done') {
      $('title').innerHTML = ICON.done + '<span>Clausa is ready</span>';
      $('lead').textContent = 'Everything is set up. From now on, start Clausa with a double-click on ' + appName(platform) + '.';
    } else if (phase === 'error') {
      $('title').textContent = 'First start stopped';
      $('lead').textContent = 'What is done stays done: First start continues where it stopped.';
    } else {
      $('title').textContent = st.resumed ? 'Continuing where it stopped' : 'Setting up Clausa';
      // Windows: the GPU libraries, or on a PC without an NVIDIA driver "PyTorch for the processor"
      var lib = dl && dl.items.filter(function (it) { return it.id === 'gpu-libs'; })[0];
      var what = dl && dl.status === 'skipped' ? 'It unpacks the app and prepares it (the speech models came with the download).'
        : 'It unpacks the app, downloads the speech models' + (lib ? ' and ' + (/^GPU/.test(lib.label) ? 'GPU libraries' : 'PyTorch') : '') +
          (dl && dl.total ? ' (' + size(dl.total) + ')' : '') + ' and prepares them.';
      $('lead').textContent = 'This happens only the first time. ' + what;
    }

    var html = st.steps.map(function (s) { return stepCard(s, st); }).join('');
    if (html !== shown.steps) {
      $('steps').innerHTML = html;
      shown.steps = html;
      // new spinners and moving bars pick up the phase of a common clock, so a redraw does not
      // restart them
      var ms = performance.now();
      Array.prototype.forEach.call($('steps').querySelectorAll('.arc'), function (el) { el.style.animationDelay = -(ms % 1000) + 'ms'; });
      Array.prototype.forEach.call($('steps').querySelectorAll('.fill.indet'), function (el) { el.style.animationDelay = -(ms % 1600) + 'ms'; });
    }

    var e = st.error;
    $('error').hidden = !e;
    if (e) {
      $('errTitle').textContent = ERR_TITLE[e.kind] || 'First start stopped';
      $('errText').textContent = e.message;
      $('errDetail').hidden = !e.detail;
      $('errDetail').textContent = e.detail || '';
    }
    var notes = st.messages || [];
    $('notes').hidden = !notes.length;
    $('notes').innerHTML = notes.map(function (m) { return '<li class="' + esc(m.level) + '">' + esc(m.text) + '</li>'; }).join('');

    var canRetry = phase === 'error' && e && e.again;
    $('retryBtn').hidden = !canRetry;
    $('logBtn').hidden = phase !== 'error';
    $('openBtn').hidden = phase !== 'done' || st.starter_gone;  // no one left to open it
    $('footText').textContent = phase === 'done' ? (st.starter_gone ? 'You can close this window.' : 'You can also close this window and open Clausa later.')
      : phase === 'error' ? (canRetry ? 'The log has the details, if you need to ask for help.' : 'To continue, ' + again(platform) + '. The log has the details.')
      : 'Do not move the Clausa folder or close this window until it says Ready.';
    if (phase !== shown.phase) {  // the button to press next gets the focus once
      $('retryBtn').disabled = $('openBtn').disabled = false;
      if (phase === 'done') $('openBtn').focus();
      else if (phase === 'error') (canRetry ? $('retryBtn') : $('logBtn')).focus();
      shown.phase = phase;
    }
  }

  // ---------- the app's side ----------
  function poll() {
    api.get_state().then(render).catch(function () {}).then(function () { setTimeout(poll, 500); });
  }
  $('retryBtn').addEventListener('click', function () { if (api) { $('retryBtn').disabled = true; api.retry(); } });
  $('openBtn').addEventListener('click', function () { if (api) { $('openBtn').disabled = true; api.open_app(); } });
  $('logBtn').addEventListener('click', function () { if (api) api.show_log(); });

  if (params.get('still') === '1') document.documentElement.classList.add('still');
  var mock = params.get('mock');
  if (mock) {
    render((window.FIRST_START_MOCKS || {})[mock] || { phase: 'starting', steps: [] });
  } else {
    var start = function () { api = window.pywebview.api; poll(); };
    if (window.pywebview && window.pywebview.api) start(); else window.addEventListener('pywebviewready', start);
  }
})();
