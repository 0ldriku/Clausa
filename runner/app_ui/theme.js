/* The window's theme, before the first paint (loaded in <head>): html data-theme "light" or "dark",
   from ?theme= (Settings › Appearance: light, dark, or system), else the system's, followed as it
   changes. app.js passes the setting on with window.clausaTheme(setting). app.css draws both. */
(function () {
  'use strict';
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
  var setting = new URLSearchParams(window.location.search).get('theme');
  function apply() {
    var theme = setting === 'light' || setting === 'dark' ? setting : (media && media.matches ? 'light' : 'dark');
    if (document.documentElement.getAttribute('data-theme') !== theme) document.documentElement.setAttribute('data-theme', theme);
  }
  window.clausaTheme = function (s) { if (s !== setting) { setting = s; apply(); } };
  if (media && media.addEventListener) media.addEventListener('change', apply);
  apply();
})();
