// docs/javascripts/extra.js
// OptiFlowX compact theme toggle
// Place at docs/javascripts/extra.js
(function () {
  const STORAGE_KEY = 'optiflowx-color-scheme';
  const HTML_ATTR = 'data-md-color-scheme';
  const DARK = 'slate';
  const LIGHT = 'default';

  function prefersLight() {
    try {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
    } catch (e) {
      return false;
    }
  }

  function readStored() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function store(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {}
  }

  function getCurrent() {
    const attr = document.documentElement.getAttribute(HTML_ATTR);
    if (attr) return attr;
    const stored = readStored();
    if (stored) return stored;
    return prefersLight() ? LIGHT : DARK;
  }

  function applyScheme(scheme) {
    document.documentElement.setAttribute(HTML_ATTR, scheme);
    // Update toggle UI state
    const btn = document.querySelector('.ofx-toggle');
    if (!btn) return;
    const icon = btn.querySelector('.ofx-toggle-icon');
    const label = btn.querySelector('.ofx-toggle-label');
    if (scheme === DARK) {
      icon.innerHTML = '🌙';
      label.textContent = 'Dark';
      btn.setAttribute('aria-pressed', 'true');
    } else {
      icon.innerHTML = '☀️';
      label.textContent = 'Light';
      btn.setAttribute('aria-pressed', 'false');
    }
  }

  function toggleOptiFlowXScheme() {
    const current = getCurrent();
    const next = current === DARK ? LIGHT : DARK;
    applyScheme(next);
    store(next);
  }

  // Expose for console/testing
  window.toggleOptiFlowXScheme = toggleOptiFlowXScheme;

  // Keyboard shortcut: Ctrl+Shift+D toggles theme
  function onKey(e) {
    if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
      e.preventDefault();
      toggleOptiFlowXScheme();
    }
  }

  // Inject the floating toggle if not present
  function createToggle() {
    if (document.querySelector('.ofx-toggle')) return;

    const btn = document.createElement('button');
    btn.className = 'ofx-toggle';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Toggle color scheme');
    btn.setAttribute('title', 'Toggle color scheme (Ctrl+Shift+D)');
    btn.setAttribute('role', 'switch');
    btn.setAttribute('aria-pressed', 'false');

    const icon = document.createElement('span');
    icon.className = 'ofx-toggle-icon';
    icon.style.fontSize = '16px';

    const label = document.createElement('span');
    label.className = 'ofx-toggle-label';
    label.style.fontSize = '0.86rem';
    label.style.opacity = '0.96';

    btn.appendChild(icon);
    btn.appendChild(label);

    btn.addEventListener('click', function (ev) {
      ev.preventDefault();
      toggleOptiFlowXScheme();
      btn.focus();
    });

    // Make toggle keyboard operable (Enter/Space)
    btn.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        toggleOptiFlowXScheme();
      }
    });

    document.body.appendChild(btn);
    // If favicon exists, ensure it's set for the page
    try {
      const link = document.querySelector('link[rel="icon"]') || document.createElement('link');
      link.setAttribute('rel', 'icon');
      link.setAttribute('type', 'image/svg+xml');
      // Use relative path so it works with subpath deployments
      link.setAttribute('href', 'assets/images/opti-mark.svg');
      if (!document.querySelector('link[rel="icon"]')) document.head.appendChild(link);
    } catch (e) {}
    return btn;
  }

  // Initialize scheme on DOMContentLoaded
  function init() {
    const initial = getCurrent();
    applyScheme(initial);
    createToggle();
    document.addEventListener('keydown', onKey, false);

    // Observe changes to storage (multi-tab)
    window.addEventListener('storage', function (ev) {
      if (ev.key === STORAGE_KEY) {
        const scheme = ev.newValue || (prefersLight() ? LIGHT : DARK);
        applyScheme(scheme);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
