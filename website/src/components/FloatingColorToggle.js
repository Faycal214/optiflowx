import React, {useEffect, useState} from 'react';
import {useColorMode} from '@docusaurus/theme-common';

export default function FloatingColorToggle() {
  const {colorMode, setColorMode} = useColorMode();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    // expose global toggle function
    window.toggleOptiFlowXScheme = () => {
      setColorMode(colorMode === 'dark' ? 'light' : 'dark');
      try { localStorage.setItem('optiflowx-color-scheme', colorMode === 'dark' ? 'default' : 'slate'); } catch(e){}
    };

    function onKey(e) {
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        window.toggleOptiFlowXScheme();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [colorMode, setColorMode]);

  useEffect(() => {
    // Respect stored preference on mount
    try {
      const stored = localStorage.getItem('optiflowx-color-scheme');
      if (stored === 'slate') setColorMode('dark');
      if (stored === 'default') setColorMode('light');
    } catch (e) {}
  }, [setColorMode]);

  if (!mounted) return null;

  return (
    <button
      className="ofx-toggle"
      aria-label="Toggle color scheme"
      title="Toggle color scheme (Ctrl+Shift+D)"
      onClick={() => {
        const next = colorMode === 'dark' ? 'light' : 'dark';
        setColorMode(next);
        try { localStorage.setItem('optiflowx-color-scheme', next === 'dark' ? 'slate' : 'default'); } catch(e){}
      }}
    >
      <span className="ofx-toggle-icon" aria-hidden>
        {colorMode === 'dark' ? '🌙' : '☀️'}
      </span>
      <span className="label ofx-toggle-label">{colorMode === 'dark' ? 'Dark' : 'Light'}</span>
    </button>
  );
}
