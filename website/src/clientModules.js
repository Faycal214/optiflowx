// website/src/clientModules.js
// Client module to mount the FloatingColorToggle component
import React from 'react';
import ReactDOM from 'react-dom';
import FloatingColorToggle from './components/FloatingColorToggle';

export default function() {
  // create container
  const id = 'ofx-toggle-root';
  let container = document.getElementById(id);
  if (!container) {
    container = document.createElement('div');
    container.id = id;
    document.body.appendChild(container);
  }
  // render
  try {
    ReactDOM.render(React.createElement(FloatingColorToggle), container);
  } catch (err) {
    // fallback for react 18+ with createRoot
    try {
      const { createRoot } = require('react-dom/client');
      const root = createRoot(container);
      root.render(React.createElement(FloatingColorToggle));
    } catch (e) {
      // give up silently
      console.error('Failed to mount FloatingColorToggle', e);
    }
  }
}
