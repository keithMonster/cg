/**
 * Ouroboros Kernel: Satellite.js
 * The shared brain for all Ouroboros Satellite Applications.
 * Handles: Local Vault Authorization, Supabase Client Initialization, and Common Utilities.
 */

// 1. Ouroboros Standard Configuration (The Local Vault)
const ODS_CONFIG = {
  url:
    localStorage.getItem('sb_url') ||
    'https://pscrjxtqukzivnuiqwah.supabase.co',
  key: localStorage.getItem('sb_key') || '',
  theme: localStorage.getItem('ods_theme') || 'dark',
};

// 2. Global Supabase Client Instance
// Satellites can access this globally as `sbClient`
window.sbClient =
  window.supabase && ODS_CONFIG.url && ODS_CONFIG.key
    ? window.supabase.createClient(ODS_CONFIG.url, ODS_CONFIG.key)
    : null;

// 3. Alpine.js Satellite Mixin
// Usage: x-data="Satellite({ title: 'App Name' })"
document.addEventListener('alpine:init', () => {
  Alpine.data('Satellite', (config = {}) => ({
    // Meta
    appTitle: config.title || 'Untitled Satellite',

    // System State
    dbState: window.sbClient ? 'online' : 'offline', // 'online', 'offline', 'loading'

    // Inherited Methods
    init() {
      this.setTheme();
      console.log(`[Ouroboros Kernel] ${this.appTitle} initialized.`);
      if (config.init) config.init.call(this); // Run custom init if provided
    },

    setTheme() {
      document.documentElement.classList.add(`sl-theme-${ODS_CONFIG.theme}`);
    },

    // Helper: Format seconds to HH:MM:SS
    formatTime(seconds) {
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      return [h, m, s].map((v) => (v < 10 ? '0' + v : v)).join(':');
    },

    // Helper: Simple Toast (wraps console for now, can expand to UI)
    notify(message, type = 'info') {
      console.log(`[${type.toUpperCase()}] ${message}`);
      // Future: Integrate with Shoelace alert
    },
  }));
});
