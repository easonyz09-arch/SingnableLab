// ─────────────────────────────────────────────
// app.js — Main Application Logic
// Handles: init, tab switching, feedback, fullscreen, main loop
// ─────────────────────────────────────────────

const App = (() => {
  let curTab     = 1;
  let frameN     = 0;
  let lastFbTime = 0;
  let fbTimer    = null;
  let paused     = false;

  const FEEDBACK_RULES = [
    [(f, p, s) => s > 0.75 && f.rms > 0.03,              '◎ Great intonation!',   2500],
    [(f, p, s) => f.rms > 0.12 && p > 300,                '★ Powerful high note!', 3000],
    [(f, p, s) => f.rms > 0.04 && p > 0 && p < 220,       '✦ Rich low tone',       3000],
    [(f, p, s) => f.rms > 0.03 && f.zcr < 0.045 && p > 0, '◈ Clean tone',          2800],
    [(f, p, s) => f.rms > 0.06 && f.rms < 0.1 && s > 0.5, '✿ Sounding great',      3500],
  ];

  function init() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        curTab = parseInt(btn.dataset.tab);
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.tab-content').forEach((el, i) => el.classList.toggle('active', i + 1 === curTab));
        if (curTab === 2) setTimeout(() => Space.resize(), 10);
      });
    });

    // Start microphone
    document.getElementById('mic-btn').addEventListener('click', async () => {
      const ok = await Audio.start();
      if (ok) {
        paused = false;
        document.getElementById('mic-btn').style.display  = 'none';
        document.getElementById('stop-btn').style.display = 'inline-block';
        document.getElementById('status-text').textContent = 'Analyzing in real-time — color follows pitch';
        loop();
      } else {
        document.getElementById('status-text').textContent = 'Could not access microphone. Check permissions.';
      }
    });

    // Pause / Resume
    document.getElementById('stop-btn').addEventListener('click', () => {
      paused = !paused;
      document.getElementById('stop-btn').textContent    = paused ? 'Resume' : 'Pause';
      document.getElementById('status-text').textContent = paused ? 'Paused' : 'Analyzing in real-time — color follows pitch';
    });

    // Fullscreen toggle
    document.getElementById('fs-btn').addEventListener('click', toggleFullscreen);

    document.addEventListener('fullscreenchange', () => {
      const isFS = !!document.fullscreenElement;
      document.body.classList.toggle('fs-mode', isFS);
      document.getElementById('fs-btn').textContent = isFS ? '✕ Exit Fullscreen' : '⛶ Fullscreen';
      // Give CSS time to apply new heights before resizing canvases
      setTimeout(() => {
        Immersive.resize();
        Space.resize();
      }, 80);
    });

    Immersive.init(document.getElementById('cv-immersive'));
    Space.init(document.getElementById('cv-space'));
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(e => {
        console.warn('Fullscreen request failed:', e);
      });
    } else {
      document.exitFullscreen();
    }
  }

  function loop() {
    requestAnimationFrame(loop);
    if (!Audio.isRunning() || paused) return;
    frameN++;

    const features = Audio.getFeatures();
    const pitch    = Audio.smoothPitch(features.pitch);

    if (curTab === 1) {
      const { hue } = Immersive.draw(features, pitch);
      updateImmersiveUI(features, pitch, hue);
      triggerFeedback(features, pitch, features.stability, hue);
    } else {
      const nearZone = Space.draw(features);
      updateSpaceUI(features, nearZone);
    }
  }

  function updateImmersiveUI(f, pitch, hue) {
    const note  = Audio.freqToNote(pitch);
    const label = document.getElementById('pitch-label');
    const sub   = document.getElementById('pitch-sub');

    if (note && f.rms > 0.01) {
      label.childNodes[0].textContent = note.name + note.octave;
      label.style.color = `hsla(${hue}, 85%, 80%, 0.9)`;
      sub.textContent   = Math.round(pitch) + ' Hz';
    } else {
      label.childNodes[0].textContent = '—';
      label.style.color = 'rgba(255,255,255,0.4)';
      sub.textContent   = f.rms > 0.005 ? 'Pitch unclear' : 'Waiting for sound';
    }

    document.getElementById('stat-pitch').textContent  = note ? note.name + note.octave : '—';
    document.getElementById('stat-hz').textContent     = pitch > 0 ? Math.round(pitch) + ' Hz' : '—';
    document.getElementById('stat-volume').textContent = f.rms > 0.005 ? Math.round(f.rms * 100) + '%' : '0%';

    const stability = f.stability || 0;
    const stabEl    = document.getElementById('stat-stability');
    stabEl.textContent = stability > 0.7 ? 'Stable ◎' : stability > 0.4 ? 'Fair' : 'Unstable';
    stabEl.style.color = stability > 0.7 ? '#1D9E75' : stability > 0.4 ? '#BA7517' : '#E24B4A';
  }

  function updateSpaceUI(f, nearZone) {
    document.getElementById('f2-centroid').textContent = (f.centroid * 1000).toFixed(1);
    document.getElementById('f2-rms').textContent      = (f.rms * 100).toFixed(2) + '%';
    document.getElementById('f2-zcr').textContent      = (f.zcr * 1000).toFixed(1);
    const zoneEl = document.getElementById('f2-zone');
    zoneEl.textContent = nearZone ? nearZone.name : '—';
    zoneEl.style.color = nearZone ? nearZone.color : 'var(--text-2)';
  }

  function triggerFeedback(features, pitch, stability, hue) {
    const now = Date.now();
    for (const [cond, msg, cooldown] of FEEDBACK_RULES) {
      if (cond(features, pitch, stability) && now - lastFbTime > cooldown) {
        lastFbTime = now;
        const el = document.getElementById('feedback-msg');
        el.textContent   = msg;
        el.style.color   = `hsl(${hue}, 90%, 78%)`;
        el.style.opacity = '1';
        clearTimeout(fbTimer);
        fbTimer = setTimeout(() => { el.style.opacity = '0'; }, 1700);
        break;
      }
    }
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', () => App.init());