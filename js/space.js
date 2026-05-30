// ─────────────────────────────────────────────
// space.js — Vocal Zone Analysis Visualization
// Handles: 2D feature space, real-time cursor, trail
// Zone coordinates loaded from data/zones.json
// (replaced with real UMAP output after running ml/extract_and_umap.py)
// ─────────────────────────────────────────────

const Space = (() => {
  let canvas, ctx;
  let zones = [];
  let trail = [];

  // Smoothed position
  let smX = 0, smY = 0;

  async function init(canvasEl) {
    canvas = canvasEl;
    ctx    = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize);

    // Load zone coordinates from JSON
    // After running the ML pipeline, this file contains real UMAP coordinates
    try {
      const res  = await fetch('data/zones.json');
      const data = await res.json();
      zones = data.zones;
      console.log(`✓ Loaded ${zones.length} vocal zones`);
    } catch (e) {
      console.warn('Could not load zones.json, using defaults');
      zones = _defaultZones();
    }
    drawStatic();
  }

  function resize() {
    // Fallback if canvas is hidden (offsetWidth = 0 when tab is not active)
    const w = canvas.offsetWidth || canvas.parentElement?.offsetWidth || 700;
    canvas.width        = w;
    canvas.height       = 300;
    canvas.style.height = '300px';
    drawStatic();
  }

  // Called every frame
  function draw(features) {
    // Map features to [0, 1] coordinates
    smX = smX * 0.78 + Math.min(1, Math.max(0, features.centroid * 13)) * 0.22;
    smY = smY * 0.78 + Math.min(1, Math.max(0, features.rms       * 11)) * 0.22;

    // Record trail
    if (features.rms > 0.012) {
      trail.push({ x: smX, y: smY });
      if (trail.length > 100) trail.shift();
    }

    drawStatic();
    _drawTrail();
    _drawCursor(smX, smY);

    return _nearestZone(smX, smY);
  }

  function drawStatic() {
    if (!canvas.width) return;
    const W = canvas.width, H = canvas.height;
    const { pad, pw, ph } = _layout(W, H);
    ctx.clearRect(0, 0, W, H);
    _drawGrid(pad, pw, ph, W, H);
    _drawAxisLabels(pad, pw, ph, W, H);
    _drawZones(pad, pw, ph);
  }

  // ── Private Drawing ────────────────────────

  function _layout(W, H) {
    const pad = 44;
    return { pad, pw: W - pad * 2, ph: H - pad * 2 };
  }

  function _drawGrid(pad, pw, ph, W, H) {
    ctx.strokeStyle = 'rgba(128,128,128,0.1)';
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(pad + i/4 * pw, pad); ctx.lineTo(pad + i/4 * pw, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, pad + i/4 * ph); ctx.lineTo(W - pad, pad + i/4 * ph); ctx.stroke();
    }
  }

  function _drawAxisLabels(pad, pw, ph, W, H) {
    ctx.fillStyle  = 'rgba(128,128,128,0.5)';
    ctx.font       = '10px sans-serif';
    ctx.textAlign  = 'center';
    ctx.fillText('← Dark / Low    Brightness    Bright / Metallic →', pad + pw / 2, H - 6);
    ctx.save();
    ctx.translate(11, pad + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('← Soft   Energy   Powerful →', 0, 0);
    ctx.restore();
  }

  function _drawZones(pad, pw, ph) {
    zones.forEach(z => {
      const zx = pad + z.x * pw;
      const zy  = pad + (1 - z.y) * ph;
      const r   = z.radius || 50;
      const g   = ctx.createRadialGradient(zx, zy, 0, zx, zy, r);
      g.addColorStop(0, z.color + '32');
      g.addColorStop(1, z.color + '00');
      ctx.beginPath(); ctx.arc(zx, zy, r, 0, Math.PI * 2);
      ctx.fillStyle = g; ctx.fill();
      ctx.fillStyle  = z.color;
      ctx.font       = '11px sans-serif';
      ctx.textAlign  = 'center';
      ctx.fillText(z.name, zx, zy + 4);
    });
  }

  function _drawTrail() {
    if (trail.length < 2) return;
    const W = canvas.width, H = canvas.height;
    const { pad, pw, ph } = _layout(W, H);
    ctx.beginPath();
    trail.forEach((p, i) => {
      const tx = pad + p.x * pw, ty = pad + (1 - p.y) * ph;
      i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty);
    });
    ctx.strokeStyle = 'rgba(83, 74, 183, 0.28)';
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }

  function _drawCursor(x, y) {
    const W = canvas.width, H = canvas.height;
    const { pad, pw, ph } = _layout(W, H);
    const px = pad + x * pw, py = pad + (1 - y) * ph;

    const near = _nearestZone(x, y);
    if (near) {
      ctx.strokeStyle = near.color + '50';
      ctx.lineWidth   = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(pad + near.x * pw, pad + (1 - near.y) * ph);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Outer ring
    ctx.beginPath(); ctx.arc(px, py, 20, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(83, 74, 183, 0.18)';
    ctx.lineWidth   = 1; ctx.stroke();

    // Core dot
    ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#534AB7'; ctx.fill();
  }

  function _nearestZone(x, y) {
    let near = null, minDist = Infinity;
    zones.forEach(z => {
      const d = Math.hypot(z.x - x, z.y - y);
      if (d < minDist) { minDist = d; near = z; }
    });
    return minDist < 0.42 ? near : null;
  }

  function _defaultZones() {
    return [
      { name: 'Whisper',   x: 0.12, y: 0.07, color: '#888780', radius: 50 },
      { name: 'Breathy',   x: 0.26, y: 0.24, color: '#7F77DD', radius: 48 },
      { name: 'Falsetto',  x: 0.74, y: 0.30, color: '#378ADD', radius: 50 },
      { name: 'Chest',     x: 0.38, y: 0.64, color: '#1D9E75', radius: 52 },
      { name: 'Resonance', x: 0.54, y: 0.74, color: '#BA7517', radius: 48 },
      { name: 'Belting',   x: 0.64, y: 0.90, color: '#E24B4A', radius: 50 },
    ];
  }

  function clearTrail() { trail = []; }

  return { init, draw, drawStatic, resize, clearTrail };
})();
