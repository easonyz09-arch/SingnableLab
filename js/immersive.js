// ─────────────────────────────────────────────
// immersive.js — 3D Immersive Mode Visualization
//
// Artistic concepts:
// 1. Domain-warped noise: coordinates are warped before sampling,
//    creating organic irregular deformation (not just smooth waves)
// 2. Hysteresis: visuals decay slowly after sound stops
// 3. Transient spikes: sudden volume jumps cause random vertex spikes
// 4. Dynamic intensity: displacement amplitude scales aggressively with RMS
// 5. Color maps to pitch
// ─────────────────────────────────────────────

const Immersive = (() => {
  let renderer, scene, camera;
  let sphere, sphereGeo, originalPositions;
  let pointLight, rimLight, fillLight;
  let particleSystem, particlePositions, particleVelocities, particleLifetimes;
  let canvas;
  let frameN   = 0;
  let prevRaw  = 0;   // for transient detection
  let spikes   = [];  // [{idx, strength}] random vertex spikes on volume hit

  // Smoothed features
  let smRMS  = 0, smCent = 0, smZCR = 0;

  const MAX_PARTICLES = 800;

  function init(canvasEl) {
    canvas = canvasEl;

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x04030d, 1);

    scene  = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, canvas.offsetWidth / 320, 0.1, 100);
    camera.position.z = 3.4;

    // Higher resolution sphere for more detailed deformation
    sphereGeo = new THREE.SphereGeometry(1, 96, 96);
    originalPositions = new Float32Array(sphereGeo.attributes.position.array);

    const mat = new THREE.MeshPhongMaterial({
      color:      0x7744ee,
      emissive:   0x220033,
      specular:   0xffffff,
      shininess:  120,
      wireframe:  false,
    });
    sphere = new THREE.Mesh(sphereGeo, mat);
    scene.add(sphere);

    // Lighting — three lights for dramatic shading
    scene.add(new THREE.AmbientLight(0x080814, 1));

    pointLight = new THREE.PointLight(0xffffff, 3, 25);
    pointLight.position.set(4, 3, 4);
    scene.add(pointLight);

    rimLight = new THREE.PointLight(0x4466ff, 2, 20);
    rimLight.position.set(-3, -2, -3);
    scene.add(rimLight);

    fillLight = new THREE.PointLight(0xff4488, 1.5, 15);
    fillLight.position.set(0, -4, 2);
    scene.add(fillLight);

    // Particle system
    const pGeo = new THREE.BufferGeometry();
    particlePositions  = new Float32Array(MAX_PARTICLES * 3);
    particleVelocities = new Float32Array(MAX_PARTICLES * 3);
    particleLifetimes  = new Float32Array(MAX_PARTICLES);

    pGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const pColors = new Float32Array(MAX_PARTICLES * 3);
    pGeo.setAttribute('color', new THREE.BufferAttribute(pColors, 3));

    const pMat = new THREE.PointsMaterial({
      size:         0.07,
      vertexColors: true,
      transparent:  true,
      opacity:      0.9,
      blending:     THREE.AdditiveBlending,
      depthWrite:   false,
    });
    particleSystem = new THREE.Points(pGeo, pMat);
    scene.add(particleSystem);

    resize();
    window.addEventListener('resize', resize);
  }

  function resize() {
    const W = canvas.offsetWidth || 700;
    const H = 320;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  }

  function draw(features, smoothPitch) {
    frameN++;

    // ── Hysteresis smoothing ──────────────────
    const rmsAlpha = features.rms > prevRaw ? 0.35 : 0.04;
    smRMS  = smRMS  * (1 - rmsAlpha) + features.rms      * rmsAlpha;
    smCent = smCent * 0.88            + features.centroid * 0.12;
    smZCR  = smZCR  * 0.88            + features.zcr      * 0.12;

    // ── Transient detection → spikes ─────────
    const transient = features.rms - prevRaw;
    if (transient > 0.04) _addSpikes(Math.floor(transient * 40 + 5));
    prevRaw = features.rms;

    // ── Pitch → hue ───────────────────────────
    const pitchNorm = smoothPitch > 0
      ? Math.min(1, Math.max(0, (smoothPitch - 100) / 900))
      : 0.5;
    const hue = pitchNorm * 240 + 10;
    const col = new THREE.Color(`hsl(${hue}, 80%, 58%)`);

    sphere.material.color.set(col);
    sphere.material.emissive.setHSL(hue / 360, 0.7, 0.05 + smRMS * 0.18);

    // ── Vertex deformation ────────────────────
    _deformSphere(hue);

    // ── Rotation — faster when louder ─────────
    sphere.rotation.y += 0.004 + smRMS * 0.025;
    sphere.rotation.x += 0.001 + smZCR  * 0.008;
    sphere.rotation.z += smRMS * 0.006;

    // ── Lights ────────────────────────────────
    pointLight.color.setHSL(hue / 360, 0.85, 0.65);
    pointLight.intensity = 2 + smRMS * 6;
    rimLight.intensity   = 1.5 + smCent * 5;
    fillLight.position.x = Math.sin(frameN * 0.02) * 3;
    fillLight.intensity  = smRMS * 4;

    // ── Particles ─────────────────────────────
    if (smRMS > 0.025 && frameN % 2 === 0) _emitParticles(smRMS, col);
    _updateParticles();

    // ── Camera subtle drift ───────────────────
    camera.position.x = Math.sin(frameN * 0.005) * 0.4;
    camera.position.y = Math.cos(frameN * 0.004) * 0.2;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
    return { hue, smRMS, smCent };
  }

  // ── Sphere Deformation ─────────────────────
  // Uses domain warping: warp the sampling coordinates first,
  // then sample noise on warped coords → much more organic/irregular result

  function _deformSphere(hue) {
    const pos  = sphereGeo.attributes.position;
    const time = frameN * 0.022;

    // Reset spikes that expired
    spikes = spikes.filter(s => s.strength > 0.008);

    for (let i = 0; i < pos.count; i++) {
      const ox = originalPositions[i * 3];
      const oy = originalPositions[i * 3 + 1];
      const oz = originalPositions[i * 3 + 2];

      // Domain warp: offset coords by low-freq noise before sampling
      // This breaks the regularity of pure sine waves → organic look
      const warpAmp = 0.3 + smRMS * 1.0;
      const wx = Math.sin(oy * 2.1 + time * 0.55) * warpAmp;
      const wy = Math.cos(oz * 1.8 + time * 0.45) * warpAmp;
      const wz = Math.sin(ox * 2.3 + time * 0.65) * warpAmp;

      const nx = ox + wx, ny = oy + wy, nz = oz + wz;

      // Multi-layer noise on warped coords
      const wave =
        Math.sin(nx * 5.5 + time * 2.2) * Math.cos(ny * 4.8 + time * 1.4) * 0.30 +
        Math.sin(ny * 7.0 + time * 1.6) * Math.cos(nz * 6.0 + time * 2.5) * 0.28 +
        Math.sin(nz * 5.0 + time * 1.9) * Math.cos(nx * 6.5 + time * 1.0) * 0.24 +
        Math.sin(nx * 10  + ny * 8.5   + time * 3.5)                       * 0.12 +
        Math.cos(ny * 12  + nz * 9.0   + time * 2.8)                       * 0.06;

      // Displacement: base + audio-driven amplitude
      // smRMS squared makes it respond more aggressively at high volume
      const audioAmp  = 0.25 + smRMS * 1.4 + smRMS * smRMS * 1.2;
      let   disp      = 1.0 + wave * audioAmp;

      pos.setXYZ(i, ox * disp, oy * disp, oz * disp);
    }

    // Apply spikes on top of wave deformation
    spikes.forEach(s => {
      if (s.idx >= pos.count) return;
      const ox = originalPositions[s.idx * 3];
      const oy = originalPositions[s.idx * 3 + 1];
      const oz = originalPositions[s.idx * 3 + 2];
      const scale = 1 + s.strength;
      pos.setXYZ(s.idx, ox * scale, oy * scale, oz * scale);
      s.strength *= 0.88;  // decay
    });

    pos.needsUpdate = true;
    sphereGeo.computeVertexNormals();
  }

  function _addSpikes(count) {
    const maxIdx = sphereGeo.attributes.position.count;
    for (let i = 0; i < count; i++) {
      spikes.push({
        idx:      Math.floor(Math.random() * maxIdx),
        strength: 0.4 + Math.random() * 0.9,
      });
    }
  }

  // ── Particle Helpers ───────────────────────

  function _emitParticles(rms, color) {
    const colors = particleSystem.geometry.attributes.color.array;
    const count  = Math.floor(rms * 14 + 2);
    let emitted  = 0;

    for (let i = 0; i < MAX_PARTICLES && emitted < count; i++) {
      if (particleLifetimes[i] <= 0) {
        const theta = Math.random() * Math.PI * 2;
        const phi   = Math.acos(2 * Math.random() - 1);
        const r     = 1.1 + smRMS * 0.4;
        particlePositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        particlePositions[i * 3 + 2] = r * Math.cos(phi);

        const spd = rms * 0.06 + Math.random() * 0.03;
        particleVelocities[i * 3]     = particlePositions[i * 3]     * spd + (Math.random() - 0.5) * 0.015;
        particleVelocities[i * 3 + 1] = particlePositions[i * 3 + 1] * spd + (Math.random() - 0.5) * 0.015;
        particleVelocities[i * 3 + 2] = particlePositions[i * 3 + 2] * spd;

        particleLifetimes[i] = 0.6 + Math.random() * 1.0;
        colors[i * 3]     = color.r + (Math.random() - 0.5) * 0.3;
        colors[i * 3 + 1] = color.g + (Math.random() - 0.5) * 0.3;
        colors[i * 3 + 2] = color.b + (Math.random() - 0.5) * 0.3;
        emitted++;
      }
    }
    particleSystem.geometry.attributes.position.needsUpdate = true;
    particleSystem.geometry.attributes.color.needsUpdate    = true;
  }

  function _updateParticles() {
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (particleLifetimes[i] <= 0) continue;
      particleLifetimes[i]         -= 0.01;
      particlePositions[i * 3]     += particleVelocities[i * 3];
      particlePositions[i * 3 + 1] += particleVelocities[i * 3 + 1];
      particlePositions[i * 3 + 2] += particleVelocities[i * 3 + 2];
      particleVelocities[i * 3 + 1] += 0.0003;
    }
    particleSystem.geometry.attributes.position.needsUpdate = true;
  }

  return { init, draw, resize };
})();