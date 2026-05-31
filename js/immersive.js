// ─────────────────────────────────────────────
// immersive.js — 3D Immersive Mode Visualization
//
// Artistic concepts:
// 1. Domain-warped noise: organic irregular deformation
// 2. Hysteresis: visuals decay slowly after sound stops
// 3. Transient spikes: sudden volume jumps cause vertex spikes
// 4. Idle breathing: sphere shrinks and calms when no sound
// 5. Premium color palette: amber → sapphire, skips harsh greens
// ─────────────────────────────────────────────

const Immersive = (() => {
  let renderer, scene, camera;
  let sphere, sphereGeo, originalPositions;
  let pointLight, rimLight, fillLight;
  let particleSystem, particlePositions, particleVelocities, particleLifetimes;
  let canvas;
  let frameN  = 0;
  let prevRaw = 0;
  let spikes  = [];

  // Smoothed features
  let smRMS  = 0, smCent = 0, smZCR = 0;

  // Idle state: sphere shrinks smoothly when no sound
  let idleScale = 0.35;

  const MAX_PARTICLES = 800;

  function init(canvasEl) {
    canvas = canvasEl;

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x04030d, 1);

    scene  = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(58, 1, 0.1, 100);
    camera.position.z = 3.4;

    sphereGeo         = new THREE.SphereGeometry(1, 96, 96);
    originalPositions = new Float32Array(sphereGeo.attributes.position.array);

    const mat = new THREE.MeshPhongMaterial({
      color:     0x5533aa,
      emissive:  0x110022,
      specular:  0xffffff,
      shininess: 140,
    });
    sphere = new THREE.Mesh(sphereGeo, mat);
    scene.add(sphere);

    scene.add(new THREE.AmbientLight(0x060610, 1));

    pointLight = new THREE.PointLight(0xffffff, 3, 25);
    pointLight.position.set(4, 3, 4);
    scene.add(pointLight);

    rimLight = new THREE.PointLight(0x4422ff, 2, 20);
    rimLight.position.set(-3, -2, -3);
    scene.add(rimLight);

    fillLight = new THREE.PointLight(0xff2266, 1.5, 15);
    fillLight.position.set(0, -4, 2);
    scene.add(fillLight);

    const pGeo = new THREE.BufferGeometry();
    particlePositions  = new Float32Array(MAX_PARTICLES * 3);
    particleVelocities = new Float32Array(MAX_PARTICLES * 3);
    particleLifetimes  = new Float32Array(MAX_PARTICLES);
    const pColors      = new Float32Array(MAX_PARTICLES * 3);

    pGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    pGeo.setAttribute('color',    new THREE.BufferAttribute(pColors, 3));

    particleSystem = new THREE.Points(pGeo, new THREE.PointsMaterial({
      size:         0.07,
      vertexColors: true,
      transparent:  true,
      opacity:      0.85,
      blending:     THREE.AdditiveBlending,
      depthWrite:   false,
    }));
    scene.add(particleSystem);

    resize();
    window.addEventListener('resize', resize);
  }

  function resize() {
    const W = window.innerWidth;
    const H = document.body.classList.contains('fs-mode')
        ? window.innerHeight - 190
        : 540;

    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    }

  // Premium color mapping — amber (25°) → sapphire (260°), skips harsh greens
  function pitchToHue(pitchNorm) {
    if (pitchNorm < 0.45) {
      return 25 + pitchNorm / 0.45 * 155;   // 25° amber → 180° teal
    } else {
      return 200 + (pitchNorm - 0.45) / 0.55 * 80; // 200° blue → 280° violet
    }
  }

  function draw(features, smoothPitch) {
    frameN++;

    // Hysteresis smoothing
    const rmsAlpha = features.rms > prevRaw ? 0.28 : 0.05;
    smRMS  = smRMS  * (1 - rmsAlpha) + features.rms      * rmsAlpha;
    smCent = smCent * 0.88            + features.centroid * 0.12;
    smZCR  = smZCR  * 0.88            + features.zcr      * 0.12;

    // Idle scale: shrinks smoothly to 0.32 when quiet, expands to 1.0 when singing
    const targetScale = smRMS > 0.015 ? 1.0 : 0.32;
    idleScale         = idleScale * 0.96 + targetScale * 0.04;

    // Transient detection → spikes
    const transient = features.rms - prevRaw;
    if (transient > 0.04) _addSpikes(Math.floor(transient * 40 + 5));
    prevRaw = features.rms;

    // Color
    const pitchNorm = smoothPitch > 0
      ? Math.min(1, Math.max(0, (smoothPitch - 100) / 900))
      : 0.5;
    const hue = pitchToHue(pitchNorm);
    const col = new THREE.Color(`hsl(${hue}, 88%, 52%)`);

    sphere.material.color.set(col);
    sphere.material.emissive.setHSL(hue / 360, 0.75, 0.04 + smRMS * 0.14);

    // Apply idle scale to entire sphere
    sphere.scale.setScalar(idleScale);

    // Vertex deformation (amplitude also reduced when idle)
    _deformSphere();

    // Rotation — slows down when idle
    sphere.rotation.y += (0.003 + smRMS * 0.022) * idleScale;
    sphere.rotation.x += (0.001 + smZCR  * 0.007) * idleScale;
    sphere.rotation.z += smRMS * 0.005 * idleScale;

    // Lights
    pointLight.color.setHSL(hue / 360, 0.9, 0.62);
    pointLight.intensity = 2 + smRMS * 5;
    rimLight.intensity   = 1.2 + smCent * 4;
    fillLight.position.x = Math.sin(frameN * 0.02) * 3;
    fillLight.intensity  = smRMS * 3;

    // Particles only when actually singing
    if (smRMS > 0.03 && frameN % 2 === 0) _emitParticles(smRMS, col);
    _updateParticles();

    // Camera drift
    camera.position.x = Math.sin(frameN * 0.005) * 0.35;
    camera.position.y = Math.cos(frameN * 0.004) * 0.18;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
    return { hue, smRMS, smCent };
  }

  function _deformSphere() {
    const pos  = sphereGeo.attributes.position;
    const time = frameN * 0.022;

    spikes = spikes.filter(s => s.strength > 0.008);

    for (let i = 0; i < pos.count; i++) {
      const ox = originalPositions[i * 3];
      const oy = originalPositions[i * 3 + 1];
      const oz = originalPositions[i * 3 + 2];

      // Domain warp — scales down when idle for gentler surface
      const warpAmp = (0.2 + smRMS * 0.9) * idleScale;
      const wx      = Math.sin(oy * 2.1 + time * 0.55) * warpAmp;
      const wy      = Math.cos(oz * 1.8 + time * 0.45) * warpAmp;
      const wz      = Math.sin(ox * 2.3 + time * 0.65) * warpAmp;

      const nx = ox + wx, ny = oy + wy, nz = oz + wz;

      const wave =
        Math.sin(nx * 5.5 + time * 2.2) * Math.cos(ny * 4.8 + time * 1.4) * 0.30 +
        Math.sin(ny * 7.0 + time * 1.6) * Math.cos(nz * 6.0 + time * 2.5) * 0.28 +
        Math.sin(nz * 5.0 + time * 1.9) * Math.cos(nx * 6.5 + time * 1.0) * 0.24 +
        Math.sin(nx * 10  + ny * 8.5   + time * 3.5)                       * 0.12 +
        Math.cos(ny * 12  + nz * 9.0   + time * 2.8)                       * 0.06;

      // Amplitude shrinks significantly when idle
      const audioAmp = (0.12 + smRMS * 1.2 + smRMS * smRMS * 1.0) * idleScale;
      const disp     = 1.0 + wave * audioAmp;

      pos.setXYZ(i, ox * disp, oy * disp, oz * disp);
    }

    spikes.forEach(s => {
      if (s.idx >= pos.count) return;
      const ox    = originalPositions[s.idx * 3];
      const oy    = originalPositions[s.idx * 3 + 1];
      const oz    = originalPositions[s.idx * 3 + 2];
      const scale = 1 + s.strength;
      pos.setXYZ(s.idx, ox * scale, oy * scale, oz * scale);
      s.strength *= 0.88;
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

  function _emitParticles(rms, color) {
    const colors = particleSystem.geometry.attributes.color.array;
    const count  = Math.floor(rms * 14 + 2);
    let emitted  = 0;

    for (let i = 0; i < MAX_PARTICLES && emitted < count; i++) {
      if (particleLifetimes[i] <= 0) {
        const theta = Math.random() * Math.PI * 2;
        const phi   = Math.acos(2 * Math.random() - 1);
        const r     = idleScale * (1.1 + smRMS * 0.4);
        particlePositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        particlePositions[i * 3 + 2] = r * Math.cos(phi);

        const spd = rms * 0.06 + Math.random() * 0.03;
        particleVelocities[i * 3]     = particlePositions[i * 3]     * spd;
        particleVelocities[i * 3 + 1] = particlePositions[i * 3 + 1] * spd;
        particleVelocities[i * 3 + 2] = particlePositions[i * 3 + 2] * spd;

        particleLifetimes[i] = 0.6 + Math.random() * 1.0;
        colors[i * 3]     = color.r + (Math.random() - 0.5) * 0.2;
        colors[i * 3 + 1] = color.g + (Math.random() - 0.5) * 0.2;
        colors[i * 3 + 2] = color.b + (Math.random() - 0.5) * 0.2;
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