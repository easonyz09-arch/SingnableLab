// ─────────────────────────────────────────────
// audio.js — Audio Engine
// Handles: microphone input, feature extraction, pitch detection
// ─────────────────────────────────────────────

const Audio = (() => {
  let audioCtx  = null;
  let analyser  = null;
  let floatBuf  = null;   // 32-bit time-domain data (for pitch detection)
  let freqBuf   = null;   // frequency-domain data (for spectral centroid)
  let running   = false;

  let pitchHistory = [];

  const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

  // ── Public Methods ─────────────────────────

  async function start() {
    if (running) return true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new AudioContext();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 4096;
      floatBuf = new Float32Array(analyser.fftSize);
      freqBuf  = new Uint8Array(analyser.frequencyBinCount);
      audioCtx.createMediaStreamSource(stream).connect(analyser);
      running = true;
      return true;
    } catch (e) {
      console.error('Microphone error:', e);
      return false;
    }
  }

  function isRunning() { return running; }

  // Returns all features for the current frame
  function getFeatures() {
    if (!running) return null;
    analyser.getFloatTimeDomainData(floatBuf);
    analyser.getByteFrequencyData(freqBuf);
    const rms      = computeRMS();
    const zcr      = computeZCR();
    const centroid = computeCentroid();
    const pitch    = computePitch(rms);
    const stability = computeStability();
    return { rms, zcr, centroid, pitch, stability };
  }

  // ── Private: Feature Computation ──────────

  function computeRMS() {
    let sum = 0;
    for (let i = 0; i < floatBuf.length; i++) sum += floatBuf[i] ** 2;
    return Math.sqrt(sum / floatBuf.length);
  }

  function computeZCR() {
    let count = 0;
    for (let i = 1; i < floatBuf.length; i++) {
      if ((floatBuf[i] >= 0) !== (floatBuf[i - 1] >= 0)) count++;
    }
    return count / floatBuf.length;
  }

  function computeCentroid() {
    let num = 0, den = 0;
    for (let i = 0; i < freqBuf.length; i++) {
      num += i * freqBuf[i];
      den += freqBuf[i];
    }
    return den > 0 ? (num / den) / freqBuf.length : 0;
  }

  // Normalized cross-correlation pitch detection
  // Search range limited to human singing range: 60–1100 Hz
  // Avoids detecting harmonics (which was the bug in the original version)
  function computePitch(rms) {
    if (rms < 0.008) return -1;
    const sr        = audioCtx.sampleRate;
    const N         = floatBuf.length;
    const halfN     = Math.floor(N / 2);
    const minPeriod = Math.ceil(sr / 1100);  // ~40 samples at 44100 Hz
    const maxPeriod = Math.floor(sr / 60);   // ~735 samples at 44100 Hz
    let bestTau = -1, bestR = -1, prevR = 0, found = false;
    for (let tau = minPeriod; tau <= Math.min(maxPeriod, halfN - 1); tau++) {
      let sum = 0, sq1 = 0, sq2 = 0;
      for (let i = 0; i < halfN; i++) {
        sum += floatBuf[i] * floatBuf[i + tau];
        sq1 += floatBuf[i] ** 2;
        sq2 += floatBuf[i + tau] ** 2;
      }
      const r = sq1 * sq2 > 0 ? sum / Math.sqrt(sq1 * sq2) : 0;
      if (r > 0.82 && r > prevR) {
        found = true;
        if (r > bestR) { bestR = r; bestTau = tau; }
      } else if (found && r < prevR - 0.05) break;
      prevR = r;
    }
    return bestTau > 0 && bestR >= 0.82 ? sr / bestTau : -1;
  }

  // Median smoothing over last 12 frames to remove pitch outliers
  function smoothPitch(rawPitch) {
    pitchHistory.push(rawPitch);
    if (pitchHistory.length > 12) pitchHistory.shift();
    const valid = pitchHistory.filter(p => p > 60 && p < 1200);
    if (valid.length < 3) return -1;
    const sorted = [...valid].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  // Pitch stability score (0–1): measures consistency of recent pitch values
  function computeStability() {
    const valid = pitchHistory.filter(p => p > 60 && p < 1200);
    if (valid.length < 5) return 0;
    const mean     = valid.reduce((a, b) => a + b, 0) / valid.length;
    const variance = valid.reduce((a, b) => a + (b - mean) ** 2, 0) / valid.length;
    const cv       = Math.sqrt(variance) / mean;  // coefficient of variation
    return Math.max(0, Math.min(1, 1 - cv * 8));
  }

  // Convert frequency (Hz) to note name (e.g. "A4", "C3")
  function freqToNote(freq) {
    if (freq < 60 || freq > 1400) return null;
    const midi = Math.round(12 * Math.log2(freq / 440)) + 69;
    return { name: NOTE_NAMES[midi % 12], octave: Math.floor(midi / 12) - 1, midi };
  }

  return { start, isRunning, getFeatures, smoothPitch, freqToNote };
})();
