# VOCALAB

Real-time singing visualization tool that helps amateur singers improve emotional expression and self-awareness.

## Problem Statement

1. **Immersion**: Singers lack real-time visual feedback that deepens their emotional engagement while singing
2. **Self-awareness**: Amateur singers don't know what vocal zone they're in or how to improve

## Features

**Immersive Mode** — Real-time cosmic visualization driven by your voice. Color maps to pitch (low = warm orange, high = blue-purple). Positive feedback messages appear when good vocal qualities are detected.

**Vocal Zone Analysis** — Displays your voice's position in a 2D feature space in real-time. Shows which vocal zone you're closest to (Chest, Breathy, Belting, etc.).

## Project Structure

```
vocalab/
├── index.html          # Page structure
├── style.css           # Styles
├── js/
│   ├── audio.js        # Audio engine: microphone, pitch detection, feature extraction
│   ├── immersive.js    # Tab 1: immersive visualization
│   ├── space.js        # Tab 2: vocal zone analysis (loads coordinates from zones.json)
│   └── app.js          # Main logic: tab switching, feedback system, render loop
├── data/
│   └── zones.json      # Zone coordinates (placeholder → replaced by ML pipeline output)
└── ml/
    └── extract_and_umap.py   # Python ML pipeline
```

## Running Locally

Direct file:// access won't work (browser blocks local fetch requests).
Use a local server:

**Option A: VS Code Live Server**
1. Install the "Live Server" extension
2. Right-click index.html → "Open with Live Server"

**Option B: Python**
```bash
cd vocalab
python3 -m http.server 8000
# Open http://localhost:8000
```

## ML Pipeline

Replaces placeholder zone coordinates with real data derived from the VocalSet dataset.

**Setup**
```bash
pip3 install librosa umap-learn numpy scipy pandas scikit-learn
```

**Download VocalSet**
Download from https://zenodo.org/record/1193957 and extract to `ml/VocalSet/`.

**Run**
```bash
python3 ml/extract_and_umap.py
```

This will:
1. Extract MFCC features from VocalSet recordings
2. Compare PCA, t-SNE, and UMAP dimensionality reduction
3. Select the best method by Silhouette Score
4. Write real zone coordinates to `data/zones.json`
5. Refresh the browser to see the updated map

## Technical Details

- **Pitch detection**: Normalized cross-correlation, search range limited to 60–1100 Hz (human singing range) to avoid detecting harmonics
- **Features**: Spectral Centroid, RMS Energy, Zero-Crossing Rate
- **Smoothing**: Pitch uses 12-frame median filter; other features use exponential moving average (α=0.22)
- **ML**: 13 MFCC coefficients (mean + std = 26-dim feature vector) → StandardScaler → UMAP/PCA/t-SNE → 2D zone map
