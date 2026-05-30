# SingnableLab

Real-time singing visualization tool that helps amateur singers improve emotional expression and self-awareness.

## Problem Statement

1. **Immersion**: Singers lack real-time visual feedback that deepens their emotional engagement while singing
2. **Self-awareness**: Amateur singers don't know what vocal zone they're in or how to improve

## Features

**Immersive Mode** — Real-time cosmic visualization driven by your voice. Color maps to pitch (low = warm orange, high = blue-purple). Positive feedback messages appear when good vocal qualities are detected.

**Vocal Zone Analysis** — Displays your voice's position in a 2D feature space in real-time. Shows which vocal zone you're closest to.

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
│   └── zones.json      # Zone coordinates (ML pipeline output)
└── ml/
    └── extract_and_umap.py   # Python ML pipeline
```

```
**Download VocalSet**
Download from https://zenodo.org/record/1193957 and extract to `ml/VocalSet/`.
```
