# Resonance

A lightweight, build-free Three.js foundation for authoring virtual acoustic environments.

## Getting started

No package manager or build step is required. Because JavaScript modules cannot be loaded reliably from a `file://` URL, serve the folder with any basic static server. For example, if Python is installed:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`. Three.js is loaded directly from unpkg through the import map in `index.html`, so an internet connection is required on first load.

## Structure

```text
src/
├── main.js                 # Application entry point
├── scene/
│   └── AcousticScene.js    # Three.js scene lifecycle
└── styles/
    └── main.css            # Application styles
```

The current scaffold provides the editor shell and a navigable 3D viewport. Room editing, audio modeling, persistence, and camera-based scanning are intentionally left as future modules.
