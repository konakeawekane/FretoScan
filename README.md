# Scan

A static Three.js site for live camera capture. Scan records the device camera and available motion readings. Geometry reconstruction is the next phase.

## Run

No package manager or build step is required. For desktop testing:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. To use a phone, serve the site over HTTPS; a plain `http://` address on your local network will not grant camera access. Three.js currently loads from a CDN.

Tap **Scan** to start and stop capture. After stopping, save the video and scan data separately. The JSON contains video-frame timestamps and motion readings when available. Everything stays on the device until you download it.

## Files

```text
index.html           Page and tools
src/main.js          Capture and UI
src/scene/Viewport.js Three.js viewport and camera preview
src/styles/main.css  Interface styles
```
