# Scan

A static Three.js site for live camera capture and approximate 3D reconstruction. It tracks image features between frames, estimates relative camera movement, and triangulates coarse mesh patches.

## Run

No package manager or build step is required. For desktop testing:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. To use a phone, serve the site over HTTPS; a plain `http://` address on your local network will not grant camera access. Three.js currently loads from a CDN.

Tap **Scan** to start and stop capture. Move sideways slowly while keeping detailed, well-lit surfaces in view. Use **Erase** to brush and delete triangles, or **Export** to save a GLB or OBJ mesh. After stopping, save the video and scan data separately. The JSON contains video-frame timestamps and motion readings when available. Everything stays on the device until you download it.

The mesh has an arbitrary scale and may drift or contain gaps. It is a sparse camera-only reconstruction, not SimpleMapping's neural dense mapping pipeline.

## Files

```text
index.html           Page and tools
src/main.js          Capture and UI
src/scene/Viewport.js Three.js viewport and camera preview
src/scan/            Feature tracking and triangulation worker
src/styles/main.css  Interface styles
```
