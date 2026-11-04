# Scan

A static Three.js site for live camera capture and approximate 3D reconstruction. It tracks landmarks across frames, estimates dense depth from nearby views, and fuses depth into a mesh.

## Run

No package manager or build step is required. For desktop testing:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. To use a phone, serve the site over HTTPS; a plain `http://` address on your local network will not grant camera access. Three.js currently loads from a CDN.

Set the camera field of view with the gear button if known, then tap **Scan**. Move sideways slowly while keeping detailed, well-lit surfaces in view. Green dots show anchored tracks; amber dots show relative tracks. Use **Erase** to brush and delete triangles, or **Export** to save a GLB or OBJ mesh. After stopping, save the video and scan data separately. The JSON contains video-frame timestamps and motion readings when available. Everything stays on the device until you download it.

The mesh has an arbitrary scale and may drift or contain gaps. Depth estimation uses classical multi-view stereo, not SimpleMapping's neural model. Motion sensor readings only reject some inconsistent visual estimates; they do not provide full visual-inertial tracking.

## Files

```text
index.html           Page and tools
src/main.js          Capture and UI
src/scene/Viewport.js Three.js viewport and camera preview
src/scan/            Tracking, depth estimation, TSDF fusion, and worker
src/styles/main.css  Interface styles
```
