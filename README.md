# Scan

A static Three.js site for camera capture and post-capture mesh reconstruction. Live capture tracks image features and camera pose; mesh generation runs after recording.

## Run

No package manager or build step is required. For desktop testing:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. To use a phone, serve the site over HTTPS; a plain `http://` address on your local network will not grant camera access.

Set the camera field of view with the gear button if known, then tap **Scan**. Keep detailed, well-lit surfaces in view and move sideways slowly. Amber dots show image tracking, blue dots show a relative pose, and green dots show landmark-anchored tracking. After stopping, choose **Quarter**, **Half**, or **Full** to build a mesh from the recorded video. The bake shows progress and can be canceled. Use **Erase** and **Export** after the mesh is ready. Save the video and scan data separately; the JSON includes timestamps, tracks, landmarks, and poses. Processing stays on the device.

Higher quality uses more video frames and views, larger images, and finer voxels. Monocular scale is arbitrary. The reconstruction needs overlapping views with sideways motion; blur and hidden surfaces may leave gaps.

## Bake pipeline

1. **Keyframes** (`frames.js`, `webm.js`, `sharpness.js`): the recording is decoded with WebCodecs (mp4box.js for MP4, a small parser for WebM), or by playing the video where WebCodecs is missing. The sharpest frame in each time slot is kept.
2. **Camera poses** (`sfm.js`): with WebGPU, colmap-wasm extracts SIFT features, matches nearby and loop-closing frame pairs, and solves all cameras with a global mapper. If WebGPU is missing or too few frames register, the live tracking poses are used.
3. **Depth** (`mono.js`, `depth.js`, `bake.js`): Depth Anything V2 (transformers.js) predicts a depth prior per keyframe, aligned to the solved points. Plane-sweep stereo refines it; the prior fills plain surfaces where stereo fails. Depth that disagrees with neighbouring views is dropped.
4. **Mesh** (`volume.js`, `mesh.js`): depth maps are fused into a sparse TSDF volume, extracted with marching tetrahedra, cleaned of small fragments, smoothed, decimated to the quality's face limit, and coloured from the sharpest view that sees each vertex.

The neural depth toggle in the quality panel downloads the model (about 50 MB) once and runs it on WebGPU, or on WebAssembly otherwise. Without it, plain surfaces may stay open.

## Dependencies

Loaded from CDNs at pinned versions: Three.js (unpkg, in `index.html`), and mp4box.js, colmap-wasm, and transformers.js (jsDelivr, in `src/scan/vendor.js`). Model weights come from Hugging Face. Chrome or Edge with WebGPU gives the best results; other browsers fall back to live tracking poses and slower depth inference.

## Tests

```bash
node --test tests/reconstruction.test.mjs
```

`tests/bake.html` renders a synthetic scene, records it, and runs the full bake in the browser. Query options: `quality=quarter|half|full`, `neural=1`, `seconds=6`, `webm=1` (record VP8), and `playback=1` (skip WebCodecs). Serve with caching disabled while editing worker code, or the worker may load stale modules.

## Files

```text
index.html           Page and tools
src/main.js          Capture and UI
src/scene/Viewport.js Three.js viewport and camera preview
src/scan/            Live tracker and post-capture reconstruction workers
src/styles/main.css  Interface styles
tests/               Unit tests and browser bake harness
```
