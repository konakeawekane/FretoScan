# Project specification

## Goal

Build a mobile-friendly website that records a moving device camera and constructs a mesh after capture. Standard cameras must work; LiDAR is not required.

## Interface

- A Three.js scene fills the viewport. During capture, it shows the live camera view and followed image points. After processing, it shows the mesh.
- A small floating menu at the top contains **Scan**, **Erase**, and **Export**.
- The menu and dialogs use a liquid-glass style: translucent layers, highlights, and SVG filter-based refraction. Provide a readable fallback where the effect is unsupported.
- Touch targets, gestures, and layout work on mobile screens.

## Modes

- **Scan:** Request camera permission, follow image features, estimate camera pose, and record video and tracking data. After stopping, ask for Quarter, Half, or Full quality and build the mesh from the video.
- **Erase:** Brush over triangles and delete the selection.
- **Export:** Download the mesh as GLB or OBJ.

## Constraints

- Run in the browser without a package manager or build step. Libraries may load from a CDN.
- Process camera frames on the device; do not upload them.
- Keep tracking responsive on typical mobile hardware. Run dense reconstruction after capture in workers, with progress and cancellation.
- Bake from the recorded video: select sharp keyframes, solve camera poses with structure from motion, estimate dense depth, and fuse it into a mesh. Use WebGPU where available and fall back to live tracking poses where it is not.
- Camera-only reconstruction is approximate. It requires camera movement, adequate light, and visible surface detail; featureless or hidden surfaces may remain incomplete.

## Completion

On a supported phone, a user can record a scan, choose a quality level, wait for a baked mesh, erase triangles, and export the result. The video and tracking data remain available for download.
