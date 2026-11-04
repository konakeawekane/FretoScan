# Project specification

## Goal

Build a mobile-friendly website that uses a device camera to construct and display a triangle mesh as the user moves around a space. The scan updates during capture. Standard cameras must work; LiDAR is not required.

## Interface

- A Three.js scene fills the viewport. During scanning, it shows the live camera view and the emerging mesh.
- A small floating menu at the top contains **Scan**, **Erase**, and **Export**.
- The menu and dialogs use a liquid-glass style: translucent layers, highlights, and SVG filter-based refraction. Provide a readable fallback where the effect is unsupported.
- Touch targets, gestures, and layout work on mobile screens.

## Modes

- **Scan:** Request camera permission, start or pause capture, and update visible geometry while the camera moves.
- **Erase:** Brush over the mesh to highlight triangles. A trash button deletes the highlighted triangles. The user can clear the selection without deleting it.
- **Export:** Open a compact menu to download the current mesh as GLB or OBJ.

## Constraints

- Run in the browser without a package manager or build step.
- Process camera frames on the device; do not upload them.
- Keep rendering and scanning responsive on typical mobile hardware by limiting frame processing and mesh detail as needed.
- Camera-only reconstruction is approximate. It requires camera movement, adequate light, and visible surface detail; featureless or hidden surfaces may remain incomplete.

## First-release completion

On a supported phone, a user can grant camera access, move around a scene, watch a mesh grow, remove selected triangles, and download the edited mesh in either format.
