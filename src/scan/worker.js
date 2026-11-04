import { grayscale, detect, match } from './features.js';
import { estimatePose, meshFaces, initialPose, advancePose, toWorld } from './reconstruction.js';

let previous = null;
let pose = initialPose();
let misses = 0;

self.onmessage = ({ data }) => {
  if (data.type === 'reset') {
    previous = null;
    pose = initialPose();
    misses = 0;
    return;
  }
  const { width, height, rgba, sessionId } = data;
  const gray = grayscale(rgba, width, height);
  const current = { width, height, rgba, gray, features: detect(gray, width, height) };
  if (!previous || previous.width !== width || previous.height !== height) {
    previous = current;
    self.postMessage({ type: 'result', sessionId, message: 'Move sideways slowly to reveal depth' });
    return;
  }
  const pairs = match(previous.gray, gray, previous.features, width, height);
  if (pairs.length < 25) {
    if (++misses > 2) { previous = current; misses = 0; }
    self.postMessage({ type: 'result', sessionId, message: 'Point at a detailed, well-lit surface' });
    return;
  }
  const flows = pairs.map((pair) => Math.hypot(pair.x2 - pair.x1, pair.y2 - pair.y1)).sort((a, b) => a - b);
  if (flows[Math.floor(flows.length / 2)] < 5) {
    self.postMessage({ type: 'result', sessionId, message: 'Move sideways slowly to reveal depth' });
    return;
  }
  const estimate = estimatePose(pairs, width, height);
  if (!estimate) {
    if (++misses > 2) { previous = current; misses = 0; }
    self.postMessage({ type: 'result', sessionId, message: 'Tracking uncertain; move slowly and keep surfaces in view' });
    return;
  }
  misses = 0;
  const faces = meshFaces(estimate.points, width, height);
  const positions = new Float32Array(faces.length * 9);
  const colors = new Float32Array(faces.length * 9);
  let offset = 0;
  for (const face of faces) for (const id of face) {
    const point = estimate.points[id];
    const world = toWorld(point.point, pose);
    positions.set([world[0], world[1], -world[2]], offset);
    const pixel = (Math.round(point.y) * width + Math.round(point.x)) * 4;
    colors.set([previous.rgba[pixel] / 255, previous.rgba[pixel + 1] / 255, previous.rgba[pixel + 2] / 255], offset);
    offset += 3;
  }
  pose = advancePose(pose, estimate);
  previous = current;
  if (!faces.length) {
    self.postMessage({ type: 'result', sessionId, message: 'Depth found; more surface detail needed for triangles' });
    return;
  }
  self.postMessage({ type: 'patch', sessionId, positions, colors, pose, focal: estimate.focal, width, height, points: estimate.points.length, triangles: faces.length }, [positions.buffer, colors.buffer]);
};
