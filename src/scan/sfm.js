import { COLMAP_URL } from './vendor.js';
import { mul3, multiply3, transpose3 } from './reconstruction.js';

const aborted = () => new DOMException('Bake canceled', 'AbortError');
const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];

function call(worker, message, signal, onLog) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(aborted()); return; }
    const cancel = () => reject(aborted());
    signal.addEventListener('abort', cancel, { once: true });
    worker.onmessage = ({ data }) => {
      if (data.kind === 'log') { onLog?.(data.text); return; }
      signal.removeEventListener('abort', cancel);
      if (data.kind === 'error') reject(new Error(`COLMAP: ${data.text}`));
      else resolve(data);
    };
    worker.onerror = (event) => {
      signal.removeEventListener('abort', cancel);
      reject(new Error(`COLMAP worker: ${event.message || 'failed to load'}`));
    };
    worker.postMessage(message);
  });
}

export function colmapToScene(model, keypoints, count) {
  const views = [];
  for (let i = 0; i < count; i++) {
    const p = model.poses.subarray(13 * i, 13 * i + 13);
    if (!p[0]) { views.push(null); continue; }
    const R = Array.from(p.subarray(1, 10));
    const C = mul3(transpose3(R), [p[10], p[11], p[12]]).map((value) => -value);
    views.push({ R: [R[0], R[1], R[2], -R[3], -R[4], -R[5], R[6], R[7], R[8]], C, landmarks: [] });
  }
  const anchor = views.find(Boolean);
  if (!anchor) return { views, focal: model.camera[0], registered: 0, points: 0 };
  const R0 = anchor.R, C0 = anchor.C;
  const points = [];
  const depths = [];
  for (let j = 0, at = 0; j < model.points.length / 4; j++) {
    const length = model.tracks[at++];
    const track = [];
    for (let k = 0; k < length; k++, at += 2) track.push([model.tracks[at], model.tracks[at + 1]]);
    if (model.points[4 * j + 3] > 2.5) continue;
    const world = [model.points[4 * j], model.points[4 * j + 1], model.points[4 * j + 2]];
    points.push({ id: j, world, track });
    for (const [image] of track) {
      const view = views[image];
      if (view) depths.push(mul3(view.R, world.map((value, i) => value - view.C[i]))[2]);
    }
  }
  const scale = depths.length ? 1 / Math.max(1e-9, median(depths.filter((value) => value > 0))) : 1;
  const move = (world) => mul3(R0, world.map((value, i) => value - C0[i])).map((value) => value * scale);
  for (const view of views) if (view) {
    view.C = move(view.C);
    view.R = multiply3(view.R, transpose3(R0));
  }
  for (const point of points) {
    const world = move(point.world);
    for (const [image, index] of point.track) {
      const view = views[image];
      if (!view) continue;
      const kp = keypoints[image];
      view.landmarks.push({ id: point.id, x: kp[6 * index] - 0.5, y: kp[6 * index + 1] - 0.5, world });
    }
  }
  return { views, focal: model.camera[0], registered: views.filter(Boolean).length, points: points.length };
}

export async function solveCameras(frames, { signal, onProgress }) {
  if (!navigator.gpu || !(await navigator.gpu.requestAdapter())) throw new Error('WebGPU is unavailable');
  const { SiftMatcher, sequentialPairs } = await import(COLMAP_URL);
  const memory = navigator.deviceMemory ?? 8;
  const count = Math.max(1, Math.min(frames.length, (navigator.hardwareConcurrency || 4) - 1, 4, Math.floor(memory / 1.5)));
  const workers = Array.from({ length: count }, () => new Worker(new URL('./colmap-worker.js', import.meta.url), { type: 'module' }));
  const features = new Array(frames.length);
  try {
    let next = 0, done = 0;
    await Promise.all(workers.map(async (worker) => {
      while (next < frames.length) {
        const id = next++;
        features[id] = await call(worker, { kind: 'extract', id, data: frames[id].blob }, signal);
        onProgress?.({ stage: 'features', done: ++done, total: frames.length });
      }
    }));
    const pairs = sequentialPairs(frames.length, { window: 6, jumps: [10, 16, 24, 32], loop: frames.length > 12 });
    const matcher = await SiftMatcher.create();
    let matches;
    try {
      matches = await matcher.match(features.map((feature) => feature.descriptors), pairs,
        { onProgress: (done, total) => onProgress?.({ stage: 'matching', done, total }) });
    } finally {
      matcher.destroy();
    }
    if (signal.aborted) throw aborted();
    onProgress?.({ stage: 'mapping', done: 0, total: 1 });
    const { width, height } = features[0];
    const model = await call(workers[0], {
      kind: 'map', width, height, names: frames.map((_, i) => `frame-${String(i).padStart(4, '0')}.jpg`),
      keypoints: features.map((feature) => feature.keypoints), matches, seed: 0,
    }, signal);
    onProgress?.({ stage: 'mapping', done: 1, total: 1 });
    return { ...colmapToScene(model, features.map((feature) => feature.keypoints), frames.length), width, height };
  } finally {
    for (const worker of workers) worker.terminate();
  }
}

export function trackingScene(frames, data) {
  const tracking = data.tracking.filter((entry) => entry.pose?.R?.every(Number.isFinite) && entry.pose?.C?.every(Number.isFinite)
    && Number.isFinite(entry.focal) && entry.width > 0 && Number.isFinite(entry.tMs)
    && entry.landmarks?.filter(({ world }) => world?.every(Number.isFinite)).length >= 8);
  const views = frames.map((frame) => {
    const tMs = data.recordStartMs + frame.timeS * 1000;
    let best = null;
    for (const entry of tracking) if (!best || Math.abs(entry.tMs - tMs) < Math.abs(best.tMs - tMs)) best = entry;
    if (!best || Math.abs(best.tMs - tMs) > 250) return null;
    const sx = frame.width / best.width, sy = frame.height / best.height;
    return {
      R: best.pose.R, C: best.pose.C, focal: best.focal * sx,
      landmarks: best.landmarks.filter(({ world }) => world?.every(Number.isFinite)).map(({ id, x, y, world }) => ({ id, x: x * sx, y: y * sy, world })),
    };
  });
  return { views, registered: views.filter(Boolean).length };
}
