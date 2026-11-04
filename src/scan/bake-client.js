import { QUALITY } from './bake.js';
import { extractKeyframes } from './frames.js';
import { solveCameras, trackingScene } from './sfm.js';

function request(worker, payload, signal, onStatus) {
  return new Promise((resolve, reject) => {
    const clean = () => {
      worker.removeEventListener('message', done);
      worker.removeEventListener('error', fail);
      signal.removeEventListener('abort', cancel);
    };
    const done = ({ data }) => {
      if (data.type === 'status') { onStatus?.(data); return; }
      clean();
      if (data.type === 'error') {
        const error = new Error(data.message);
        if (data.stack) error.stack = data.stack;
        reject(error);
      }
      else resolve(data);
    };
    const fail = (event) => { clean(); reject(new Error(`Mesh worker failed${event.message ? `: ${event.message}` : ''}`)); };
    const cancel = () => { clean(); reject(new DOMException('Bake canceled', 'AbortError')); };
    worker.addEventListener('message', done);
    worker.addEventListener('error', fail);
    signal.addEventListener('abort', cancel, { once: true });
    if (signal.aborted) { cancel(); return; }
    worker.postMessage(payload);
  });
}

export async function bakeCapture(capture, quality, onProgress, signal, { neural = true } = {}) {
  const settings = QUALITY[quality];
  if (!settings) throw new Error('Unknown quality level');
  onProgress({ phase: 'frames', done: 0, total: 1 });
  const keyframes = await extractKeyframes(capture.video, {
    count: settings.maxFrames, size: settings.solveSize, rate: settings.rate, durationMs: capture.data.durationMs, signal,
    onProgress: (done, total) => onProgress({ phase: 'frames', done, total }),
  });
  const frames = keyframes.frames;
  if (frames.length < 3) throw new Error('The recording is too short. Record a longer scan.');

  let scene = null, poseSource = 'tracking', warning;
  try {
    scene = await solveCameras(frames, { signal, onProgress: (progress) => onProgress({ phase: 'poses', ...progress }) });
    if (scene.registered < Math.max(3, frames.length * 0.4)) {
      warning = `Camera solve placed only ${scene.registered} of ${frames.length} frames`;
      scene = null;
    } else poseSource = 'sfm';
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    warning = `Camera solve unavailable: ${error.message}`;
    console.warn(error);
  }
  if (!scene) {
    const tracked = trackingScene(frames, capture.data);
    if (tracked.registered >= 3) scene = tracked;
  }
  if (!scene) throw new Error('Could not recover camera motion. Move sideways slowly over a detailed, well-lit surface.');

  const usable = frames.flatMap((frame, i) => {
    const view = scene.views[i];
    return view && view.landmarks.length >= 8 ? [{ frame, view }] : [];
  });
  if (usable.length < 3) throw new Error('Not enough tracked frames with 3D points. Record a longer scan with more sideways motion.');

  const worker = new Worker(new URL('./bake-worker.js', import.meta.url), { type: 'module' });
  try {
    onProgress({ phase: 'model', message: neural ? 'Loading depth model' : 'Preparing mesh worker' });
    const ready = await request(worker, { type: 'start', quality, neural }, signal,
      (status) => onProgress({ phase: 'model', message: status.message }));
    if (ready.warning) warning = warning ? `${warning}. ${ready.warning}` : ready.warning;
    let metrics = {};
    for (let i = 0; i < usable.length; i++) {
      const { frame, view } = usable[i];
      const response = await request(worker, { type: 'frame', frame: {
        blob: frame.blob, width: frame.width, height: frame.height, sharpness: frame.sharpness,
        pose: { R: view.R, C: view.C }, focal: view.focal ?? scene.focal, landmarks: view.landmarks,
      } }, signal);
      metrics = response.metrics;
      onProgress({ phase: 'depth', done: i + 1, total: usable.length, metrics });
    }
    onProgress({ phase: 'mesh', metrics });
    const result = await request(worker, { type: 'finish' }, signal);
    if (!result.positions?.length) throw new Error('No surface was reconstructed. Try more sideways motion or better lighting.');
    return { ...result, selected: usable.length, keyframes: frames.length, poseSource, decoder: keyframes.method,
      neural: ready.neural, warning };
  } finally {
    worker.terminate();
  }
}
