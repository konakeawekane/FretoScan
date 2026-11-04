import { Viewport } from './scene/Viewport.js';

const $ = (selector) => document.querySelector(selector);
const view = new Viewport($('#viewport'));
const scanButton = $('#scan');
const panels = { erase: $('#erase-panel'), export: $('#export-panel') };
const worker = new Worker(new URL('./scan/worker.js', import.meta.url), { type: 'module' });
const sampleCanvas = document.createElement('canvas');
const sampleContext = sampleCanvas.getContext('2d', { willReadFrequently: true });
const video = document.createElement('video');
video.muted = true;
video.autoplay = true;
video.playsInline = true;

let stream;
let recorder;
let chunks = [];
let motion = [];
let frames = [];
let savedCapture;
let startedAt = 0;
let frameRequest;
let timer;
let busy = false;
let processing = false;
let lastScanAt = 0;
let sessionId = 0;
let motionCursor = 0;
const fovInput = $('#fov');
const storedFov = Number(localStorage.getItem('scan-fov'));
if (storedFov >= 40 && storedFov <= 100) fovInput.value = storedFov;
$('#fov-value').textContent = `${fovInput.value}°`;

function status(message, active = false) {
  $('#message').textContent = message;
  $('#indicator').classList.toggle('indicator--active', active);
}

function mode(name) {
  for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== name;
  for (const key of ['erase', 'export']) $(`#${key}`).setAttribute('aria-pressed', String(key === name));
  view.setEraseMode(name === 'erase');
}

function updateMeshUI() {
  const count = view.triangles;
  const selected = view.selectionCount;
  $('#erase-message').textContent = count ? selected ? `${selected} triangle${selected === 1 ? '' : 's'} selected` : 'Brush over triangles to select them.' : 'Scan geometry, then brush over triangles to select them.';
  $('#delete-selection').disabled = selected === 0;
  $('#clear-selection').disabled = selected === 0;
  $('#export-message').textContent = count ? `${count} triangles ready to download.` : 'Scan geometry before exporting.';
  $('#export-glb').disabled = count === 0;
  $('#export-obj').disabled = count === 0;
}

view.onSelectionChange = updateMeshUI;
worker.onmessage = ({ data }) => {
  if (data.sessionId !== sessionId) return;
  processing = false;
  if (data.pose) view.setPose(data.pose, data.focal, data.width);
  if (data.tracks) view.setTracks(data.tracks, data.width, data.height, data.metrics.tracking);
  if (data.type === 'patch') {
    const added = view.addPatch(data.positions, data.colors);
    updateMeshUI();
    status(added ? `K${data.metrics.keyframes} · ${data.metrics.inliers} tracks · ${view.triangles} faces` : 'Move to a new part of the scene', true);
  } else if (data.type === 'volume') {
    view.replaceMesh(data.positions, data.colors);
    updateMeshUI();
    status(`K${data.metrics.keyframes} · ${data.metrics.depthPixels} depth · ${view.triangles} faces`, true);
  } else status(data.message, Boolean(stream));
  if (data.metrics) $('#message').title = `${data.metrics.tracking || 'searching'} tracking · ${data.metrics.keyframes} keyframes · ${data.metrics.matches} matches · ${data.metrics.inliers || 0} inliers · ${data.metrics.landmarks} landmarks · ${data.metrics.depthPixels || 0} depth pixels · ${data.metrics.voxels} voxels`;
};
worker.onerror = () => { processing = false; status('Reconstruction failed; restart the scan'); };

function processFrame(now) {
  if (!stream || processing || now - lastScanAt < 450 || !video.videoWidth) return;
  const scale = 320 / Math.max(video.videoWidth, video.videoHeight);
  const width = Math.round(video.videoWidth * scale), height = Math.round(video.videoHeight * scale);
  if (sampleCanvas.width !== width || sampleCanvas.height !== height) { sampleCanvas.width = width; sampleCanvas.height = height; }
  sampleContext.drawImage(video, 0, 0, width, height);
  const rgba = sampleContext.getImageData(0, 0, width, height).data;
  const samples = motion.slice(motionCursor);
  motionCursor = motion.length;
  processing = true;
  lastScanAt = now;
  worker.postMessage({ type: 'frame', sessionId, width, height, rgba, motion: samples }, [rgba.buffer]);
}

function elapsed() {
  const seconds = Math.floor((performance.now() - startedAt) / 1000);
  $('#timer').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function onMotion(event) {
  const acceleration = event.accelerationIncludingGravity;
  const rotation = event.rotationRate;
  if (!acceleration && !rotation) return;
  motion.push({
    tMs: Math.round(performance.now() - startedAt),
    acceleration: acceleration ? [acceleration.x, acceleration.y, acceleration.z] : null,
    rotation: rotation ? [rotation.alpha, rotation.beta, rotation.gamma] : null,
  });
}

function sampleFrame(now, metadata) {
  if (!stream) return;
  if (recorder?.state === 'recording') frames.push({ tMs: Math.round(now - startedAt), mediaTime: metadata.mediaTime });
  processFrame(now);
  frameRequest = video.requestVideoFrameCallback(sampleFrame);
}

function recorderOptions() {
  if (!window.MediaRecorder) return null;
  const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find((type) => MediaRecorder.isTypeSupported?.(type));
  return mimeType ? { mimeType } : {};
}

async function start(replace = false) {
  if (busy || stream) return;
  if (!replace && (savedCapture || view.triangles)) {
    $('#replace-panel').hidden = false;
    return;
  }
  busy = true;
  scanButton.disabled = true;
  mode();
  $('#replace-panel').hidden = true;
  status('Waiting for camera permission…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.');
    const motionPermission = typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function'
      ? DeviceMotionEvent.requestPermission().catch(() => 'denied')
      : Promise.resolve('granted');
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    video.srcObject = stream;
    await video.play();
    view.clearMesh();
    updateMeshUI();
    sessionId++;
    processing = false;
    lastScanAt = 0;
    worker.postMessage({ type: 'reset', sessionId, fov: Number(fovInput.value) });
    view.showCamera(video);
    chunks = [];
    motion = [];
    motionCursor = 0;
    frames = [];
    savedCapture = null;
    $('#recording-panel').hidden = true;
    $('#calibration-panel').hidden = true;
    $('#calibrate').disabled = true;
    $('#calibrate').setAttribute('aria-expanded', 'false');
    startedAt = performance.now();
    if (await motionPermission === 'granted') window.addEventListener('devicemotion', onMotion);
    const options = recorderOptions();
    if (options) {
      try { recorder = new MediaRecorder(stream, options); }
      catch { recorder = new MediaRecorder(stream); }
      recorder.addEventListener('dataavailable', (event) => { if (event.data.size) chunks.push(event.data); });
      recorder.addEventListener('error', () => status('Recording failed. Stop and try again.'));
      recorder.start(1000);
    }
    if (video.requestVideoFrameCallback) frameRequest = video.requestVideoFrameCallback(sampleFrame);
    stream.getVideoTracks()[0].addEventListener('ended', () => stop());
    scanButton.classList.add('tool--active');
    scanButton.setAttribute('aria-pressed', 'true');
    scanButton.querySelector('span').textContent = 'Stop';
    $('#timer').hidden = false;
    elapsed();
    timer = setInterval(() => { elapsed(); if (!video.requestVideoFrameCallback) processFrame(performance.now()); }, 500);
    status(recorder ? 'Recording camera locally' : 'Camera live · recording unavailable', true);
  } catch (error) {
    clearInterval(timer);
    window.removeEventListener('devicemotion', onMotion);
    if (frameRequest !== undefined) video.cancelVideoFrameCallback?.(frameRequest);
    frameRequest = undefined;
    if (recorder?.state === 'recording') recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    sessionId++;
    processing = false;
    recorder = null;
    video.pause();
    video.srcObject = null;
    view.hideCamera();
    scanButton.classList.remove('tool--active');
    scanButton.setAttribute('aria-pressed', 'false');
    scanButton.querySelector('span').textContent = 'Scan';
    $('#timer').hidden = true;
    const message = error.name === 'NotAllowedError' ? 'Camera permission denied' : error.name === 'NotFoundError' ? 'No camera found' : error.message;
    status(message);
  } finally {
    busy = false;
    scanButton.disabled = false;
    if (!stream) $('#calibrate').disabled = false;
  }
}

async function stop() {
  if (busy || !stream) return;
  busy = true;
  scanButton.disabled = true;
  clearInterval(timer);
  window.removeEventListener('devicemotion', onMotion);
  if (frameRequest !== undefined) video.cancelVideoFrameCallback?.(frameRequest);
  frameRequest = undefined;
  sessionId++;
  processing = false;
  try {
    if (recorder?.state === 'recording') {
      await new Promise((resolve) => {
        recorder.addEventListener('stop', resolve, { once: true });
        recorder.stop();
      });
    }
    if (chunks.length) {
      const type = recorder?.mimeType || chunks[0].type || 'video/webm';
      savedCapture = {
        name: `scan-${new Date().toISOString().replace(/[:.]/g, '-')}`,
        video: new Blob(chunks, { type }),
        data: {
          version: 1,
          mimeType: type,
          width: video.videoWidth,
          height: video.videoHeight,
          durationMs: Math.round(performance.now() - startedAt),
          frames,
          motion,
        },
      };
      $('#recording-panel').hidden = false;
      status('Capture ready to save');
    } else status('Camera stopped');
  } catch (error) {
    status(`Could not finish recording: ${error.message}`);
  } finally {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    recorder = null;
    video.pause();
    video.srcObject = null;
    view.hideCamera();
    scanButton.classList.remove('tool--active');
    scanButton.setAttribute('aria-pressed', 'false');
    scanButton.querySelector('span').textContent = 'Scan';
    $('#timer').hidden = true;
    scanButton.disabled = false;
    $('#calibrate').disabled = false;
    busy = false;
  }
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

scanButton.addEventListener('click', () => stream ? stop() : start());
for (const key of ['erase', 'export']) {
  $(`#${key}`).addEventListener('click', async () => {
    if (stream) await stop();
    $('#replace-panel').hidden = true;
    mode(panels[key].hidden ? key : undefined);
  });
}
$('#calibrate').addEventListener('click', () => {
  const panel = $('#calibration-panel');
  panel.hidden = !panel.hidden;
  $('#calibrate').setAttribute('aria-expanded', String(!panel.hidden));
});
fovInput.addEventListener('input', () => { $('#fov-value').textContent = `${fovInput.value}°`; });
fovInput.addEventListener('change', () => { localStorage.setItem('scan-fov', fovInput.value); status('Calibration saved for next scan'); });
$('#replace').addEventListener('click', () => start(true));
$('#keep').addEventListener('click', () => { $('#replace-panel').hidden = true; });
$('#delete-selection').addEventListener('click', () => { view.deleteSelection(); updateMeshUI(); status(`${view.triangles} triangles remain`); });
$('#clear-selection').addEventListener('click', () => view.clearSelection());
for (const format of ['glb', 'obj']) {
  $(`#export-${format}`).addEventListener('click', async () => {
    if (!view.triangles) return;
    status('Preparing mesh…');
    try {
      const blob = await view.exportMesh(format);
      download(blob, `scan-mesh.${format}`);
      status(`${format.toUpperCase()} ready`);
    } catch (error) { status(`Export failed: ${error.message}`); }
  });
}
$('#save-video').addEventListener('click', () => {
  if (savedCapture) download(savedCapture.video, `${savedCapture.name}.${savedCapture.video.type.includes('mp4') ? 'mp4' : 'webm'}`);
});
$('#save-motion').addEventListener('click', () => {
  if (savedCapture) download(new Blob([JSON.stringify(savedCapture.data)], { type: 'application/json' }), `${savedCapture.name}.json`);
});
window.addEventListener('pagehide', () => stream?.getTracks().forEach((track) => track.stop()));
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
