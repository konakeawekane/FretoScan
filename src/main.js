import { Viewport } from './scene/Viewport.js';
import { bakeCapture } from './scan/bake-client.js';

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
let trackingLog = [];
let savedCapture;
let activeBake;
let recorderStartedAt = 0;
let startedAt = 0;
let frameRequest;
let timer;
let busy = false;
let processing = false;
let lastScanAt = 0;
let sessionId = 0;
const neuralInput = $('#neural-depth');
neuralInput.checked = localStorage.getItem('scan-neural') ? localStorage.getItem('scan-neural') === 'on' : Boolean(navigator.gpu);
neuralInput.addEventListener('change', () => localStorage.setItem('scan-neural', neuralInput.checked ? 'on' : 'off'));
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
  $('#erase-message').textContent = count ? selected ? `${selected} triangle${selected === 1 ? '' : 's'} selected` : 'Brush over triangles to select them.' : 'Build a mesh, then brush over triangles.';
  $('#delete-selection').disabled = selected === 0 || Boolean(activeBake);
  $('#clear-selection').disabled = selected === 0 || Boolean(activeBake);
  $('#export-message').textContent = count ? `${count} triangles ready to download.` : 'Build a mesh before exporting.';
  $('#export-glb').disabled = !count || Boolean(activeBake);
  $('#export-obj').disabled = !count || Boolean(activeBake);
  for (const key of ['erase', 'export']) {
    const button = $(`#${key}`);
    button.disabled = !count || Boolean(stream) || Boolean(activeBake);
    button.setAttribute('aria-label', button.disabled ? `${key}, available after mesh bake` : key);
    button.title = button.disabled ? 'Available after mesh bake' : '';
  }
}

view.onSelectionChange = updateMeshUI;
worker.onmessage = ({ data }) => {
  if (data.sessionId !== sessionId) return;
  processing = false;
  if (data.pose) view.setPose(data.pose, data.focal, data.width);
  if (data.tracks) view.setTracks(data.tracks, data.width, data.height, data.metrics.tracking);
  trackingLog.push({ tMs: data.tMs, mediaTime: data.mediaTime, pose: data.pose || null, focal: data.focal,
    width: data.width, height: data.height, metrics: data.metrics, tracks: data.tracks, ids: data.trackIds,
    landmarks: data.landmarks });
  status(data.message, Boolean(stream));
  $('#message').title = `${data.metrics.matches} followed features · ${data.metrics.landmarks} pose landmarks · ${Math.round(data.metrics.displacementPx || 0)} px keyframe motion`;
};
worker.onerror = () => { processing = false; status('Tracking failed; restart the scan'); };

function processFrame(now, mediaTime) {
  if (!stream || processing || now - lastScanAt < 120 || !video.videoWidth) return;
  const scale = 320 / Math.max(video.videoWidth, video.videoHeight);
  const width = Math.round(video.videoWidth * scale), height = Math.round(video.videoHeight * scale);
  if (sampleCanvas.width !== width || sampleCanvas.height !== height) { sampleCanvas.width = width; sampleCanvas.height = height; }
  sampleContext.drawImage(video, 0, 0, width, height);
  const rgba = sampleContext.getImageData(0, 0, width, height).data;
  processing = true;
  lastScanAt = now;
  worker.postMessage({ type: 'frame', sessionId, tMs: Math.round(now - startedAt), mediaTime, width, height, rgba }, [rgba.buffer]);
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
  processFrame(now, metadata.mediaTime);
  frameRequest = video.requestVideoFrameCallback(sampleFrame);
}

function recorderOptions() {
  if (!window.MediaRecorder) return null;
  const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find((type) => MediaRecorder.isTypeSupported?.(type));
  return mimeType ? { mimeType } : {};
}

async function start(replace = false) {
  if (busy || stream || activeBake) return;
  if (!replace && (savedCapture || view.triangles)) {
    $('#replace-panel').hidden = false;
    $('#quality-panel').hidden = true;
    return;
  }
  busy = true;
  scanButton.disabled = true;
  mode();
  $('#replace-panel').hidden = true;
  $('#quality-panel').hidden = true;
  status('Waiting for camera permission…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.');
    if (!window.MediaRecorder) throw new Error('Video recording is unavailable in this browser.');
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
    frames = [];
    trackingLog = [];
    savedCapture = null;
    $('#recording-panel').hidden = true;
    $('#calibration-panel').hidden = true;
    $('#calibrate').disabled = true;
    $('#calibrate').setAttribute('aria-expanded', 'false');
    startedAt = performance.now();
    recorderStartedAt = 0;
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission !== 'function') window.addEventListener('devicemotion', onMotion);
    const options = recorderOptions();
    if (options) {
      try { recorder = new MediaRecorder(stream, options); }
      catch { recorder = new MediaRecorder(stream); }
      recorder.addEventListener('dataavailable', (event) => { if (event.data.size) chunks.push(event.data); });
      recorder.addEventListener('error', () => status('Recording failed. Stop and try again.'));
      recorder.start(1000);
      recorderStartedAt = Math.round(performance.now() - startedAt);
    }
    if (video.requestVideoFrameCallback) frameRequest = video.requestVideoFrameCallback(sampleFrame);
    stream.getVideoTracks()[0].addEventListener('ended', () => stop());
    scanButton.classList.add('tool--active');
    scanButton.setAttribute('aria-pressed', 'true');
    scanButton.querySelector('span').textContent = 'Stop';
    $('#timer').hidden = false;
    elapsed();
    timer = setInterval(() => { elapsed(); if (!video.requestVideoFrameCallback) processFrame(performance.now()); }, 150);
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
          version: 2,
          mimeType: type,
          width: video.videoWidth,
          height: video.videoHeight,
          durationMs: Math.round(performance.now() - startedAt),
          recordStartMs: recorderStartedAt,
          frames,
          motion,
          tracking: trackingLog,
        },
      };
      $('#recording-panel').hidden = true;
      $('#quality-description').textContent = 'Choose a quality level. Full may take several minutes.';
      $('#bake-status').hidden = true;
      $('#quality-panel').hidden = false;
      status('Capture ready; choose mesh quality');
    } else status('Camera stopped; recording unavailable');
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
    updateMeshUI();
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

function bakeProgress({ phase, stage, done = 0, total = 1, message, metrics }) {
  const part = total ? Math.min(1, done / total) : 0;
  if (phase === 'frames') return [Math.round(part * 20), 'Picking sharp video frames'];
  if (phase === 'poses') {
    if (stage === 'features') return [20 + Math.round(part * 14), `Finding image features ${done} of ${total}`];
    if (stage === 'matching') return [34 + Math.round(part * 6), 'Matching frames'];
    return [40 + Math.round(part * 5), 'Solving camera positions'];
  }
  if (phase === 'model') return [46, message];
  if (phase === 'depth') return [47 + Math.round(part * 48), `Estimating depth ${done} of ${total} - ${metrics.fused} depth views`];
  return [97, 'Fusing views and extracting mesh'];
}

async function buildMesh(quality) {
  if (!savedCapture || activeBake || stream) return;
  mode();
  const controller = new AbortController();
  activeBake = controller;
  scanButton.disabled = true;
  $('#calibrate').disabled = true;
  for (const button of document.querySelectorAll('.quality')) button.disabled = true;
  neuralInput.disabled = true;
  $('#cancel-bake').hidden = false;
  $('#close-quality').hidden = true;
  $('#bake-progress').hidden = false;
  $('#bake-progress').value = 0;
  $('#bake-status').hidden = false;
  $('#bake-status').textContent = 'Preparing video';
  updateMeshUI();
  status(`Baking ${quality} mesh`, true);
  try {
    const result = await bakeCapture(savedCapture, quality, (progress) => {
      const [value, message] = bakeProgress(progress);
      $('#bake-progress').value = value;
      $('#bake-status').textContent = message;
      status(message, true);
    }, controller.signal, { neural: $('#neural-depth').checked });
    view.replaceMesh(result.positions, result.colors);
    $('#quality-panel').hidden = true;
    $('#recording-panel').hidden = false;
    const source = result.poseSource === 'sfm' ? 'solved cameras' : 'live tracking poses';
    const notes = [result.metrics.saturated && 'volume limit reached', result.warning].filter(Boolean).join('; ');
    status(`Mesh ready - ${view.triangles} triangles from ${result.selected} views (${source})${notes ? ` · ${notes}` : ''}`);
  } catch (error) {
    const message = error.name === 'AbortError' ? 'Mesh bake canceled' : `Mesh bake failed: ${error.message}`;
    $('#bake-status').textContent = message;
    status(message);
  } finally {
    activeBake = null;
    scanButton.disabled = false;
    $('#calibrate').disabled = false;
    for (const button of document.querySelectorAll('.quality')) button.disabled = false;
    neuralInput.disabled = false;
    $('#cancel-bake').hidden = true;
    $('#close-quality').hidden = false;
    $('#bake-progress').hidden = true;
    updateMeshUI();
  }
}

scanButton.addEventListener('click', () => stream ? stop() : start());
$('#open-quality').addEventListener('click', () => {
  if (!savedCapture || activeBake) return;
  mode();
  $('#quality-panel').hidden = false;
  $('#recording-panel').hidden = true;
  $('#quality-description').textContent = view.triangles ? 'Choose a quality level to replace the current mesh.' : 'Choose a quality level. Full may take several minutes.';
  $('#bake-status').hidden = true;
});
for (const button of document.querySelectorAll('.quality')) button.addEventListener('click', () => buildMesh(button.dataset.quality));
$('#cancel-bake').addEventListener('click', () => activeBake?.abort());
$('#close-quality').addEventListener('click', () => {
  if (activeBake) return;
  $('#quality-panel').hidden = true;
  $('#recording-panel').hidden = false;
});
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
$('#keep').addEventListener('click', () => {
  $('#replace-panel').hidden = true;
  $('#quality-panel').hidden = !savedCapture;
  $('#recording-panel').hidden = Boolean(savedCapture);
});
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
window.addEventListener('pagehide', () => { activeBake?.abort(); stream?.getTracks().forEach((track) => track.stop()); });
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
