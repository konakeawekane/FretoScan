import { Viewport } from './scene/Viewport.js';

const $ = (selector) => document.querySelector(selector);
const view = new Viewport($('#viewport'));
const scanButton = $('#scan');
const panels = { erase: $('#erase-panel'), export: $('#export-panel') };
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

function status(message, active = false) {
  $('#message').textContent = message;
  $('#indicator').classList.toggle('indicator--active', active);
}

function mode(name) {
  for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== name;
  for (const key of ['erase', 'export']) $(`#${key}`).setAttribute('aria-pressed', String(key === name));
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
  frameRequest = video.requestVideoFrameCallback(sampleFrame);
}

function recorderOptions() {
  if (!window.MediaRecorder) return null;
  const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = types.find((type) => MediaRecorder.isTypeSupported?.(type));
  return mimeType ? { mimeType } : {};
}

async function start() {
  if (busy || stream) return;
  if (savedCapture) {
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
    view.showCamera(video);
    chunks = [];
    motion = [];
    frames = [];
    savedCapture = null;
    $('#recording-panel').hidden = true;
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
    timer = setInterval(elapsed, 500);
    status(recorder ? 'Recording camera locally' : 'Camera live · recording unavailable', true);
  } catch (error) {
    clearInterval(timer);
    window.removeEventListener('devicemotion', onMotion);
    if (frameRequest !== undefined) video.cancelVideoFrameCallback?.(frameRequest);
    frameRequest = undefined;
    if (recorder?.state === 'recording') recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
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
  $(`#${key}`).addEventListener('click', () => {
    $('#replace-panel').hidden = true;
    mode(panels[key].hidden ? key : undefined);
  });
}
$('#replace').addEventListener('click', () => { savedCapture = null; start(); });
$('#keep').addEventListener('click', () => { $('#replace-panel').hidden = true; });
$('#save-video').addEventListener('click', () => {
  if (savedCapture) download(savedCapture.video, `${savedCapture.name}.${savedCapture.video.type.includes('mp4') ? 'mp4' : 'webm'}`);
});
$('#save-motion').addEventListener('click', () => {
  if (savedCapture) download(new Blob([JSON.stringify(savedCapture.data)], { type: 'application/json' }), `${savedCapture.name}.json`);
});
window.addEventListener('pagehide', () => stream?.getTracks().forEach((track) => track.stop()));
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
