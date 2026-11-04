import { MP4BOX_URL } from './vendor.js';
import { parseWebm } from './webm.js';
import { grayscale } from './features.js';
import { sharpness, KeyframePicker } from './sharpness.js';

const aborted = () => new DOMException('Bake canceled', 'AbortError');
const fit = (width, height, size) => {
  const scale = Math.min(1, size / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
};

class Collector {
  constructor(width, height, size, durationS, count) {
    [this.width, this.height] = fit(width, height, size);
    [this.thumbWidth, this.thumbHeight] = fit(width, height, 320);
    this.thumb = new OffscreenCanvas(this.thumbWidth, this.thumbHeight).getContext('2d', { willReadFrequently: true });
    this.canvas = new OffscreenCanvas(this.width, this.height);
    this.context = this.canvas.getContext('2d');
    this.encoder = new OffscreenCanvas(this.width, this.height);
    this.encoderContext = this.encoder.getContext('2d');
    this.picker = new KeyframePicker(durationS, count);
    this.frames = [];
    this.pending = null;
    this.encoding = Promise.resolve();
    this.seen = 0;
  }

  add(source, timeS) {
    this.seen++;
    this.thumb.drawImage(source, 0, 0, this.thumbWidth, this.thumbHeight);
    const rgba = this.thumb.getImageData(0, 0, this.thumbWidth, this.thumbHeight).data;
    const score = sharpness(grayscale(rgba, this.thumbWidth, this.thumbHeight), this.thumbWidth, this.thumbHeight);
    const { flush, keep } = this.picker.offer(timeS, score);
    if (flush) this.flush();
    if (keep) {
      this.context.drawImage(source, 0, 0, this.width, this.height);
      this.pending = { timeS, sharpness: score };
    }
  }

  flush() {
    if (!this.pending) return;
    const meta = this.pending;
    const bitmap = this.canvas.transferToImageBitmap();
    this.pending = null;
    this.encoding = this.encoding.then(async () => {
      this.encoderContext.drawImage(bitmap, 0, 0);
      bitmap.close();
      const blob = await this.encoder.convertToBlob({ type: 'image/jpeg', quality: 0.93 });
      this.frames.push({ ...meta, blob, width: this.width, height: this.height });
    });
  }

  async finish() {
    this.flush();
    await this.encoding;
    return this.frames;
  }
}

async function demuxMp4(buffer) {
  const { createFile, DataStream, Endianness, MP4BoxBuffer } = await import(MP4BOX_URL);
  const file = createFile();
  let track, failure;
  const samples = [];
  file.onError = (module, message) => { failure = new Error(`MP4: ${message}`); };
  file.onReady = (info) => {
    track = info.videoTracks[0];
    if (!track) return;
    file.setExtractionOptions(track.id, null, { nbSamples: 1e9 });
    file.start();
  };
  file.onSamples = (id, user, list) => {
    for (const sample of list) {
      samples.push({ timestamp: sample.cts * 1e6 / sample.timescale, duration: sample.duration * 1e6 / sample.timescale,
        key: sample.is_sync, data: sample.data });
    }
  };
  file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, 0), true);
  file.flush();
  if (failure) throw failure;
  if (!track) throw new Error('No video track in MP4');
  let description;
  for (const entry of file.getTrackById(track.id).mdia.minf.stbl.stsd.entries) {
    const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
    if (!box) continue;
    const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
    box.write(stream);
    description = new Uint8Array(stream.buffer, 8);
  }
  return { width: track.video.width, height: track.video.height, configs: [{ codec: track.codec, description }], samples };
}

async function supportedConfig(track) {
  for (const candidate of track.configs) {
    const config = { ...candidate, codedWidth: track.width, codedHeight: track.height };
    if (!config.description) delete config.description;
    try { if ((await VideoDecoder.isConfigSupported(config)).supported) return config; } catch { }
  }
  throw new Error('Video codec is not supported by WebCodecs');
}

async function decodeKeyframes(blob, options) {
  const buffer = await blob.arrayBuffer();
  const head = new Uint8Array(buffer, 0, Math.min(12, buffer.byteLength));
  const isMp4 = String.fromCharCode(...head.subarray(4, 8)) === 'ftyp';
  const track = isMp4 ? await demuxMp4(buffer) : parseWebm(new Uint8Array(buffer));
  if (!track.samples.length || !track.width || !track.height) throw new Error('No video frames found');
  const config = await supportedConfig(track);
  const firstKey = track.samples.findIndex((sample) => sample.key);
  if (firstKey < 0) throw new Error('Video has no keyframe');
  const samples = track.samples.slice(firstKey);
  const start = Math.min(...samples.map((sample) => sample.timestamp));
  const durationS = Math.max(...samples.map((sample) => sample.timestamp)) / 1e6 - start / 1e6;
  const collector = new Collector(track.width, track.height, options.size, Math.max(durationS, 0.1), options.count);
  let failure;
  const decoder = new VideoDecoder({
    output: (frame) => {
      try { collector.add(frame, (frame.timestamp - start) / 1e6); } catch (error) { failure = error; } finally { frame.close(); }
    },
    error: (error) => { failure = error; },
  });
  decoder.configure(config);
  try {
    for (let i = 0; i < samples.length; i++) {
      if (options.signal.aborted) throw aborted();
      if (failure) throw failure;
      const sample = samples[i];
      decoder.decode(new EncodedVideoChunk({ type: sample.key ? 'key' : 'delta', timestamp: Math.round(sample.timestamp),
        duration: sample.duration ? Math.round(sample.duration) : undefined, data: sample.data }));
      while (decoder.decodeQueueSize > 6) await new Promise((resolve) => setTimeout(resolve, 2));
      if (i % 10 === 0) options.onProgress?.(i, samples.length);
    }
    await decoder.flush();
  } finally {
    if (decoder.state !== 'closed') decoder.close();
  }
  if (failure) throw failure;
  if (collector.seen < samples.length * 0.5) throw new Error('Video decoder dropped most frames');
  return { frames: await collector.finish(), method: 'webcodecs' };
}

function waitFor(target, name, signal, ms = 15000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { clean(); reject(new Error('Video decoding timed out')); }, ms);
    const done = () => { clean(); resolve(); };
    const fail = () => { clean(); reject(new Error('Recorded video could not be decoded')); };
    const cancel = () => { clean(); reject(aborted()); };
    const clean = () => {
      clearTimeout(timer);
      target.removeEventListener(name, done);
      target.removeEventListener('error', fail);
      signal.removeEventListener('abort', cancel);
    };
    target.addEventListener(name, done, { once: true });
    target.addEventListener('error', fail, { once: true });
    signal.addEventListener('abort', cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

async function playbackKeyframes(blob, options) {
  const player = document.createElement('video');
  player.muted = player.defaultMuted = player.playsInline = true;
  player.setAttribute('muted', '');
  player.setAttribute('playsinline', '');
  player.preload = 'auto';
  player.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none';
  document.body.append(player);
  const url = URL.createObjectURL(blob);
  let callback;
  try {
    const loaded = waitFor(player, 'loadeddata', options.signal);
    player.src = url;
    player.load();
    await loaded;
    if (!player.videoWidth || !player.videoHeight) throw new Error('Recorded video has no readable frames');
    const durationS = Number.isFinite(player.duration) && player.duration > 0.1 ? player.duration : options.durationMs / 1000;
    const collector = new Collector(player.videoWidth, player.videoHeight, options.size, durationS, options.count);
    let last = -1;
    const sample = (now, metadata) => {
      const time = metadata?.mediaTime ?? player.currentTime;
      if (time > last && player.readyState >= 2) {
        last = time;
        collector.add(player, time);
        options.onProgress?.(Math.min(time, durationS), durationS);
      }
      callback = player.requestVideoFrameCallback ? player.requestVideoFrameCallback(sample) : requestAnimationFrame(sample);
    };
    const ended = waitFor(player, 'ended', options.signal, (durationS / options.rate + 20) * 1000);
    player.playbackRate = options.rate;
    callback = player.requestVideoFrameCallback ? player.requestVideoFrameCallback(sample) : requestAnimationFrame(sample);
    await player.play();
    await ended;
    return { frames: await collector.finish(), method: 'playback' };
  } finally {
    if (callback !== undefined) {
      if (player.cancelVideoFrameCallback) player.cancelVideoFrameCallback(callback);
      else cancelAnimationFrame(callback);
    }
    player.pause();
    player.removeAttribute('src');
    player.load();
    player.remove();
    URL.revokeObjectURL(url);
  }
}

export async function extractKeyframes(blob, options) {
  if (typeof VideoDecoder !== 'undefined' && typeof OffscreenCanvas !== 'undefined') {
    try {
      const result = await decodeKeyframes(blob, options);
      if (result.frames.length >= 3) return result;
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      console.warn('WebCodecs decoding failed; falling back to playback', error);
    }
  }
  return playbackKeyframes(blob, options);
}
