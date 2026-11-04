import test from 'node:test';
import assert from 'node:assert/strict';
import { estimatePose, meshFaces, mul3, advancePose, initialPose, toWorld } from '../src/scan/reconstruction.js';
import { detect, grayscale, match } from '../src/scan/features.js';
import { refinePose } from '../src/scan/tracking.js';
import { estimateDenseDepth } from '../src/scan/depth.js';
import { TsdfVolume } from '../src/scan/volume.js';
import { Scanner } from '../src/scan/pipeline.js';
import { Baker } from '../src/scan/bake.js';
import { parseWebm } from '../src/scan/webm.js';
import { sharpness, KeyframePicker } from '../src/scan/sharpness.js';
import { alignPrior } from '../src/scan/mono.js';
import { colmapToScene, trackingScene } from '../src/scan/sfm.js';
import { removeSmallComponents, simplify } from '../src/scan/mesh.js';

test('recovers a camera step and triangulates points', () => {
  const width = 320, height = 240, focal = 288;
  const angle = 0.07;
  const R = [Math.cos(angle), 0, Math.sin(angle), 0, 1, 0, -Math.sin(angle), 0, Math.cos(angle)];
  const t = [-0.32, 0.02, 0.01];
  const project = ([x, y, z]) => [width / 2 + focal * x / z, height / 2 - focal * y / z];
  let seed = 4;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const pairs = [];
  for (let i = 0; i < 100; i++) {
    const point = [(random() - 0.5) * 2, (random() - 0.5) * 1.5, 2 + random() * 2];
    const [x1, y1] = project(point);
    const [x2, y2] = project(mul3(R, point).map((x, j) => x + t[j]));
    pairs.push({ x1, y1, x2, y2 });
  }
  for (let i = 0; i < 15; i++) pairs.push({ x1: random() * width, y1: random() * height, x2: random() * width, y2: random() * height });
  const result = estimatePose(pairs, width, height, random);
  assert.ok(result);
  assert.ok(result.points.length > 70);
  assert.ok(result.t[0] < -0.8);
  assert.ok(Math.abs(result.R[0] - R[0]) < 0.1);
  const next = advancePose(initialPose(), result);
  assert.ok(next.C[0] > 0);
  assert.equal(toWorld([0, 0, 5], initialPose())[2], 1);
});

test('triangulates a connected surface', () => {
  const points = [];
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) points.push({ x: 30 + x * 20, y: 30 + y * 20, point: [x, y, 5] });
  assert.ok(meshFaces(points, 160, 160).length > 10);
});

test('tracks camera image features after translation', () => {
  const width = 160, height = 120;
  let seed = 9;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const previous = Uint8Array.from({ length: width * height }, () => Math.floor(random() * 256));
  const current = new Uint8Array(previous.length);
  for (let y = 0; y < height; y++) for (let x = 8; x < width; x++) current[y * width + x] = previous[y * width + x - 8];
  const pairs = match(previous, current, detect(previous, width, height), width, height);
  assert.ok(pairs.length > 20);
  assert.ok(pairs.filter((pair) => pair.x2 - pair.x1 === 8 && pair.y2 === pair.y1).length > pairs.length * 0.8);
});

test('keeps feature tracks through exposure change and larger motion', () => {
  const width = 160, height = 120;
  let seed = 21;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const previous = Uint8Array.from({ length: width * height }, () => Math.floor(random() * 256));
  const current = new Uint8Array(previous.length);
  const dx = 38, dy = 9;
  for (let y = dy; y < height; y++) for (let x = dx; x < width; x++) {
    current[y * width + x] = Math.round(previous[(y - dy) * width + x - dx] * 0.75 + 25);
  }
  const pairs = match(previous, current, detect(previous, width, height), width, height);
  assert.ok(pairs.length > 10);
  assert.ok(pairs.filter((pair) => pair.x2 - pair.x1 === dx && pair.y2 - pair.y1 === dy).length > pairs.length * 0.8);
});

test('keeps image tracking through low motion and reacquires after feature loss', () => {
  const width = 160, height = 120;
  let seed = 27;
  const first = new Uint8Array(width * height * 4);
  const shifted = new Uint8Array(first.length);
  for (let i = 0; i < width * height; i++) {
    const value = ((seed = (seed * 1664525 + 1013904223) >>> 0) >>> 24);
    first.set([value, value, value, 255], i * 4);
  }
  for (let y = 0; y < height; y++) for (let x = 8; x < width; x++) {
    shifted.set(first.subarray((y * width + x - 8) * 4, (y * width + x - 8) * 4 + 4), (y * width + x) * 4);
  }
  const scanner = new Scanner();
  scanner.process({ width, height, rgba: first });
  const still = scanner.process({ width, height, rgba: first });
  assert.ok(still.metrics.matches > 20);
  assert.equal(still.metrics.tracking, 'image');
  scanner.process({ width, height, rgba: new Uint8Array(first.length) });
  scanner.process({ width, height, rgba: first });
  const recovered = scanner.process({ width, height, rgba: shifted });
  assert.ok(recovered.metrics.matches > 10);
  assert.equal(recovered.type, 'tracking');
  assert.equal(recovered.positions, undefined);
});

test('keeps tracking through synthetic camera frames without generating geometry', async () => {
  const width = 320, height = 240, focal = 288;
  const first = new Uint8Array(width * height * 4);
  const second = new Uint8Array(first.length);
  const third = new Uint8Array(first.length);
  let seed = 14;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const used = [];
  const project = ([x, y, z]) => [Math.round(width / 2 + focal * x / z), Math.round(height / 2 - focal * y / z)];
  for (let i = 0; i < 160; i++) {
    const point = [(random() - 0.5) * 2.4, (random() - 0.5) * 1.6, 2.4 + random() * 1.6];
    const a = project(point), b = project([point[0] - 0.28, point[1], point[2]]), c = project([point[0] - 0.56, point[1], point[2]]);
    if (a.some((v, j) => v < 10 || v > [width - 10, height - 10][j]) || b[0] < 10 || used.some((other) => Math.hypot(a[0] - other[0], a[1] - other[1]) < 15)) continue;
    used.push(a);
    const pattern = Array.from({ length: 81 }, () => Math.floor(random() * 200) + 40);
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
      const color = pattern[(dy + 4) * 9 + dx + 4];
      for (const [image, center] of [[first, a], [second, b], [third, c]]) {
        const index = ((center[1] + dy) * width + center[0] + dx) * 4;
        image[index] = image[index + 1] = image[index + 2] = color;
        image[index + 3] = 255;
      }
    }
  }
  const gray1 = grayscale(first, width, height), gray2 = grayscale(second, width, height);
  const pairs = match(gray1, gray2, detect(gray1, width, height), width, height);
  const pose = estimatePose(pairs, width, height, random);
  assert.ok(pose);
  assert.ok(meshFaces(pose.points, width, height).length > 0);
  const messages = [];
  globalThis.self = { postMessage: (message) => messages.push(message) };
  await import('../src/scan/worker.js');
  self.onmessage({ data: { type: 'reset', sessionId: 7 } });
  self.onmessage({ data: { type: 'frame', sessionId: 7, width, height, rgba: first } });
  self.onmessage({ data: { type: 'frame', sessionId: 7, width, height, rgba: second } });
  assert.equal(messages.at(-1).type, 'tracking');
  assert.ok(messages.at(-1).metrics.matches > 25);
  assert.equal(messages.at(-1).metrics.tracking, 'relative');
  const ids = new Set(messages.at(-1).trackIds);
  self.onmessage({ data: { type: 'frame', sessionId: 7, tMs: 300, mediaTime: 0.3, width, height, rgba: third } });
  assert.equal(messages.at(-1).type, 'tracking');
  assert.ok(messages.at(-1).metrics.matches > 25);
  assert.ok(messages.at(-1).tracks.length > 50);
  assert.ok(messages.at(-1).landmarks.length > 8);
  assert.equal(messages.at(-1).mediaTime, 0.3);
  assert.ok(messages.at(-1).trackIds.filter((id) => ids.has(id)).length > 20);
  assert.equal(messages.at(-1).positions, undefined);
  self.onmessage({ data: { type: 'frame', sessionId: 7, width, height, rgba: third } });
  assert.ok(messages.at(-1).metrics.matches > 25);
  assert.equal(messages.at(-1).metrics.tracking, 'anchored');
});

test('refines camera position from persistent landmarks', () => {
  const width = 320, height = 240, focal = 288;
  const angle = 0.08;
  const R = [Math.cos(angle), 0, Math.sin(angle), 0, 1, 0, -Math.sin(angle), 0, Math.cos(angle)];
  const C = [0.24, -0.06, 0.03];
  const observations = [];
  for (let i = 0; i < 80; i++) {
    const point = [((i * 37) % 81 - 40) / 36, ((i * 19) % 61 - 30) / 40, 2 + (i % 9) * 0.25];
    const [x, y, z] = mul3(R, point.map((value, j) => value - C[j]));
    observations.push({ point, x: width / 2 + focal * x / z, y: height / 2 - focal * y / z });
  }
  const estimate = refinePose(observations, { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], C: [0.18, -0.03, 0.01] }, focal, width, height);
  assert.ok(estimate);
  assert.ok(estimate.errorPx < 0.1);
  assert.ok(Math.hypot(...estimate.C.map((value, i) => value - C[i])) < 0.01);
});

test('estimates dense depth from a translated image pair', () => {
  const width = 96, height = 72, focal = 86.4;
  const first = new Uint8Array(width * height), second = new Uint8Array(first.length);
  let seed = 18;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  for (let i = 0; i < first.length; i++) first[i] = Math.floor(random() * 255);
  const shift = 8;
  for (let y = 0; y < height; y++) for (let x = 0; x < width - shift; x++) second[y * width + x] = Math.round(first[y * width + x + shift] * 0.75 + 25);
  const reference = { width, height, gray: first, pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], C: [0, 0, 0] } };
  const neighbor = { width, height, gray: second, pose: { R: reference.pose.R, C: [shift * 2 / focal, 0, 0] } };
  const landmarks = [];
  for (let y = 12; y < height - 12; y += 12) for (let x = 12; x < width - 12; x += 12) landmarks.push({ x, y, world: [(x - width / 2) * 2 / focal, (height / 2 - y) * 2 / focal, 2] });
  const result = estimateDenseDepth(reference, [neighbor], landmarks, focal);
  const valid = [...result.depths].filter(Boolean);
  valid.sort((a, b) => a - b);
  assert.ok(valid.length > 100);
  assert.ok(Math.abs(valid[Math.floor(valid.length / 2)] - 2) < 0.2);
});

test('fuses depth into a continuous surface mesh', () => {
  const width = 96, height = 72, step = 2, focal = 86.4;
  const depths = new Float32Array(width / step * height / step).fill(2);
  const confidence = new Float32Array(depths.length).fill(1);
  const rgba = new Uint8Array(width * height * 4).fill(160);
  const volume = new TsdfVolume(0.08, 0.24);
  const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const depthMap = { width: width / step, height: height / step, step, depths, confidence };
  const first = { width, height, rgba, pose: { R: identity, C: [0, 0, 0] } };
  const second = { width, height, rgba, pose: { R: identity, C: [0.1, 0, 0] } };
  volume.integrate(depthMap, first, focal);
  volume.integrate(depthMap, second, focal);
  const mesh = volume.mesh();
  assert.ok(mesh.positions.length > 300);
  const z = [];
  for (let i = 2; i < mesh.positions.length; i += 3) z.push(mesh.positions[i]);
  z.sort((a, b) => a - b);
  assert.ok(Math.abs(z[Math.floor(z.length / 2)] + 2) < 0.2);
});

test('matches keyframes to the nearest live tracking pose', () => {
  const tracking = Array.from({ length: 30 }, (_, i) => ({
    tMs: 500 + i * 100, focal: 90, width: 96, height: 72,
    pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], C: [i * 0.03, 0, 0] },
    landmarks: Array(8).fill({ id: 1, x: 20, y: 10, world: [0, 0, 2] }),
  }));
  const frames = [{ timeS: 0.2, width: 192, height: 144 }, { timeS: 9, width: 192, height: 144 }];
  const scene = trackingScene(frames, { tracking, recordStartMs: 400 });
  assert.equal(scene.registered, 1);
  assert.ok(Math.abs(scene.views[0].C[0] - 0.03) < 1e-9);
  assert.equal(scene.views[0].focal, 180);
  assert.deepEqual([scene.views[0].landmarks[0].x, scene.views[0].landmarks[0].y], [40, 20]);
  assert.equal(scene.views[1], null);
});

test('parses video blocks from a WebM recording', () => {
  const element = (id, payload) => {
    const idBytes = [];
    for (let value = id; value; value = Math.floor(value / 256)) idBytes.unshift(value & 255);
    return [...idBytes, 0x80 | payload.length, ...payload];
  };
  const unknown = (id, children) => [...element(id, []).slice(0, -1), 0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, ...children];
  const block = (time, key, data) => element(0xa3, [0x81, time >> 8, time & 255, key ? 0x80 : 0, ...data]);
  const bytes = Uint8Array.from([
    ...element(0x1a45dfa3, [0x42, 0x86, 0x81, 0x01]),
    ...unknown(0x18538067, [
      ...element(0x1549a966, element(0x2ad7b1, [0x0f, 0x42, 0x40])),
      ...element(0x1654ae6b, element(0xae, [...element(0xd7, [1]), ...element(0x83, [1]), ...element(0x86, [...'V_VP8'].map((c) => c.charCodeAt(0))),
        ...element(0xe0, [...element(0xb0, [0x01, 0x40]), ...element(0xba, [0xf0])])])),
      ...unknown(0x1f43b675, [...element(0xe7, [0]), ...block(0, true, [1, 2, 3]), ...block(33, false, [4, 5])]),
      ...unknown(0x1f43b675, [...element(0xe7, [0x03, 0xe8]), ...block(-10 & 0xffff, false, [6])]),
    ]),
  ]);
  const track = parseWebm(bytes);
  assert.deepEqual([track.width, track.height, track.configs[0].codec], [320, 240, 'vp8']);
  assert.deepEqual(track.samples.map(({ timestamp, key }) => [timestamp, key]), [[0, true], [33000, false], [990000, false]]);
  assert.deepEqual([...track.samples[1].data], [4, 5]);
});

test('scores sharp frames above blurred ones and keeps one per time slot', () => {
  const width = 64, height = 48;
  let seed = 5;
  const sharp = Uint8Array.from({ length: width * height }, () => ((seed = (seed * 1664525 + 1013904223) >>> 0) >>> 24));
  const blurred = new Uint8Array(sharp.length);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    let sum = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) sum += sharp[(y + dy) * width + x + dx];
    blurred[y * width + x] = sum / 9;
  }
  assert.ok(sharpness(sharp, width, height) > sharpness(blurred, width, height) * 4);
  const picker = new KeyframePicker(2, 4);
  assert.deepEqual(picker.offer(0.1, 5), { flush: false, keep: true });
  assert.deepEqual(picker.offer(0.2, 3), { flush: false, keep: false });
  assert.deepEqual(picker.offer(0.3, 9), { flush: false, keep: true });
  assert.deepEqual(picker.offer(0.6, 1), { flush: true, keep: true });
});

test('aligns relative neural depth to sparse metric points', () => {
  const width = 40, height = 30;
  const truth = (x, y) => 1.5 + x * 0.03 + y * 0.01;
  const prior = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) prior[y * width + x] = (1 / truth(x, y) - 0.1) / 2.5;
  const anchors = [];
  for (let i = 0; i < 30; i++) anchors.push({ x: (i * 7) % width, y: (i * 11) % height, depth: truth((i * 7) % width, (i * 11) % height) });
  for (let i = 0; i < 6; i++) anchors.push({ x: i * 5, y: 3, depth: 9 });
  const fit = alignPrior(prior, width, height, anchors);
  assert.ok(fit);
  assert.ok(Math.abs(fit.scale - 2.5) < 0.01 && Math.abs(fit.shift - 0.1) < 0.01);
  assert.ok(Math.abs(fit.depth[20 * width + 33] - truth(33, 20)) < 0.01);
  assert.ok(fit.inliers >= 30);
});

test('converts COLMAP cameras into the scan convention', () => {
  const width = 200, height = 100, focal = 150;
  const cameras = [[0, 0, 0], [0.4, 0, 0]];
  const world = [[0.2, 0.3, 4], [-0.5, -0.2, 3], [0.6, 0.1, 5]];
  const poses = new Float64Array(26), keypoints = [[], []], tracks = [], points = [];
  cameras.forEach((C, i) => {
    poses.set([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, -C[0], -C[1], -C[2]], i * 13);
  });
  world.forEach((X, j) => {
    points.push(...X, 0.5);
    tracks.push(2);
    cameras.forEach((C, i) => {
      const [x, y, z] = X.map((value, k) => value - C[k]);
      keypoints[i].push(focal * x / z + width / 2 + 0.5, focal * y / z + height / 2 + 0.5, 0, 0, 0, 0);
      tracks.push(i, j);
    });
  });
  const scene = colmapToScene({ camera: [focal, width / 2, height / 2], poses, points: Float64Array.from(points), tracks: Int32Array.from(tracks) },
    keypoints.map((values) => Float32Array.from(values)), 2);
  assert.equal(scene.registered, 2);
  const second = scene.views[1];
  for (const { x, y, world: point } of second.landmarks) {
    const camera = mul3(second.R, point.map((value, k) => value - second.C[k]));
    assert.ok(Math.abs(width / 2 + focal * camera[0] / camera[2] - x) < 1e-3);
    assert.ok(Math.abs(height / 2 - focal * camera[1] / camera[2] - y) < 1e-3);
  }
  const depths = scene.views[0].landmarks.map(({ world: point }) => point[2]).sort((a, b) => a - b);
  assert.ok(Math.abs(depths[1] - 1) < 1e-6);
});

test('drops floating fragments and decimates large meshes', () => {
  const positions = [], colors = [], indices = [];
  const grid = (ox, n) => {
    const base = positions.length / 3;
    for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) { positions.push(ox + x * 0.01, y * 0.01, 1); colors.push(0.5, 0.5, 0.5); }
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const a = base + y * (n + 1) + x;
      indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
    }
  };
  grid(0, 40);
  grid(5, 2);
  const mesh = { positions: Float32Array.from(positions), colors: Float32Array.from(colors), indices: Uint32Array.from(indices) };
  const cleaned = removeSmallComponents(mesh);
  assert.equal(cleaned.indices.length / 3, 40 * 40 * 2);
  const reduced = simplify(cleaned, 800, 0.015);
  assert.ok(reduced.indices.length / 3 <= 800 && reduced.indices.length / 3 > 100);
});

test('bakes recorded views into a mesh after capture', () => {
  const width = 96, height = 72, focal = 86.4, shift = 8;
  let seed = 39;
  const first = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const value = ((seed = (seed * 1664525 + 1013904223) >>> 0) >>> 24);
    first.set([value, value, value, 255], i * 4);
  }
  const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const baker = new Baker('half');
  for (let view = 0; view < 4; view++) {
    const rgba = new Uint8Array(first.length);
    for (let y = 0; y < height; y++) for (let x = 0; x < width - shift * view; x++) {
      const from = (y * width + x + shift * view) * 4;
      rgba.set(first.subarray(from, from + 4), (y * width + x) * 4);
    }
    const landmarks = [];
    for (let y = 12; y < height - 12; y += 12) for (let x = 24; x < width - 12; x += 12) {
      landmarks.push({ x: x - shift * view, y, world: [(x - width / 2) * 2 / focal, (height / 2 - y) * 2 / focal, 2] });
    }
    baker.addFrame({ width, height, rgba, focal, pose: { R: identity, C: [view * shift * 2 / focal, 0, 0] }, landmarks });
  }
  const result = baker.finish();
  assert.equal(result.metrics.received, 4);
  assert.ok(result.metrics.fused >= 3);
  assert.ok(result.positions.length > 300);
});
