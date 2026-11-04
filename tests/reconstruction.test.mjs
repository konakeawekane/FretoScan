import test from 'node:test';
import assert from 'node:assert/strict';
import { estimatePose, meshFaces, mul3, advancePose, initialPose, toWorld } from '../src/scan/reconstruction.js';
import { detect, grayscale, match } from '../src/scan/features.js';
import { refinePose } from '../src/scan/tracking.js';
import { estimateDenseDepth } from '../src/scan/depth.js';
import { TsdfVolume } from '../src/scan/volume.js';

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

test('builds mesh faces from two synthetic camera images', async () => {
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
  assert.ok(messages.some((message) => message.type === 'patch' && message.positions.length >= 9));
  self.onmessage({ data: { type: 'frame', sessionId: 7, width, height, rgba: third } });
  assert.ok(messages.at(-1).metrics?.inliers > 10);
  assert.ok(messages.at(-1).metrics?.landmarks > 5);
  assert.ok(messages.at(-1).metrics?.depthPixels > 150);
  assert.equal(messages.at(-1).metrics.tracking, 'anchored');
  assert.equal(messages.at(-1).type, 'volume');
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
