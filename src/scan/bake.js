import { grayscale } from './features.js';
import { estimateDenseDepth } from './depth.js';
import { TsdfVolume } from './volume.js';
import { alignPrior } from './mono.js';
import { cleanMesh, expand, vertexNormals } from './mesh.js';
import { mul3, transpose3 } from './reconstruction.js';

export const QUALITY = {
  quarter: { size: 160, solveSize: 640, maxFrames: 40, views: 2, voxel: 1 / 40, maxCells: 160000, faces: 20000, rate: 3 },
  half: { size: 320, solveSize: 800, maxFrames: 70, views: 3, voxel: 1 / 70, maxCells: 320000, faces: 50000, rate: 2 },
  full: { size: 640, solveSize: 1024, maxFrames: 110, views: 4, voxel: 1 / 110, maxCells: 640000, faces: 120000, rate: 1 },
};

const RADIUS = 2;
const median = (values) => values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : 0;
const cameraPoint = (pose, world) => mul3(pose.R, world.map((value, i) => value - pose.C[i]));

function project(view, world) {
  const [x, y, z] = cameraPoint(view.pose, world);
  if (z <= 1e-6) return null;
  return { x: view.width / 2 + view.focal * x / z, y: view.height / 2 - view.focal * y / z, z };
}

function sampleDepth(view, x, y) {
  const { depthMap } = view;
  const ox = Math.round(x / depthMap.step), oy = Math.round(y / depthMap.step);
  if (ox < 0 || oy < 0 || ox >= depthMap.width || oy >= depthMap.height) return -1;
  const index = oy * depthMap.width + ox;
  return depthMap.confidence[index] >= 0.15 ? depthMap.depths[index] : 0;
}

export class Baker {
  constructor(level) {
    this.settings = QUALITY[level];
    if (!this.settings) throw new Error('Unknown quality level');
    this.frames = [];
    this.views = [];
    this.cursor = 0;
    this.received = 0;
    this.fused = 0;
    this.depthPixels = 0;
    this.filledPixels = 0;
    this.aligned = 0;
  }

  prepare(frame) {
    frame.gray = grayscale(frame.rgba, frame.width, frame.height);
    const depths = frame.landmarks.map(({ world }) => cameraPoint(frame.pose, world)[2]).filter((depth) => depth > 0);
    frame.depthScale = median(depths) || 1;
    if (frame.prior) {
      const fit = alignPrior(frame.prior, frame.width, frame.height,
        frame.landmarks.map(({ x, y, world }) => ({ x, y, depth: cameraPoint(frame.pose, world)[2] })));
      frame.priorDepth = fit?.depth || null;
      if (fit) this.aligned++;
    }
  }

  process(reference, neighbors) {
    const scale = reference.depthScale;
    const ids = new Set(reference.landmarks.map(({ id }) => id));
    const shared = (view) => view.landmarks.reduce((count, { id }) => count + ids.has(id), 0);
    const views = neighbors.filter((view) => {
      const distance = Math.hypot(...view.pose.C.map((value, i) => value - reference.pose.C[i]));
      return distance > scale * 0.015 && distance < scale * 0.8;
    }).sort((a, b) => shared(b) - shared(a)).slice(0, this.settings.views);
    let depth = estimateDenseDepth(reference, views, reference.landmarks, reference.focal, 2, reference.priorDepth);
    if (reference.prior) {
      const anchors = reference.landmarks.map(({ x, y, world }) => ({ x, y, depth: cameraPoint(reference.pose, world)[2], weight: 2 }));
      for (let i = 0; i < depth.depths.length; i += 7) if (depth.confidence[i] >= 0.4) {
        anchors.push({ x: (i % depth.width) * depth.step, y: Math.floor(i / depth.width) * depth.step, depth: depth.depths[i] });
      }
      const fit = alignPrior(reference.prior, reference.width, reference.height, anchors) || (reference.priorDepth && { depth: reference.priorDepth });
      if (fit) {
        for (let oy = 0; oy < depth.height; oy++) for (let ox = 0; ox < depth.width; ox++) {
          const index = oy * depth.width + ox;
          if (depth.depths[index]) continue;
          const value = fit.depth[oy * depth.step * reference.width + ox * depth.step];
          if (!(value > 0)) continue;
          depth.depths[index] = value;
          depth.confidence[index] = 0.3;
          this.filledPixels++;
        }
      }
    }
    const valid = depth.confidence.reduce((count, value) => count + (value >= 0.15), 0);
    if (valid < 40) return;
    this.depthPixels += valid;
    const rgb = new Uint8Array(reference.width * reference.height * 3);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { rgb[i] = reference.rgba[j]; rgb[i + 1] = reference.rgba[j + 1]; rgb[i + 2] = reference.rgba[j + 2]; }
    depth.colors = rgb;
    this.views.push({ pose: reference.pose, focal: reference.focal, width: reference.width, height: reference.height,
      depthMap: depth, rgb, depthScale: scale, sharpness: reference.sharpness || 1 });
    this.fused++;
  }

  advance(final = false) {
    while (this.cursor < this.frames.length && (final || this.cursor + RADIUS < this.frames.length)) {
      const index = this.cursor++;
      const reference = this.frames[index];
      if (reference.landmarks.length < 8) continue;
      const neighbors = this.frames.slice(Math.max(0, index - RADIUS), index + RADIUS + 1).filter((frame) => frame !== reference && frame.gray);
      this.process(reference, neighbors);
    }
    for (let i = 0; i < this.cursor - RADIUS; i++) {
      const frame = this.frames[i];
      frame.gray = frame.rgba = frame.prior = frame.priorDepth = null;
    }
  }

  addFrame(frame) {
    this.prepare(frame);
    this.frames.push(frame);
    this.received++;
    this.advance();
    return this.metrics();
  }

  metrics() {
    return { received: this.received, fused: this.fused, depthPixels: this.depthPixels, filledPixels: this.filledPixels,
      aligned: this.aligned, voxels: this.volume?.size || 0, saturated: this.volume ? this.volume.size >= this.volume.maxCells : false };
  }

  checkConsistency(tolerance = 0.05) {
    for (const view of this.views) {
      const others = this.views.filter((other) => other !== view)
        .map((other) => ({ other, distance: Math.hypot(...other.pose.C.map((value, i) => value - view.pose.C[i])) }))
        .sort((a, b) => a.distance - b.distance).slice(0, 4).map(({ other }) => other);
      if (!others.length) continue;
      const { depthMap } = view;
      const inverse = transpose3(view.pose.R);
      for (let oy = 0; oy < depthMap.height; oy++) for (let ox = 0; ox < depthMap.width; ox++) {
        const index = oy * depthMap.width + ox;
        const depth = depthMap.depths[index];
        if (!depth || depthMap.confidence[index] < 0.15) continue;
        const x = ox * depthMap.step, y = oy * depthMap.step;
        const local = [(x - view.width / 2) / view.focal * depth, (view.height / 2 - y) / view.focal * depth, depth];
        const world = mul3(inverse, local).map((value, i) => value + view.pose.C[i]);
        let agree = 0, conflict = 0, seen = 0;
        for (const other of others) {
          const point = project(other, world);
          if (!point) continue;
          const observed = sampleDepth(other, point.x, point.y);
          if (observed < 0) continue;
          seen++;
          if (!observed) continue;
          if (Math.abs(observed - point.z) < point.z * tolerance) agree++;
          else if (point.z < observed * (1 - tolerance * 2)) conflict++;
        }
        const confidence = depthMap.confidence[index];
        if (conflict >= 2 && conflict > agree) depthMap.confidence[index] = 0;
        else if (agree) depthMap.confidence[index] = Math.min(1, confidence * (1 + 0.25 * agree) + (confidence < 0.35 ? 0.15 : 0));
        else if (seen >= 2) depthMap.confidence[index] = confidence * (confidence < 0.35 ? 0.5 : 0.8);
      }
    }
  }

  colorize(mesh) {
    const normals = vertexNormals(mesh);
    const best = Math.max(...this.views.map((view) => view.sharpness));
    for (let v = 0; v < mesh.positions.length / 3; v++) {
      const world = [mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]];
      let score = 0, color = null;
      for (const view of this.views) {
        const point = project(view, world);
        if (!point || point.x < 0 || point.y < 0 || point.x > view.width - 1 || point.y > view.height - 1) continue;
        const observed = sampleDepth(view, point.x, point.y);
        if (observed <= 0 || Math.abs(observed - point.z) > this.volume.truncation) continue;
        const toCamera = view.pose.C.map((value, i) => value - world[i]);
        const length = Math.hypot(...toCamera) || 1;
        const facing = Math.abs(toCamera.reduce((sum, value, i) => sum + value * normals[v * 3 + i], 0)) / length;
        const candidate = facing * Math.sqrt(view.sharpness / best) / point.z;
        if (candidate <= score) continue;
        score = candidate;
        const pixel = (Math.round(point.y) * view.width + Math.round(point.x)) * 3;
        color = [view.rgb[pixel] / 255, view.rgb[pixel + 1] / 255, view.rgb[pixel + 2] / 255];
      }
      if (color) mesh.colors.set(color, v * 3);
    }
    return mesh;
  }

  finish() {
    this.advance(true);
    if (!this.views.length) return { positions: new Float32Array(), colors: new Float32Array(), metrics: this.metrics() };
    const depth = median(this.views.map((view) => view.depthScale));
    const footprint = median(this.views.map((view) => view.depthMap.step / view.focal));
    const voxel = depth * Math.max(this.settings.voxel, footprint * 1.4);
    this.volume = new TsdfVolume(voxel, voxel * 3, this.settings.maxCells);
    this.checkConsistency();
    for (const view of this.views) this.volume.integrate(view.depthMap, view, view.focal);
    const mesh = this.colorize(cleanMesh(this.volume.extract(), { limit: this.settings.faces, voxel }));
    for (let i = 0; i < mesh.colors.length; i++) {
      const value = mesh.colors[i];
      mesh.colors[i] = value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    }
    return { ...expand(mesh), metrics: this.metrics() };
  }
}
