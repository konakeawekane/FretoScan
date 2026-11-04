import { grayscale, detect, match } from './features.js';
import { estimatePose, meshFaces, initialPose, advancePose, mul3, transpose3 } from './reconstruction.js';
import { refinePose, worldPoints } from './tracking.js';
import { estimateDenseDepth } from './depth.js';
import { TsdfVolume } from './volume.js';

const id = ({ x, y }) => `${x}:${y}`;
const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];

export class Scanner {
  constructor(fov = 58) {
    this.fov = fov;
    this.pose = initialPose();
    this.previous = null;
    this.keyframes = [];
    this.volume = new TsdfVolume();
    this.misses = 0;
    this.gyroTravel = 0;
    this.lastMotionTime = null;
    this.fused = false;
    this.keyframeCount = 0;
    this.flow = [0, 0];
    this.seed = 7;
    this.random = () => ((this.seed = (this.seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  }

  motion(samples = []) {
    for (const sample of samples) {
      if (this.lastMotionTime !== null && sample.rotation?.every(Number.isFinite)) {
        const interval = Math.min(100, Math.max(0, sample.tMs - this.lastMotionTime));
        this.gyroTravel += Math.hypot(...sample.rotation) * Math.PI / 180 * interval / 1000;
      }
      this.lastMotionTime = sample.tMs;
    }
  }

  process({ width, height, rgba, motion = [] }) {
    this.motion(motion);
    const gray = grayscale(rgba, width, height);
    const current = { width, height, rgba, gray, features: detect(gray, width, height), landmarks: new Map() };
    const focal = Math.max(width, height) / (2 * Math.tan(this.fov * Math.PI / 360));
    if (!this.previous || this.previous.width !== width || this.previous.height !== height) {
      current.pose = this.pose;
      this.previous = current;
      this.keyframes.push(current);
      this.keyframeCount++;
      return { type: 'result', message: 'Move sideways slowly to reveal depth' };
    }
    const reference = this.previous;
    let pairs = match(reference.gray, gray, reference.features, width, height, this.flow);
    if (pairs.length < 25 && (this.flow[0] || this.flow[1])) pairs = match(reference.gray, gray, reference.features, width, height);
    const metrics = { matches: pairs.length, landmarks: reference.landmarks.size, voxels: this.volume.cells.size, keyframes: this.keyframeCount };
    const failed = (message) => {
      if (++this.misses > 2) this.flow = [0, 0];
      return { type: 'result', message, metrics };
    };
    if (pairs.length < 25) return failed('Point at a detailed, well-lit surface');
    const flows = pairs.map((pair) => Math.hypot(pair.x2 - pair.x1, pair.y2 - pair.y1)).sort((a, b) => a - b);
    if (flows[Math.floor(flows.length / 2)] < 5) return { type: 'result', message: 'Move sideways slowly to reveal depth', metrics };
    const landmarkPairs = pairs.flatMap((pair) => {
      const landmark = reference.landmarks.get(`${pair.x1}:${pair.y1}`);
      return landmark ? [{ pair, point: landmark.world, x: pair.x2, y: pair.y2 }] : [];
    });
    const observations = landmarkPairs;
    const accept = (pose) => pose && pose.errorPx < 3.2 && pose.inliers >= Math.max(10, observations.length * 0.5)
      && Math.hypot(...pose.C.map((value, i) => value - reference.pose.C[i])) < 0.8;
    let predicted = reference.pose;
    if (landmarkPairs.length >= 8) {
      const motion = [0, 1].map((axis) => median(landmarkPairs.map(({ pair, point }) => {
        const depth = mul3(reference.pose.R, point.map((value, i) => value - reference.pose.C[i]))[2];
        return (axis ? pair.y2 - pair.y1 : pair.x1 - pair.x2) * depth / focal;
      })));
      const step = mul3(transpose3(reference.pose.R), [motion[0], motion[1], 0]);
      predicted = { R: reference.pose.R, C: reference.pose.C.map((value, i) => value + step[i]) };
    }
    let refined = refinePose(observations, predicted, focal, width, height);
    let anchored = Boolean(accept(refined));
    if (!anchored && predicted !== reference.pose) {
      const stationary = refinePose(observations, reference.pose, focal, width, height);
      if (accept(stationary)) { refined = stationary; anchored = true; }
    }
    let estimate;
    if (!anchored) {
      estimate = estimatePose(pairs, width, height, this.random, focal);
      if (!estimate) return failed('Tracking uncertain; move slowly and keep surfaces in view');
      const visualAngle = Math.acos(Math.max(-1, Math.min(1, (estimate.R[0] + estimate.R[4] + estimate.R[8] - 1) / 2)));
      if (this.gyroTravel > 0.02 && visualAngle > this.gyroTravel + 0.35) return failed('Camera motion inconsistent; move more slowly');
      const initial = advancePose(reference.pose, estimate);
      refined = refinePose(observations, initial, focal, width, height);
      anchored = Boolean(accept(refined));
      current.pose = anchored ? refined : initial;
    } else current.pose = refined;
    const geometryMatches = anchored ? pairs.map((pair) => ({
      x: pair.x1, y: pair.y1, x2: pair.x2, y2: pair.y2,
      a: [(pair.x1 - width / 2) / focal, (height / 2 - pair.y1) / focal, 1],
      b: [(pair.x2 - width / 2) / focal, (height / 2 - pair.y2) / focal, 1],
    })) : estimate.points;
    const points = worldPoints(geometryMatches, reference.pose, current.pose, estimate?.points || []);
    const byPoint = new Map(points.map((point) => [id(point), point]));
    for (const match of geometryMatches) {
      const oldWorld = reference.landmarks.get(id(match))?.world;
      const newWorld = byPoint.get(id(match))?.world;
      let world = oldWorld || newWorld;
      if (!world) continue;
      if (anchored && oldWorld && newWorld && Math.hypot(...oldWorld.map((value, i) => value - newWorld[i])) < 0.08) {
        world = oldWorld.map((value, i) => value * 0.75 + newWorld[i] * 0.25);
      }
      const cameraPoint = mul3(current.pose.R, world.map((value, i) => value - current.pose.C[i]));
      if (cameraPoint[2] <= 0 || Math.hypot(width / 2 + focal * cameraPoint[0] / cameraPoint[2] - match.x2,
        height / 2 - focal * cameraPoint[1] / cameraPoint[2] - match.y2) > 5) continue;
      let nearest = null, distance = 7 * 7;
      for (const feature of current.features) {
        const squared = (feature.x - match.x2) ** 2 + (feature.y - match.y2) ** 2;
        if (squared < distance) { nearest = feature; distance = squared; }
      }
      if (nearest) current.landmarks.set(id(nearest), { ...nearest, world });
    }
    this.pose = current.pose;
    this.previous = current;
    this.misses = 0;
    this.gyroTravel = 0;
    for (let axis = 0; axis < 2; axis++) this.flow[axis] = median(pairs.map((pair) => axis ? pair.y2 - pair.y1 : pair.x2 - pair.x1));
    const older = this.keyframes.slice(-3, -1).reverse();
    this.keyframes.push(current);
    if (this.keyframes.length > 6) this.keyframes.shift();
    metrics.keyframes = ++this.keyframeCount;
    metrics.inliers = anchored ? refined.inliers : estimate.inliers;
    metrics.landmarks = current.landmarks.size;
    metrics.tracking = anchored ? 'anchored' : 'relative';
    const tracks = [...current.landmarks.values()].flatMap((point) => [point.x, point.y]);

    if (reference.landmarks.size >= 8) {
      const sparse = [...reference.landmarks.values()];
      const neighbors = [current, ...older];
      const depth = estimateDenseDepth(reference, neighbors, sparse, focal);
      const valid = depth.confidence.reduce((total, value) => total + (value > 0 ? 1 : 0), 0);
      metrics.depthPixels = valid;
      if (valid > 150) {
        this.volume.integrate(depth, reference, focal);
        metrics.voxels = this.volume.cells.size;
        const mesh = this.volume.mesh();
        if (mesh.positions.length >= 90) {
          this.fused = true;
          return { type: 'volume', ...mesh, pose: current.pose, focal, width, height, metrics, tracks };
        }
      }
    }
    if (this.fused) return { type: 'pose', pose: current.pose, focal, width, height, metrics, tracks, message: 'Building a continuous surface' };

    const faces = meshFaces(points, width, height);
    const positions = new Float32Array(faces.length * 9);
    const colors = new Float32Array(faces.length * 9);
    let offset = 0;
    for (const face of faces) for (const index of face) {
      const point = points[index];
      positions.set([point.world[0], point.world[1], -point.world[2]], offset);
      const pixel = (Math.round(point.y) * width + Math.round(point.x)) * 4;
      colors.set([reference.rgba[pixel] / 255, reference.rgba[pixel + 1] / 255, reference.rgba[pixel + 2] / 255], offset);
      offset += 3;
    }
    return { type: faces.length ? 'patch' : 'pose', positions, colors, pose: current.pose, focal, width, height, metrics, tracks, message: 'Searching for connected surfaces' };
  }
}
