import { grayscale, detect, match } from './features.js';
import { estimatePose, initialPose, advancePose, mul3, transpose3 } from './reconstruction.js';
import { refinePose, worldPoints } from './tracking.js';

const key = ({ x, y }) => `${x}:${y}`;
const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];

function consistent(pairs, random) {
  if (pairs.length < 12) return pairs;
  let best = [];
  for (let iteration = 0; iteration < 60; iteration++) {
    const first = pairs[Math.floor(random() * pairs.length)];
    const second = pairs[Math.floor(random() * pairs.length)];
    const dx = second.x1 - first.x1, dy = second.y1 - first.y1;
    const distance = dx * dx + dy * dy;
    if (distance < 400) continue;
    const ux = second.x2 - first.x2, uy = second.y2 - first.y2;
    const a = (ux * dx + uy * dy) / distance;
    const b = (uy * dx - ux * dy) / distance;
    if (Math.hypot(a, b) < 0.7 || Math.hypot(a, b) > 1.35) continue;
    const tx = first.x2 - a * first.x1 + b * first.y1;
    const ty = first.y2 - b * first.x1 - a * first.y1;
    const inliers = pairs.filter((pair) => Math.hypot(a * pair.x1 - b * pair.y1 + tx - pair.x2,
      b * pair.x1 + a * pair.y1 + ty - pair.y2) < 7);
    if (inliers.length > best.length) best = inliers;
  }
  return best.length >= Math.max(10, pairs.length * 0.45) ? best : pairs;
}

export class Scanner {
  constructor(fov = 58) {
    this.fov = fov;
    this.pose = initialPose();
    this.previous = null;
    this.keyframe = null;
    this.flow = [0, 0];
    this.nextId = 0;
    this.frame = 0;
    this.seed = 7;
    this.random = () => ((this.seed = (this.seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  }

  newTracks(features) {
    const count = Math.min(240, features.length);
    return Array.from({ length: count }, (_, i) => {
      const { x, y } = features[Math.floor(i * features.length / count)];
      return { x, y, id: ++this.nextId };
    });
  }

  process({ width, height, rgba }) {
    const gray = grayscale(rgba, width, height);
    const features = detect(gray, width, height);
    const focal = Math.max(width, height) / (2 * Math.tan(this.fov * Math.PI / 360));
    const current = { width, height, gray, pose: this.pose, tracks: [] };
    this.frame++;
    if (!this.previous || this.previous.width !== width || this.previous.height !== height) {
      current.tracks = this.newTracks(features);
      this.previous = this.keyframe = current;
      return { type: 'tracking', tracks: [], trackIds: [], landmarks: [], width, height, focal, metrics: { tracking: 'image', matches: 0, features: current.tracks.length, landmarks: 0 }, message: 'Finding visual features' };
    }

    const previous = this.previous;
    let pairs = match(previous.gray, gray, previous.tracks, width, height, this.flow);
    if (pairs.length < 12 && (this.flow[0] || this.flow[1])) pairs = match(previous.gray, gray, previous.tracks, width, height);
    pairs = consistent(pairs, this.random);
    const old = new Map(previous.tracks.map((track) => [key(track), track]));
    for (const pair of pairs) {
      const track = old.get(`${pair.x1}:${pair.y1}`);
      if (track) current.tracks.push({ ...track, x: pair.x2, y: pair.y2 });
    }
    const followed = current.tracks.length;
    const stride = Math.max(1, Math.ceil(features.length / Math.max(1, 240 - current.tracks.length)));
    for (let offset = 0; offset < stride && current.tracks.length < 240; offset++) {
      for (let i = offset; i < features.length && current.tracks.length < 240; i += stride) {
        const feature = features[i];
        if (current.tracks.every((track) => Math.hypot(track.x - feature.x, track.y - feature.y) >= 10)) {
          current.tracks.push({ ...feature, id: ++this.nextId });
        }
      }
    }
    this.previous = current;
    const metrics = { tracking: 'image', matches: followed, features: current.tracks.length, landmarks: 0, displacementPx: 0 };
    const visible = current.tracks.slice(0, followed);
    const result = (message, pose) => ({ type: 'tracking', tracks: visible.flatMap(({ x, y }) => [x, y]),
      trackIds: visible.map(({ id }) => id),
      landmarks: pose ? current.tracks.filter((track) => track.world).map(({ id, x, y, world }) => ({ id, x, y, world })) : [],
      width, height, focal, pose, metrics, message });
    if (followed < 10) {
      this.flow = [0, 0];
      this.keyframe = current;
      return result('Reacquiring features; point at a detailed area');
    }
    for (let axis = 0; axis < 2; axis++) this.flow[axis] = median(pairs.map((pair) => axis ? pair.y2 - pair.y1 : pair.x2 - pair.x1));

    const observations = current.tracks.slice(0, followed).flatMap((track) => track.world ? [{ point: track.world, x: track.x, y: track.y }] : []);
    let refined = refinePose(observations, previous.pose, focal, width, height);
    if (refined && refined.errorPx < 3.5 && refined.inliers >= Math.max(10, observations.length * 0.5)
      && Math.hypot(...refined.C.map((value, i) => value - previous.pose.C[i])) < 0.5) {
      current.pose = refined;
      metrics.tracking = 'anchored';
    }

    const reference = this.keyframe;
    const byId = new Map(current.tracks.map((track) => [track.id, track]));
    const shared = reference.tracks.flatMap((track) => {
      const now = byId.get(track.id);
      return now ? [{ x1: track.x, y1: track.y, x2: now.x, y2: now.y, id: track.id }] : [];
    });
    if (shared.length) metrics.displacementPx = median(shared.map((pair) => Math.hypot(pair.x2 - pair.x1, pair.y2 - pair.y1)));
    if (metrics.tracking === 'image' && shared.length >= 25 && metrics.displacementPx >= 8 && (this.frame % 3 === 0 || metrics.displacementPx >= 16)) {
      const estimate = estimatePose(shared, width, height, this.random, focal);
      if (estimate) {
        current.pose = advancePose(reference.pose, estimate);
        metrics.tracking = 'relative';
        const source = new Map(reference.tracks.map((track) => [key(track), track.id]));
        for (const point of estimate.points) {
          const track = byId.get(source.get(key(point)));
          if (track && !track.world) track.world = mul3(transpose3(reference.pose.R), point.point)
            .map((value, i) => reference.pose.C[i] + value * 0.2);
        }
      }
    }
    if (metrics.tracking === 'anchored' && shared.length >= 12 && metrics.displacementPx >= 8) {
      const candidates = shared.filter((pair) => !byId.get(pair.id).world).map((pair) => ({
        ...pair, x: pair.x1, y: pair.y1,
        a: [(pair.x1 - width / 2) / focal, (height / 2 - pair.y1) / focal, 1],
        b: [(pair.x2 - width / 2) / focal, (height / 2 - pair.y2) / focal, 1],
      }));
      for (const point of worldPoints(candidates, reference.pose, current.pose, [])) {
        byId.get(point.id).world = point.world;
      }
    }
    if (metrics.tracking !== 'image') {
      for (const track of current.tracks) if (track.world) {
        const camera = mul3(current.pose.R, track.world.map((value, i) => value - current.pose.C[i]));
        if (camera[2] <= 0 || Math.hypot(width / 2 + focal * camera[0] / camera[2] - track.x,
          height / 2 - focal * camera[1] / camera[2] - track.y) > 6) delete track.world;
      }
      this.pose = current.pose;
      if (metrics.displacementPx >= 12 || shared.length < 50) this.keyframe = current;
    } else if (shared.length < 25) this.keyframe = current;
    metrics.landmarks = current.tracks.filter((track) => track.world).length;
    const message = metrics.tracking === 'anchored' ? `3D pose locked - ${followed} visual tracks`
      : metrics.tracking === 'relative' ? `3D pose initialized - ${followed} visual tracks`
        : `Tracking ${followed} points - move sideways for 3D pose`;
    return result(message, metrics.tracking === 'image' ? null : current.pose);
  }
}
