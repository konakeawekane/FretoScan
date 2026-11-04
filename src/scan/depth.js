import { mul3, multiply3, transpose3 } from './reconstruction.js';

function sample(image, width, height, x, y) {
  if (x < 1 || y < 1 || x >= width - 2 || y >= height - 2) return -1;
  const ix = Math.floor(x), iy = Math.floor(y), u = x - ix, v = y - iy;
  const a = image[iy * width + ix], b = image[iy * width + ix + 1];
  const c = image[(iy + 1) * width + ix], d = image[(iy + 1) * width + ix + 1];
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}

export function estimateDenseDepth(reference, neighbors, landmarks, focal, step = 2) {
  const { width, height, gray, pose } = reference;
  const outWidth = Math.floor(width / step), outHeight = Math.floor(height / step);
  const depths = new Float32Array(outWidth * outHeight);
  const confidence = new Float32Array(depths.length);
  if (!neighbors.length || landmarks.length < 8) return { width: outWidth, height: outHeight, step, depths, confidence };

  const sparse = landmarks.map(({ x, y, world }) => ({ x, y, depth: mul3(pose.R, world.map((value, i) => value - pose.C[i]))[2] }))
    .filter(({ depth }) => depth > 0.15 && depth < 20).sort((a, b) => a.depth - b.depth);
  if (sparse.length < 8) return { width: outWidth, height: outHeight, step, depths, confidence };
  const near = Math.max(0.15, sparse[Math.floor(sparse.length * 0.1)].depth * 0.65);
  const far = Math.min(20, sparse[Math.floor(sparse.length * 0.9)].depth * 1.5);
  const views = neighbors.map((view) => ({
    gray: view.gray,
    R: multiply3(view.pose.R, transpose3(pose.R)),
    t: mul3(view.pose.R, pose.C.map((value, i) => value - view.pose.C[i])),
  }));
  const offsets = [[0, 0], [-2, 0], [2, 0], [0, -2], [0, 2], [-2, -2], [2, -2], [-2, 2], [2, 2]];

  for (let oy = 2; oy < outHeight - 2; oy++) for (let ox = 2; ox < outWidth - 2; ox++) {
    const x = ox * step, y = oy * step;
    const referenceValues = offsets.map(([dx, dy]) => gray[(y + dy) * width + x + dx]);
    if (Math.max(...referenceValues) - Math.min(...referenceValues) < 12) continue;
    const referenceMean = referenceValues.reduce((sum, value) => sum + value, 0) / offsets.length;
    const centered = referenceValues.map((value) => value - referenceMean);
    const referenceVariance = centered.reduce((sum, value) => sum + value * value, 0);
    if (referenceVariance < 150) continue;
    let closest = null, distance = 35 * 35;
    for (const point of sparse) {
      const squared = (point.x - x) ** 2 + (point.y - y) ** 2;
      if (squared < distance) { closest = point; distance = squared; }
    }
    const lower = closest ? Math.max(near, closest.depth * 0.65) : near;
    const upper = closest ? Math.min(far, closest.depth * 1.4) : far;
    if (upper <= lower * 1.05) continue;
    const costs = new Float32Array(28).fill(2);
    let best = Infinity, bestLevel = -1;
    for (let level = 0; level < 28; level++) {
      const inverse = 1 / lower + (1 / upper - 1 / lower) * level / 27;
      const candidate = 1 / inverse;
      const ray = [(x - width / 2) / focal * candidate, (height / 2 - y) / focal * candidate, candidate];
      let cost = 0, usable = 0;
      for (const view of views) {
        const point = mul3(view.R, ray).map((value, i) => value + view.t[i]);
        if (point[2] <= 0) continue;
        const sx = width / 2 + focal * point[0] / point[2];
        const sy = height / 2 - focal * point[1] / point[2];
        if (sx < 4 || sy < 4 || sx >= width - 4 || sy >= height - 4) continue;
        let sum = 0, squared = 0, cross = 0;
        for (let k = 0; k < offsets.length; k++) {
          const [dx, dy] = offsets[k];
          const value = sample(view.gray, width, height, sx + dx, sy + dy);
          sum += value;
          squared += value * value;
          cross += centered[k] * value;
        }
        const variance = squared - sum * sum / offsets.length;
        if (variance < 150) continue;
        cost += 1 - cross / Math.sqrt(referenceVariance * variance);
        usable++;
      }
      if (!usable) continue;
      costs[level] = cost / usable;
      if (costs[level] < best) { best = costs[level]; bestLevel = level; }
    }
    if (best > 0.38 || bestLevel < 0) continue;
    let second = Infinity;
    for (let level = 0; level < 28; level++) if (Math.abs(level - bestLevel) > 2) second = Math.min(second, costs[level]);
    const separation = (second - best) / Math.max(0.15, second);
    if (separation < (closest ? 0.04 : 0.12)) continue;
    let sublevel = bestLevel;
    if (bestLevel > 0 && bestLevel < 27) {
      const a = costs[bestLevel - 1], b = costs[bestLevel], c = costs[bestLevel + 1];
      const denominator = a - 2 * b + c;
      if (denominator > 1e-5) sublevel += Math.max(-0.5, Math.min(0.5, (a - c) / (2 * denominator)));
    }
    const depth = 1 / (1 / lower + (1 / upper - 1 / lower) * sublevel / 27);
    const index = oy * outWidth + ox;
    depths[index] = depth;
    confidence[index] = Math.min(1, Math.max(0.2, (0.4 - best) * 2 + separation)) * (closest ? 1 : 0.8);
  }
  const filtered = new Float32Array(confidence);
  for (let y = 2; y < outHeight - 2; y++) for (let x = 2; x < outWidth - 2; x++) {
    const index = y * outWidth + x;
    if (!depths[index]) continue;
    let supporting = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const neighbor = depths[index + dy * outWidth + dx];
      if (neighbor && Math.abs(neighbor - depths[index]) < Math.max(0.03, depths[index] * 0.08)) supporting++;
    }
    if (supporting < 2) filtered[index] = 0;
  }
  confidence.set(filtered);
  return { width: outWidth, height: outHeight, step, depths, confidence };
}
