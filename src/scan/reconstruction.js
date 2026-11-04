const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (v) => Math.hypot(...v);
const unit = (v) => v.map((x) => x / (norm(v) || 1));
export const mul3 = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
export const transpose3 = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
export const multiply3 = (a, b) => {
  const t = transpose3(b);
  return Array.from({ length: 9 }, (_, i) => dot(a.slice(Math.floor(i / 3) * 3, Math.floor(i / 3) * 3 + 3), t.slice(i % 3 * 3, i % 3 * 3 + 3)));
};

function eigen(matrix, n) {
  const a = Float64Array.from(matrix);
  const vectors = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vectors[i * n + i] = 1;
  for (let step = 0; step < n * n * 8; step++) {
    let p = 0, q = 1, largest = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const value = Math.abs(a[i * n + j]);
      if (value > largest) { largest = value; p = i; q = j; }
    }
    if (largest < 1e-11) break;
    const apq = a[p * n + q];
    const tau = (a[q * n + q] - a[p * n + p]) / (2 * apq);
    const t = (tau < 0 ? -1 : 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = a[p * n + k] = c * akp - s * akq;
        a[k * n + q] = a[q * n + k] = s * akp + c * akq;
      }
      const vkp = vectors[k * n + p], vkq = vectors[k * n + q];
      vectors[k * n + p] = c * vkp - s * vkq;
      vectors[k * n + q] = s * vkp + c * vkq;
    }
    a[p * n + p] -= t * apq;
    a[q * n + q] += t * apq;
    a[p * n + q] = a[q * n + p] = 0;
  }
  return Array.from({ length: n }, (_, i) => ({ value: a[i * n + i], vector: Array.from({ length: n }, (_, j) => vectors[j * n + i]) }));
}

function fit(matches) {
  const ata = new Float64Array(81);
  for (const { a, b } of matches) {
    const row = [b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1];
    for (let i = 0; i < 9; i++) for (let j = i; j < 9; j++) ata[i * 9 + j] += row[i] * row[j];
  }
  for (let i = 0; i < 9; i++) for (let j = 0; j < i; j++) ata[i * 9 + j] = ata[j * 9 + i];
  return unit(eigen(ata, 9).sort((a, b) => a.value - b.value)[0].vector);
}

function error(e, { a, b }) {
  const ea = mul3(e, a), eb = mul3(transpose3(e), b);
  const residual = dot(b, ea);
  return residual * residual / (ea[0] ** 2 + ea[1] ** 2 + eb[0] ** 2 + eb[1] ** 2 + 1e-12);
}

function poseCandidates(e) {
  const et = transpose3(e);
  const values = eigen(multiply3(et, e), 3).sort((a, b) => b.value - a.value);
  const v1 = unit(values[0].vector), v2 = unit(values[1].vector);
  const v3 = unit(cross(v1, v2));
  const u1 = unit(mul3(e, v1));
  const raw = mul3(e, v2);
  const u2 = unit(raw.map((x, i) => x - dot(raw, u1) * u1[i]));
  const u3 = unit(cross(u1, u2));
  const u = [u1[0], u2[0], u3[0], u1[1], u2[1], u3[1], u1[2], u2[2], u3[2]];
  const v = [v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]];
  const w = [0, -1, 0, 1, 0, 0, 0, 0, 1];
  const rotations = [w, transpose3(w)].map((matrix) => multiply3(multiply3(u, matrix), transpose3(v)));
  return rotations.flatMap((R) => [{ R, t: u3 }, { R, t: u3.map((x) => -x) }]);
}

export function triangulate(a, b, R, t) {
  const ra = mul3(R, a);
  const aa = dot(ra, ra), bb = dot(b, b), ab = dot(ra, b);
  const denominator = aa * bb - ab * ab;
  if (denominator < 1e-7) return null;
  const at = dot(ra, t), bt = dot(b, t);
  const d1 = (-at * bb + ab * bt) / denominator;
  const d2 = (aa * bt - ab * at) / denominator;
  if (d1 <= 0.1 || d2 <= 0.1 || d1 > 60 || d2 > 60) return null;
  const point = a.map((x) => x * d1);
  const p2 = mul3(R, point).map((x, i) => x + t[i]);
  if (Math.hypot(p2[0] / p2[2] - b[0], p2[1] / p2[2] - b[1]) > 0.015) return null;
  return point;
}

export function estimatePose(pairs, width, height, random = Math.random) {
  if (pairs.length < 20) return null;
  const focal = Math.max(width, height) * 0.9;
  const matches = pairs.map(({ x1, y1, x2, y2 }) => ({
    x: x1, y: y1,
    a: [(x1 - width / 2) / focal, (height / 2 - y1) / focal, 1],
    b: [(x2 - width / 2) / focal, (height / 2 - y2) / focal, 1],
  }));
  const threshold = (2.5 / focal) ** 2;
  let best = [];
  for (let iteration = 0; iteration < 120; iteration++) {
    const sample = new Set();
    while (sample.size < 8) sample.add(Math.floor(random() * matches.length));
    const e = fit(Array.from(sample, (i) => matches[i]));
    const inliers = matches.filter((match) => error(e, match) < threshold);
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < 20 || best.length < matches.length * 0.4) return null;
  const e = fit(best);
  const inliers = matches.filter((match) => error(e, match) < threshold);
  let result = null, count = 0;
  for (const candidate of poseCandidates(e)) {
    const valid = inliers.map((match) => ({ ...match, point: triangulate(match.a, match.b, candidate.R, candidate.t) })).filter((match) => match.point);
    if (valid.length > count) { count = valid.length; result = { ...candidate, points: valid, focal, inliers: inliers.length }; }
  }
  if (count < 16) return null;
  const parallax = result.points.map(({ a, b }) => {
    const rotated = mul3(result.R, a);
    return Math.hypot(rotated[0] / rotated[2] - b[0], rotated[1] / rotated[2] - b[1]);
  }).sort((a, b) => a - b);
  return parallax[Math.floor(parallax.length / 2)] > 2 / focal ? result : null;
}

function circle(a, b, c) {
  const d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
  if (Math.abs(d) < 1e-6) return null;
  const aa = a.x * a.x + a.y * a.y, bb = b.x * b.x + b.y * b.y, cc = c.x * c.x + c.y * c.y;
  const x = (aa * (b.y - c.y) + bb * (c.y - a.y) + cc * (a.y - b.y)) / d;
  const y = (aa * (c.x - b.x) + bb * (a.x - c.x) + cc * (b.x - a.x)) / d;
  return { x, y, r2: (x - a.x) ** 2 + (y - a.y) ** 2 };
}

export function meshFaces(points, width, height) {
  if (points.length < 3) return [];
  const all = [...points, { x: -width * 10, y: -height * 10 }, { x: width * 10, y: -height * 10 }, { x: 0, y: height * 10 }];
  const n = points.length;
  let triangles = [{ ids: [n, n + 1, n + 2], circle: circle(all[n], all[n + 1], all[n + 2]) }];
  for (let i = 0; i < n; i++) {
    const edges = new Map();
    triangles = triangles.filter((triangle) => {
      const c = triangle.circle;
      if (!c || (points[i].x - c.x) ** 2 + (points[i].y - c.y) ** 2 > c.r2) return true;
      const [a, b, d] = triangle.ids;
      for (const edge of [[a, b], [b, d], [d, a]]) {
        const key = edge.slice().sort((x, y) => x - y).join(':');
        edges.set(key, edges.has(key) ? null : edge);
      }
      return false;
    });
    for (const edge of edges.values()) if (edge) {
      const ids = [...edge, i];
      triangles.push({ ids, circle: circle(...ids.map((id) => all[id])) });
    }
  }
  return triangles.map(({ ids }) => ids).filter((ids) => ids.every((id) => id < n) && ids.every((id, k) => {
    const a = points[id], b = points[ids[(k + 1) % 3]];
    return Math.hypot(a.x - b.x, a.y - b.y) < Math.max(width, height) * 0.18 && Math.abs(a.point[2] - b.point[2]) < Math.min(a.point[2], b.point[2]) * 0.4;
  }));
}

export const initialPose = () => ({ R: [...identity], C: [0, 0, 0] });
export function advancePose(previous, relative, scale = 0.2) {
  const R = multiply3(relative.R, previous.R);
  const offset = mul3(transpose3(R), relative.t).map((x) => -x * scale);
  return { R, C: previous.C.map((x, i) => x + offset[i]) };
}
export function toWorld(point, pose, scale = 0.2) {
  const rotated = mul3(transpose3(pose.R), point);
  return rotated.map((x, i) => pose.C[i] + x * scale);
}
