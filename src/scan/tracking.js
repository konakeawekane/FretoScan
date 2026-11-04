import { mul3, multiply3, transpose3, triangulate, toWorld } from './reconstruction.js';

function rotation([x, y, z]) {
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  x /= angle; y /= angle; z /= angle;
  const c = Math.cos(angle), s = Math.sin(angle), v = 1 - c;
  return [c + x * x * v, x * y * v - z * s, x * z * v + y * s,
    y * x * v + z * s, c + y * y * v, y * z * v - x * s,
    z * x * v - y * s, z * y * v + x * s, c + z * z * v];
}

function solve(matrix, vector) {
  const n = vector.length;
  const a = Array.from({ length: n }, (_, row) => [...matrix.slice(row * n, row * n + n), vector[row]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    if (Math.abs(a[pivot][column]) < 1e-10) return null;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column];
    for (let j = column; j <= n; j++) a[column][j] /= divisor;
    for (let row = 0; row < n; row++) if (row !== column) {
      const factor = a[row][column];
      for (let j = column; j <= n; j++) a[row][j] -= factor * a[column][j];
    }
  }
  return a.map((row) => row[n]);
}

export function refinePose(observations, initial, focal, width, height) {
  if (observations.length < 8) return null;
  let R = [...initial.R];
  let t = mul3(R, initial.C).map((x) => -x);
  for (let iteration = 0; iteration < 10; iteration++) {
    const h = new Float64Array(36), g = new Float64Array(6);
    let used = 0;
    for (const { point, x, y } of observations) {
      const [px, py, pz] = mul3(R, point).map((value, index) => value + t[index]);
      if (pz < 0.1) continue;
      const u = px / pz, v = py / pz;
      const rx = (x - width / 2) / focal - u;
      const ry = (height / 2 - y) / focal - v;
      const residual = Math.hypot(rx, ry);
      if (residual > 20 / focal) continue;
      const weight = Math.min(1, (3 / focal) / Math.max(residual, 1e-9));
      const inv = 1 / pz, inv2 = inv * inv;
      const rows = [
        [-px * py * inv2, 1 + px * px * inv2, -py * inv, inv, 0, -px * inv2],
        [-1 - py * py * inv2, px * py * inv2, px * inv, 0, inv, -py * inv2],
      ];
      for (let k = 0; k < 2; k++) for (let i = 0; i < 6; i++) {
        g[i] += weight * rows[k][i] * [rx, ry][k];
        for (let j = i; j < 6; j++) h[i * 6 + j] += weight * rows[k][i] * rows[k][j];
      }
      used++;
    }
    if (used < 8) return null;
    for (let i = 0; i < 6; i++) {
      h[i * 6 + i] += 1e-6;
      for (let j = 0; j < i; j++) h[i * 6 + j] = h[j * 6 + i];
    }
    const delta = solve(h, g);
    if (!delta) return null;
    const dR = rotation(delta.slice(0, 3));
    R = multiply3(dR, R);
    t = mul3(dR, t).map((value, index) => value + delta[index + 3]);
    if (Math.hypot(...delta) < 1e-6) break;
  }
  const C = mul3(transpose3(R), t).map((value) => -value);
  let inliers = 0, squared = 0;
  for (const { point, x, y } of observations) {
    const [px, py, pz] = mul3(R, point).map((value, index) => value + t[index]);
    if (pz <= 0) continue;
    const error = Math.hypot(focal * px / pz + width / 2 - x, height / 2 - focal * py / pz - y);
    if (error < 4) { inliers++; squared += error * error; }
  }
  if (inliers < 8 || inliers < observations.length * 0.5) return null;
  return { R, C, inliers, errorPx: Math.sqrt(squared / inliers) };
}

export function relativePose(previous, next) {
  const R = multiply3(next.R, transpose3(previous.R));
  const t = mul3(next.R, previous.C.map((value, index) => value - next.C[index]));
  return { R, t };
}

export function worldPoints(matches, previous, next, fallback) {
  const relative = relativePose(previous, next);
  const points = [];
  for (const match of matches) {
    const point = triangulate(match.a, match.b, relative.R, relative.t);
    if (!point) continue;
    points.push({ ...match, world: toWorld(point, previous, 1), point });
  }
  return points.length >= 8 ? points : fallback.map((match) => ({ ...match, world: toWorld(match.point, previous) }));
}
