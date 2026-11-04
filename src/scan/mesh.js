function compact(mesh, keepFace) {
  const remap = new Int32Array(mesh.positions.length / 3).fill(-1);
  const positions = [], colors = [], indices = [];
  for (let face = 0; face < mesh.indices.length / 3; face++) {
    if (!keepFace(face)) continue;
    for (let k = 0; k < 3; k++) {
      const vertex = mesh.indices[face * 3 + k];
      if (remap[vertex] < 0) {
        remap[vertex] = positions.length / 3;
        positions.push(mesh.positions[vertex * 3], mesh.positions[vertex * 3 + 1], mesh.positions[vertex * 3 + 2]);
        colors.push(mesh.colors[vertex * 3], mesh.colors[vertex * 3 + 1], mesh.colors[vertex * 3 + 2]);
      }
      indices.push(remap[vertex]);
    }
  }
  return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), indices: Uint32Array.from(indices) };
}

export function removeSmallComponents(mesh, fraction = 0.02, minimum = 30) {
  const parent = Int32Array.from({ length: mesh.positions.length / 3 }, (_, i) => i);
  const root = (i) => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  const { indices } = mesh;
  for (let i = 0; i < indices.length; i += 3) {
    const a = root(indices[i]), b = root(indices[i + 1]), c = root(indices[i + 2]);
    parent[b] = a;
    parent[root(c)] = a;
  }
  const faces = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const r = root(indices[i]);
    faces.set(r, (faces.get(r) || 0) + 1);
  }
  const threshold = Math.max(minimum, Math.max(0, ...faces.values()) * fraction);
  return compact(mesh, (face) => faces.get(root(indices[face * 3])) >= threshold);
}

export function smooth(mesh, iterations = 3) {
  const { positions, indices } = mesh;
  const count = positions.length / 3;
  const sum = new Float32Array(positions.length), degree = new Float32Array(count);
  const pass = (factor) => {
    sum.fill(0);
    degree.fill(0);
    for (let i = 0; i < indices.length; i += 3) for (let k = 0; k < 3; k++) {
      const a = indices[i + k], b = indices[i + (k + 1) % 3];
      for (let axis = 0; axis < 3; axis++) { sum[a * 3 + axis] += positions[b * 3 + axis]; sum[b * 3 + axis] += positions[a * 3 + axis]; }
      degree[a]++;
      degree[b]++;
    }
    for (let v = 0; v < count; v++) if (degree[v]) {
      for (let axis = 0; axis < 3; axis++) {
        const i = v * 3 + axis;
        positions[i] += factor * (sum[i] / degree[v] - positions[i]);
      }
    }
  };
  for (let i = 0; i < iterations; i++) { pass(0.5); pass(-0.53); }
  return mesh;
}

function cluster(mesh, size) {
  const ids = new Map();
  const remap = new Int32Array(mesh.positions.length / 3);
  const sums = [];
  for (let v = 0; v < remap.length; v++) {
    const key = `${Math.floor(mesh.positions[v * 3] / size)}:${Math.floor(mesh.positions[v * 3 + 1] / size)}:${Math.floor(mesh.positions[v * 3 + 2] / size)}`;
    let id = ids.get(key);
    if (id === undefined) { ids.set(key, id = sums.length); sums.push([0, 0, 0, 0, 0, 0, 0]); }
    remap[v] = id;
    const entry = sums[id];
    for (let k = 0; k < 3; k++) { entry[k] += mesh.positions[v * 3 + k]; entry[k + 3] += mesh.colors[v * 3 + k]; }
    entry[6]++;
  }
  const seen = new Set();
  const indices = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const a = remap[mesh.indices[i]], b = remap[mesh.indices[i + 1]], c = remap[mesh.indices[i + 2]];
    if (a === b || b === c || a === c) continue;
    const key = [a, b, c].sort((x, y) => x - y).join(':');
    if (seen.has(key)) continue;
    seen.add(key);
    indices.push(a, b, c);
  }
  const positions = new Float32Array(sums.length * 3), colors = new Float32Array(sums.length * 3);
  sums.forEach((entry, id) => {
    for (let k = 0; k < 3; k++) { positions[id * 3 + k] = entry[k] / entry[6]; colors[id * 3 + k] = entry[k + 3] / entry[6]; }
  });
  return compact({ positions, colors, indices: Uint32Array.from(indices) }, () => true);
}

export function simplify(mesh, limit, size) {
  let result = mesh;
  for (let attempt = 0; result.indices.length / 3 > limit && attempt < 12; attempt++, size *= 1.3) result = cluster(mesh, size);
  return result;
}

export function vertexNormals({ positions, indices }) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
    for (const vertex of [a, b, c]) for (let k = 0; k < 3; k++) normals[vertex + k] += n[k];
  }
  for (let i = 0; i < normals.length; i += 3) {
    const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= length; normals[i + 1] /= length; normals[i + 2] /= length;
  }
  return normals;
}

export function cleanMesh(mesh, { limit, voxel }) {
  if (!mesh.indices.length) return mesh;
  let result = removeSmallComponents(mesh);
  smooth(result, 3);
  if (result.indices.length / 3 > limit) result = simplify(result, limit, voxel * 1.5);
  return result;
}

export function expand({ positions, colors, indices }) {
  const outPositions = new Float32Array(indices.length * 3), outColors = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i] * 3;
    outPositions[i * 3] = positions[v];
    outPositions[i * 3 + 1] = positions[v + 1];
    outPositions[i * 3 + 2] = -positions[v + 2];
    outColors.set(colors.subarray(v, v + 3), i * 3);
  }
  return { positions: outPositions, colors: outColors };
}
